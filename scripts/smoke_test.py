"""End-to-end check of the telemetry pipeline: library -> collector -> Jaeger + Prometheus.

    docker compose -f deploy/docker-compose.yml up -d
    python scripts/smoke_test.py

Drives an instrumented ``TrainerClient`` (backed by a stub, so no Kubernetes or
training code is needed) plus a job-side ``training_run``, then polls the Jaeger
and Prometheus HTTP APIs until the resulting trace and metrics show up.
Exits non-zero if they don't arrive within the timeout.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta
from pathlib import Path

from kubeflow.trainer import CustomTrainer, LocalProcessBackendConfig, TrainerClient
from kubeflow.trainer.backends.localprocess.constants import local_runtimes
from kubeflow.trainer.options.common import Name
from kubeflow.trainer.types import types

import kubeflow_otel
from kubeflow_otel import TelemetryConfig, TrainerClientInstrumentor
from kubeflow_otel.training import training_run

DASHBOARD = Path(__file__).resolve().parent.parent / "deploy/grafana/dashboards/kubeflow-sdk.json"
REQUIRED_METRICS = [
    "kubeflow_client_operation_duration_seconds_count",
    "kubeflow_trainjob_submitted_total",
    "kubeflow_trainjob_finished_total",
    "kubeflow_training_metric",
]


class StubBackend:
    """Accepts a job, then reports it Running and Complete while waiting."""

    def __init__(self) -> None:
        self.env: dict[str, str] = {}
        self.job: types.TrainJob | None = None

    def train(self, runtime=None, initializer=None, trainer=None, options=None) -> str:
        self.env = dict(trainer.env or {})
        name = next((o.name for o in options or [] if hasattr(o, "name")), "smoke-job")
        self.job = types.TrainJob(
            name=name,
            runtime=local_runtimes[0],
            steps=[],
            num_nodes=1,
            creation_timestamp=datetime.now() - timedelta(seconds=2),
            status="Created",
        )
        return name

    def wait_for_job_status(self, name, status, timeout, polling_interval, callbacks=None):
        for current in ("Created", "Running", "Complete"):
            self.job.status = current
            for callback in callbacks or []:
                callback(self.job)
            time.sleep(0.05)
        return self.job

    def get_job(self, name):
        return self.job


def get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as resp:
        return json.load(resp)


def wait_for(description: str, check, timeout: float) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if check():
                print(f"  ok   {description}")
                return True
        except OSError:
            pass  # backend not ready yet
        time.sleep(2)
    print(f"  FAIL {description}")
    return False


def check_dashboard_queries(prometheus: str) -> bool:
    """Every PromQL query in the Grafana dashboard must be valid against live Prometheus."""
    grafana_vars = {
        "$__range": "30m",
        "$__rate_interval": "1m",
        "$__interval": "1m",
        "$trainjob": ".*",
    }
    dashboard = json.loads(DASHBOARD.read_text(encoding="utf-8"))
    bad = []
    for panel in dashboard["panels"]:
        for t in panel.get("targets", []):
            if "expr" not in t:
                continue
            query = t["expr"]
            for var, value in grafana_vars.items():
                query = query.replace(var, value)
            try:
                get_json(f"{prometheus}/api/v1/query?" + urllib.parse.urlencode({"query": query}))
            except urllib.error.HTTPError as exc:
                bad.append(f"{panel['title']}: {json.load(exc).get('error')}")
    for line in bad:
        print(f"  FAIL dashboard query: {line}")
    if not bad:
        print("  ok   all dashboard queries are valid PromQL")
    return not bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--otlp", default="http://localhost:4317")
    parser.add_argument("--jaeger", default="http://localhost:16686")
    parser.add_argument("--prometheus", default="http://localhost:9090")
    parser.add_argument("--timeout", type=float, default=90)
    args = parser.parse_args()

    kubeflow_otel.configure(
        TelemetryConfig(enabled=True, otlp_endpoint=args.otlp, metric_export_interval_ms=1000)
    )
    instrumentor = TrainerClientInstrumentor()
    instrumentor.instrument()

    client = TrainerClient(backend_config=LocalProcessBackendConfig())
    client.backend = backend = StubBackend()
    job = client.train(
        runtime="torch-distributed",
        trainer=CustomTrainer(func=lambda: None),
        options=[Name(f"smoke-{int(time.time())}")],
    )
    client.wait_for_job_status(job)
    client.get_job(job)

    # what the training process would do, using the environment the client injected
    with training_run(environ=backend.env, flush=False) as run:
        for epoch in range(3):
            with run.epoch(epoch):
                run.log_metrics({"loss": 1.0 / (epoch + 1)})

    time.sleep(1.5)  # let at least one periodic metric export happen
    kubeflow_otel.shutdown()  # flush spans and metrics
    instrumentor.uninstrument()
    print(f"sent telemetry for job {job}; checking backends...")

    def trace_arrived() -> bool:
        query = urllib.parse.urlencode(
            {"service": "kubeflow-sdk", "tags": json.dumps({"kubeflow.trainjob.name": job})}
        )
        traces = get_json(f"{args.jaeger}/api/traces?{query}")["data"]
        names = {s["operationName"] for t in traces for s in t["spans"]}
        return {"kubeflow.trainer.train", "kubeflow.trainjob.run"} <= names

    ok = wait_for("trace with client and job spans in Jaeger", trace_arrived, args.timeout)
    for metric in REQUIRED_METRICS:
        url = f"{args.prometheus}/api/v1/query?" + urllib.parse.urlencode({"query": metric})
        ok &= wait_for(
            f"{metric} in Prometheus",
            lambda url=url: bool(get_json(url)["data"]["result"]),
            args.timeout,
        )

    def exemplars_arrived() -> bool:
        now = time.time()
        query = urllib.parse.urlencode(
            {
                "query": "kubeflow_client_operation_duration_seconds_bucket",
                "start": now - 600,
                "end": now,
            }
        )
        return bool(get_json(f"{args.prometheus}/api/v1/query_exemplars?{query}")["data"])

    ok &= wait_for("exemplars (metric -> trace links) in Prometheus", exemplars_arrived, 30)
    ok &= check_dashboard_queries(args.prometheus)
    print("PASS" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
