"""Run a traced PyTorch TrainJob on Kubernetes through the Kubeflow SDK's Kubernetes backend.

Setup is described in docs/kubernetes.md (kind cluster, Kubeflow Trainer v2.3.0, the
torch-otel runtime and Jaeger). Then, in another terminal:

    kubectl -n observability port-forward svc/otel-collector 4317:4317
    kubectl -n observability port-forward svc/jaeger 16686:16686

and run:

    python examples/kubernetes/run_on_kind.py
"""

from __future__ import annotations

import json
import sys
import time
import urllib.parse
import urllib.request

from kubeflow.common.types import KubernetesBackendConfig
from kubeflow.trainer import CustomTrainer, TrainerClient
from kubeflow.trainer.options.common import Name

import kubeflow_otel
from kubeflow_otel import TelemetryConfig, TrainerClientInstrumentor

CLIENT_COLLECTOR = "http://localhost:4317"  # the client reaches the collector via port-forward
POD_COLLECTOR = "http://otel-collector.observability.svc.cluster.local:4317"  # pods: cluster DNS
JAEGER_API = "http://localhost:16686"


def train_small(params=None, **kwargs):
    """Runs inside the training pod (imports must live inside the function).

    Accepts both calling conventions: the local-process backend calls train(func_args)
    with a dict, while the Kubernetes backend calls train(**func_args).
    """
    import torch

    from kubeflow_otel.training import training_run

    params = params or kwargs
    torch.manual_seed(0)
    x = torch.randn(4000, 20)
    y = ((x @ torch.randn(20)) > 0).long()
    model = torch.nn.Sequential(torch.nn.Linear(20, 64), torch.nn.ReLU(), torch.nn.Linear(64, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    with training_run() as run:
        for epoch in range(params["epochs"]):
            with run.epoch(epoch):
                for i in range(0, 3200, 128):
                    optimizer.zero_grad()
                    loss = loss_fn(model(x[i : i + 128]), y[i : i + 128])
                    loss.backward()
                    optimizer.step()
                with torch.no_grad():
                    accuracy = (model(x[3200:]).argmax(1) == y[3200:]).float().mean().item()
                run.log_metrics({"loss": loss.item(), "accuracy": accuracy})
                print(f"epoch {epoch}: loss={loss.item():.4f} accuracy={accuracy:.3f}", flush=True)


def find_trace(job_name: str, timeout: float = 90) -> dict | None:
    query = urllib.parse.urlencode(
        {"service": "kubeflow-sdk", "tags": json.dumps({"kubeflow.trainjob.name": job_name})}
    )
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{JAEGER_API}/api/traces?{query}", timeout=10) as resp:
            traces = json.load(resp)["data"]
        for trace in traces:
            services = {p["serviceName"] for p in trace["processes"].values()}
            operations = {s["operationName"] for s in trace["spans"]}
            # client spans that end last (wait, logs) are exported at shutdown, so wait for them
            if "kubeflow-trainjob" in services and "kubeflow.trainer.get_job_logs" in operations:
                return trace
        time.sleep(3)
    return None


def main() -> int:
    kubeflow_otel.configure(
        TelemetryConfig(
            enabled=True, otlp_endpoint=CLIENT_COLLECTOR, metric_export_interval_ms=2000
        )
    )
    TrainerClientInstrumentor().instrument()
    client = TrainerClient(backend_config=KubernetesBackendConfig(namespace="default"))

    name = f"k8s-otel-{time.strftime('%H%M%S')}"
    with kubeflow_otel.get_telemetry().tracer.start_span("kubeflow.example.kubernetes"):
        job = client.train(
            runtime="torch-otel",
            trainer=CustomTrainer(
                func=train_small,
                func_args={"epochs": 5},
                num_nodes=1,
                # user-set values win over the injected ones (except the trace context),
                # which is how pods get the in-cluster collector address
                env={"OTEL_EXPORTER_OTLP_ENDPOINT": POD_COLLECTOR},
            ),
            options=[Name(name)],
        )
        print(f"submitted TrainJob {job}")
        final = client.wait_for_job_status(
            job, status={"Complete", "Failed"}, timeout=900, polling_interval=3
        )
        print(f"{job}: {final.status}")
        for line in list(client.get_job_logs(job))[-6:]:
            print("  |", line.rstrip())
    kubeflow_otel.shutdown()

    trace = find_trace(name)
    if trace is None:
        print("FAILED: no trace containing both kubeflow-sdk and kubeflow-trainjob spans")
        return 1
    processes = trace["processes"]
    spans = sorted(trace["spans"], key=lambda s: s["startTime"])
    t0 = spans[0]["startTime"]
    print(f"\ntrace {trace['traceID']}: {len(spans)} spans")
    for span in spans:
        tags = {t["key"]: t["value"] for t in span["tags"]}
        extra = (
            f" epoch={tags['kubeflow.training.epoch']}" if "kubeflow.training.epoch" in tags else ""
        )
        service = processes[span["processID"]]["serviceName"]
        print(
            f"  +{(span['startTime'] - t0) / 1e6:6.1f}s {span['duration'] / 1e6:6.2f}s  "
            f"{service:18s} {span['operationName']}{extra}"
        )
    print("PASS" if final.status == "Complete" else "job did not complete")
    return 0 if final.status == "Complete" else 1


if __name__ == "__main__":
    sys.exit(main())
