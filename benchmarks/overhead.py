"""Measure what kubeflow-sdk-otel costs: per client call, per training epoch, and at import.

    python benchmarks/overhead.py            # writes benchmarks/RESULTS.md

The client is backed by an in-memory stub so the numbers isolate the
instrumentation itself (a real backend call takes milliseconds to seconds).
"Enabled" uses a real BatchSpanProcessor and metric aggregation, exporting to a
discarding exporter, so it includes everything except network I/O, which happens
on background threads anyway.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import subprocess
import sys
import timeit
from datetime import datetime
from pathlib import Path

from kubeflow.trainer import CustomTrainer, LocalProcessBackendConfig, TrainerClient
from kubeflow.trainer.backends.localprocess.constants import local_runtimes
from kubeflow.trainer.types import types
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter, SpanExportResult

import kubeflow_otel
from kubeflow_otel import TelemetryConfig, TrainerClientInstrumentor
from kubeflow_otel.training import training_run

OUT = Path(__file__).resolve().parent / "RESULTS.md"
REPEAT = 5  # timeit repeats per measurement
ROUNDS = 5  # full rounds; reported values are medians across rounds


class DiscardExporter(SpanExporter):
    def export(self, spans):
        return SpanExportResult.SUCCESS

    def shutdown(self):
        pass


class StubBackend:
    def __init__(self) -> None:
        self.job = types.TrainJob(
            name="bench-job",
            runtime=local_runtimes[0],
            steps=[],
            num_nodes=1,
            creation_timestamp=datetime.now(),
            status="Running",
        )

    def get_job(self, name):
        return self.job

    def train(self, runtime=None, initializer=None, trainer=None, options=None):
        return "bench-job"


def train_fn():
    pass


def per_call_ns(fn, number: int) -> float:
    """Median over REPEAT runs of the mean time per call, in nanoseconds."""
    return statistics.median(timeit.repeat(fn, number=number, repeat=REPEAT)) / number * 1e9


def enable_telemetry() -> None:
    kubeflow_otel.configure(
        TelemetryConfig(enabled=True, service_name="bench"),
        span_processor=BatchSpanProcessor(DiscardExporter(), schedule_delay_millis=100),
        metric_reader=InMemoryMetricReader(),
    )


def client_benchmarks() -> dict[str, dict[str, float]]:
    client = TrainerClient(backend_config=LocalProcessBackendConfig())
    client.backend = StubBackend()
    trainer = CustomTrainer(func=train_fn, env={"A": "1"})
    calls = {
        "get_job": (lambda: client.get_job("bench-job"), 100_000, 5_000),
        "train": (
            lambda: client.train(runtime="torch-distributed", trainer=trainer),
            100_000,
            5_000,
        ),
    }
    results: dict[str, dict[str, float]] = {name: {} for name in calls}

    kubeflow_otel.shutdown()
    for name, (fn, n_fast, _) in calls.items():
        results[name]["baseline"] = per_call_ns(fn, n_fast)

    instrumentor = TrainerClientInstrumentor()
    instrumentor.instrument()
    try:
        kubeflow_otel.configure(TelemetryConfig(enabled=False))
        for name, (fn, n_fast, _) in calls.items():
            results[name]["disabled"] = per_call_ns(fn, n_fast)

        enable_telemetry()
        for name, (fn, _, n_slow) in calls.items():
            fn()  # warm up instrument creation
            results[name]["enabled"] = per_call_ns(fn, n_slow)
    finally:
        instrumentor.uninstrument()
        kubeflow_otel.shutdown()
    return results


def otel_reference_ns() -> float:
    """Bare OpenTelemetry SDK cost of one span plus one histogram record (no library code).

    Measured in the same run, so instrumented costs can be read relative to it;
    ratios are far more stable across machines and runs than absolute times.
    """
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.trace import TracerProvider

    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(
        BatchSpanProcessor(DiscardExporter(), schedule_delay_millis=100)
    )
    tracer = tracer_provider.get_tracer("reference")
    meter_provider = MeterProvider(metric_readers=[InMemoryMetricReader()])
    histogram = meter_provider.get_meter("reference").create_histogram("h", unit="s")
    attrs = {"kubeflow.operation": "get_job", "kubeflow.backend": "stub"}

    def one_span_one_metric() -> None:
        with tracer.start_as_current_span("op", attributes=attrs):
            histogram.record(0.001, attrs)

    one_span_one_metric()
    result = per_call_ns(one_span_one_metric, 5_000)
    tracer_provider.shutdown()
    meter_provider.shutdown()
    return result


def epoch_benchmarks() -> dict[str, float]:
    def plain_epoch() -> None:
        pass  # what a training loop costs without telemetry calls

    def run_epochs(number: int) -> float:
        with training_run(environ={}, flush=False) as run:
            state = {"i": 0}

            def epoch() -> None:
                with run.epoch(state["i"]):
                    run.log_metrics({"loss": 0.5, "accuracy": 0.9})
                state["i"] += 1

            return per_call_ns(epoch, number)

    results = {"baseline": per_call_ns(plain_epoch, 100_000)}
    kubeflow_otel.configure(TelemetryConfig(enabled=False))
    results["disabled"] = run_epochs(100_000)
    enable_telemetry()
    results["enabled"] = run_epochs(5_000)
    kubeflow_otel.shutdown()
    return results


_IMPORT_PROBE = """
import json, sys, time
t0 = time.perf_counter()
import kubeflow.trainer
t1 = time.perf_counter()
import kubeflow_otel
kubeflow_otel.get_telemetry()
t2 = time.perf_counter()
assert not any(m.startswith("opentelemetry") for m in sys.modules)
from kubeflow_otel import TelemetryConfig
kubeflow_otel.configure(TelemetryConfig(enabled=True, exporter="none"))
t3 = time.perf_counter()
print(json.dumps([t1 - t0, t2 - t1, t3 - t2]))
"""


def import_benchmarks(runs: int = 15) -> dict[str, float]:
    """Incremental import costs inside one fresh interpreter, in milliseconds (median).

    Each step is timed in-process with perf_counter after the Kubeflow SDK is
    already imported, so neither interpreter startup nor the (slow, noisy) SDK
    import pollutes the numbers, and stdlib modules the SDK already loaded are
    not charged to kubeflow_otel.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith(("KUBEFLOW_OTEL", "OTEL_"))}
    samples = []
    for _ in range(runs):
        out = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE],
            env=env,
            check=True,
            capture_output=True,
            text=True,
        )
        samples.append(json.loads(out.stdout))
    sdk, ours, enable = (statistics.median(s[i] for s in samples) * 1000 for i in range(3))
    return {
        "Kubeflow SDK import (`import kubeflow.trainer`), for scale": sdk,
        "+ `import kubeflow_otel`, telemetry off (no OpenTelemetry imported)": ours,
        "+ enabling telemetry (imports the OpenTelemetry SDK, builds providers)": enable,
    }


def fmt_ns(ns: float) -> str:
    return f"{ns / 1000:.2f} µs" if ns >= 1000 else f"{ns:.0f} ns"


def render(
    client: dict, reference: float, epoch: dict, ratios: dict[str, float], imports: dict
) -> str:
    def ratio(key: str) -> str:
        return f"{ratios[key]:.1f}x"

    lines = [
        "# Overhead benchmark results",
        "",
        "Generated by `python benchmarks/overhead.py`. See "
        "[docs/performance.md](../docs/performance.md) for methodology and interpretation.",
        "",
        f"- Python {platform.python_version()} on {platform.system()} {platform.release()}"
        f" ({platform.machine()}), {os.cpu_count()} logical CPUs",
        f"- Processor: {platform.processor() or 'unknown'}",
        f"- Each value is the median over {ROUNDS} rounds (each the median of {REPEAT} timeit runs)"
        " of the mean time per call. Ratios are computed within each round.",
        f"- **Reference:** the bare OpenTelemetry SDK (one span + one histogram record, no"
        f' library code) costs **{fmt_ns(reference)}** in this run. "vs OTel" columns'
        " are relative to it, which is more stable across machines than absolute times.",
        "",
        "## Per client call (instrumented `TrainerClient`, in-memory backend)",
        "",
        "| Call | Uninstrumented | Telemetry off | Added (off) | Telemetry on | vs OTel |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in client.items():
        lines.append(
            f"| `{name}()` | {fmt_ns(r['baseline'])} | {fmt_ns(r['disabled'])} | "
            f"+{fmt_ns(r['disabled'] - r['baseline'])} | {fmt_ns(r['enabled'])} | "
            f"{ratio(name)} |"
        )
    base, off, on = epoch["baseline"], epoch["disabled"], epoch["enabled"]
    lines += [
        "",
        "`train()` with telemetry on also injects trace context into a copy of the trainer.",
        "",
        "## Per training epoch (`run.epoch()` + `log_metrics()` with two metrics)",
        "",
        "| Empty loop body | Telemetry off | Telemetry on | vs OTel |",
        "|---|---|---|---|",
        f"| {fmt_ns(base)} | {fmt_ns(off)} | {fmt_ns(on)} | {ratio('epoch')} |",
        "",
        "## Import cost (incremental, in-process, median of 15 fresh interpreters)",
        "",
        "| Step | Median |",
        "|---|---|",
    ]
    for label, ms in imports.items():
        lines.append(f"| {label} | {ms:.0f} ms |")
    return "\n".join(lines) + "\n"


def main() -> None:
    med = statistics.median
    rounds = []
    for i in range(ROUNDS):
        print(f"round {i + 1}/{ROUNDS}...", flush=True)
        rounds.append((client_benchmarks(), otel_reference_ns(), epoch_benchmarks()))
    names = rounds[0][0].keys()
    client = {n: {k: med(r[0][n][k] for r in rounds) for k in rounds[0][0][n]} for n in names}
    reference = med(r[1] for r in rounds)
    epoch = {k: med(r[2][k] for r in rounds) for k in rounds[0][2]}
    ratios = {n: med(r[0][n]["enabled"] / r[1] for r in rounds) for n in names}
    ratios["epoch"] = med(r[2]["enabled"] / r[1] for r in rounds)
    print("imports...", flush=True)
    imports = import_benchmarks()
    report = render(client, reference, epoch, ratios, imports)
    OUT.write_text(report, encoding="utf-8", newline="\n")
    sys.stdout.reconfigure(encoding="utf-8")  # the report contains "µs"
    print(report)


if __name__ == "__main__":
    main()
