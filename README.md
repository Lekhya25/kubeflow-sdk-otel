# kubeflow-sdk-otel

**OpenTelemetry tracing and metrics for the [Kubeflow SDK](https://github.com/kubeflow/sdk)**, with a zero-dependency, near-zero-cost no-op default.

> Community project. Not affiliated with or endorsed by the Kubeflow project.

The Kubeflow SDK lets you submit and manage training jobs (`TrainerClient.train()`, `wait_for_job_status()`, ...) from Python, but gives no visibility into what those calls do, how long they take, or how the training jobs behave once they start. This library adds that visibility **without modifying the SDK**:

- **Off by default, free when off.** The core has no required dependencies. Until telemetry is enabled, every instrument is a no-op and OpenTelemetry is never imported.
- **Standard OpenTelemetry when on.** Spans and metrics go out over OTLP to any OpenTelemetry Collector, so they work with Jaeger, Prometheus, Grafana, or any vendor backend.
- **Plays nicely with existing OpenTelemetry setups.** It never overrides your global providers, and can reuse them if you ask it to.

> 🚧 Work in progress. Implemented so far: the core telemetry layer, `TrainerClient` instrumentation, and trace propagation into training jobs. Coming next: a Prometheus + Jaeger + Grafana stack. See [docs/architecture.md](docs/architecture.md).

## Quick start

```python
from kubeflow.trainer import CustomTrainer, LocalProcessBackendConfig, TrainerClient
from kubeflow_otel import TrainerClientInstrumentor

TrainerClientInstrumentor().instrument()   # or: import kubeflow_otel.auto

client = TrainerClient(backend_config=LocalProcessBackendConfig())
job = client.train(runtime="torch-distributed", trainer=CustomTrainer(func=my_train_fn))
client.wait_for_job_status(job)
```

```bash
KUBEFLOW_OTEL_ENABLED=true OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317 python train.py
```

Every `TrainerClient` call now produces a span (with job status transitions recorded as events while waiting), and the client emits latency, submission, completion and job-duration metrics. See [docs/instrumentation.md](docs/instrumentation.md) for the full list.

### Continue the trace inside the training job

The client passes its trace context to the job through `trainer.env`. Inside the training function:

```python
def my_train_fn():
    from kubeflow_otel.training import training_run

    with training_run() as run:              # joins the trace started by client.train()
        for epoch in range(3):
            with run.epoch(epoch):           # one child span per epoch
                loss = ...
                run.log_metrics({"loss": loss})
```

One trace then spans the notebook's `train()` call and every epoch of the job. See [docs/propagation.md](docs/propagation.md).

## Install

```bash
pip install "kubeflow-sdk-otel[otel]"      # with OpenTelemetry exporters
pip install kubeflow-sdk-otel              # core only: always no-op
```

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `KUBEFLOW_OTEL_ENABLED` | `false` | Master switch. |
| `OTEL_SDK_DISABLED` | `false` | Standard OTel kill switch; overrides the above. |
| `OTEL_SERVICE_NAME` | `kubeflow-sdk` | `service.name` of emitted telemetry. |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | `http://localhost:4317` | Collector endpoint (OTLP/gRPC). |
| `KUBEFLOW_OTEL_EXPORTER` | `otlp` | `otlp`, `console` (stdout) or `none`. |
| `OTEL_METRIC_EXPORT_INTERVAL` | `10000` | Metric push interval in ms. |
| `KUBEFLOW_OTEL_SAMPLE_RATIO` | `1.0` | Fraction of new traces sampled (parent-based). |
| `KUBEFLOW_OTEL_PROPAGATE` | `true` | Propagate trace context into submitted jobs. |
| `KUBEFLOW_OTEL_USE_GLOBAL_PROVIDERS` | `false` | Reuse the application's global OTel providers. |

Invalid values never raise. A warning is logged and the default is used, because telemetry must never break the application it observes.

## Using the telemetry API directly

```python
import kubeflow_otel

telemetry = kubeflow_otel.get_telemetry()        # configured from the environment
with telemetry.tracer.start_span("my-step", attributes={"dataset": "mnist"}):
    ...
telemetry.meter.create_counter("my.events").add(1)
```

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests && ruff format --check src tests
```

## License

Apache-2.0
