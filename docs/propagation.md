# Tracing into training jobs

With client instrumentation alone, a trace ends when `client.train()` returns, at the moment the job is *submitted*. Trace propagation continues that trace **inside the training job**, so one trace shows the whole story:

```
kubeflow-sdk       kubeflow.trainer.train ─────┐                     (client process)
kubeflow-trainjob    └─ kubeflow.trainjob.run ──────────────────────┐ (training process)
                          ├─ kubeflow.trainjob.epoch  epoch=0 ─┐
                          ├─ kubeflow.trainjob.epoch  epoch=1 ──┐
                          └─ kubeflow.trainjob.epoch  epoch=2 ───┐
```

## How it works

1. **Client side (automatic).** While the `kubeflow.trainer.train` span is current, the instrumentation builds a set of environment variables and merges them into a **copy** of the trainer's `env` (your `CustomTrainer` object is never modified):

   | Variable | Value |
   |---|---|
   | `TRACEPARENT` / `TRACESTATE` | W3C Trace Context of the `train` span |
   | `KUBEFLOW_OTEL_ENABLED` | `true` |
   | `KUBEFLOW_OTEL_EXPORTER` | the client's exporter (`otlp` / `console` / `none`) |
   | `OTEL_EXPORTER_OTLP_ENDPOINT` | the client's collector endpoint, if configured |
   | `OTEL_METRIC_EXPORT_INTERVAL` | the client's metric interval |
   | `OTEL_SERVICE_NAME` | `kubeflow-trainjob` |
   | `KUBEFLOW_TRAINJOB_NAME`, `KUBEFLOW_RUNTIME_NAME` | job and runtime names, when known |
   | `OTEL_RESOURCE_ATTRIBUTES` | `kubeflow.trainjob.name=…,kubeflow.runtime.name=…` |

   Values you set yourself in `trainer.env` take precedence, **except** `TRACEPARENT`/`TRACESTATE`, which must describe the current trace.

2. **Transport (the SDK).** Every Kubeflow backend passes `trainer.env` to the training nodes: the local-process backend to the subprocess, the container backend to the containers, and the Kubernetes backend to the pods. One mechanism covers all three.

3. **Job side (one context manager).** Inside your training function, `training_run()` reads the variables, configures telemetry, and starts the job's root span as a child of the client's `train` span.

`TRACEPARENT`/`TRACESTATE` follow OpenTelemetry's convention for using environment variables as context-propagation carriers.

## Using it in a training function

Kubeflow serializes your training function and runs it in a fresh environment, so **imports must be inside the function** and the package must be installed in the job:

```python
def train_func():
    from kubeflow_otel.training import training_run

    with training_run() as run:
        for epoch in range(3):
            with run.epoch(epoch):          # child span + epoch-duration metric
                loss, acc = train_one_epoch()
                run.log_metrics({"loss": loss, "accuracy": acc})

client.train(
    runtime="torch-distributed",
    trainer=CustomTrainer(
        func=train_func,
        packages_to_install=["kubeflow-sdk-otel[otel]"],
    ),
    options=[Name("resnet-lr-0.01")],   # recommended: gives telemetry a readable job name
)
```

`training_run()` is safe everywhere: without the injected variables it starts a new trace, and with telemetry disabled it does nothing.

## What the job emits

**Spans** (service `kubeflow-trainjob`):

| Span | Kind | Attributes |
|---|---|---|
| `kubeflow.trainjob.run` | `SERVER` | `kubeflow.trainjob.name`, `kubeflow.runtime.name`, `kubeflow.training.rank`, `kubeflow.training.world_size`; a `kubeflow.training.metrics` event per `log_metrics()` call |
| `kubeflow.trainjob.epoch` | `INTERNAL` | `kubeflow.training.epoch` |

An exception raised inside an epoch marks both the epoch and the run span `ERROR` (with the stack trace) and is re-raised, so the failing epoch is visible in the trace.

**Metrics:**

| OpenTelemetry name | Prometheus name | Type | Labels |
|---|---|---|---|
| `kubeflow.training.metric` | `kubeflow_training_metric` | gauge | `kubeflow_trainjob_name`, `kubeflow_training_metric_name`, `kubeflow_training_rank` (multi-worker) |
| `kubeflow.training.epoch` | `kubeflow_training_epoch` | gauge | `kubeflow_trainjob_name`, rank |
| `kubeflow.training.epoch.duration` | `kubeflow_training_epoch_duration_seconds` | histogram (s) | `kubeflow_trainjob_name`, rank |

**Cardinality note:** unlike the client metrics, training metrics *are* labeled by job name, because per-job loss curves are their whole purpose. That's one series per job per metric: fine for tens to hundreds of concurrent jobs, but at thousands of jobs you'd rely on Prometheus retention or move per-job curves to an experiment tracker.

## Distributed training

`torchrun` starts one process per worker and sets `RANK` and `WORLD_SIZE` in each. Every worker's `training_run()` joins the same trace (they all inherit the same `TRACEPARENT`) and tags its spans and metrics with its rank, so stragglers are visible side by side.

## Running on Kubernetes

The job exports to `OTEL_EXPORTER_OTLP_ENDPOINT`. On Kubernetes, that endpoint must be reachable **from the training pods**, not just from your laptop: typically an in-cluster collector such as `http://otel-collector.observability:4317`. Set it on the client (the value is forwarded) or directly in `trainer.env`.

## Short-lived processes

Training processes often exit as soon as training ends. `training_run()` therefore flushes buffered spans and metrics when the run finishes (`flush=True` by default), and an `atexit` hook flushes again on interpreter exit.
