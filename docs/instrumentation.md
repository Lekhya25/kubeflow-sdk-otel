# TrainerClient instrumentation

`TrainerClientInstrumentor` adds a span and metrics to every public `kubeflow.trainer.TrainerClient` operation, without modifying the SDK.

## Enabling it

```python
from kubeflow_otel import TrainerClientInstrumentor

TrainerClientInstrumentor().instrument()   # once, at startup
```

or, with no code changes beyond an import:

```python
import kubeflow_otel.auto  # noqa: F401
```

Telemetry itself is still switched on by `KUBEFLOW_OTEL_ENABLED=true` (see the [README](../README.md#configuration)). With it off, wrapped methods delegate straight to the originals.

`uninstrument()` restores the original methods. Instrumenting twice is a no-op with a warning.

## How it works

`instrument()` replaces each method **on the `TrainerClient` class** with a wrapper and keeps a reference to the original on the wrapper (`_kubeflow_otel_original`). Patching the class rather than an instance means every client is covered, including clients created inside other libraries.

On each call the wrapper:

1. Fetches the active telemetry (`get_telemetry()`). If it's disabled, it calls the original immediately. This check happens **per call, not at instrument time**, so telemetry can be configured after instrumenting.
2. Binds the call's arguments to the method signature (`inspect.Signature.bind`), so attributes are read correctly whether arguments were passed by position or by keyword.
3. Starts a `CLIENT` span, calls the original, records the outcome, and records the operation-duration histogram.

Every telemetry hook runs inside a guard: if recording fails (a broken exporter, an unexpected argument shape), the failure is logged at debug level and **the user's call proceeds unaffected**.

## Spans

| Span name | Notable attributes |
|---|---|
| `kubeflow.trainer.train` | `kubeflow.runtime.name`, `kubeflow.trainer.type`, `kubeflow.trainjob.num_nodes`, `kubeflow.trainjob.name` (from a `Name` option up front, otherwise set when `train()` returns) |
| `kubeflow.trainer.get_job` | `kubeflow.trainjob.name`, `kubeflow.trainjob.status`, `kubeflow.trainjob.num_nodes`, `kubeflow.runtime.name` |
| `kubeflow.trainer.list_jobs` | `kubeflow.runtime.name` (filter), `kubeflow.result.count` |
| `kubeflow.trainer.wait_for_job_status` | `kubeflow.trainjob.name`, `kubeflow.wait.target_status`, `kubeflow.wait.timeout`, `kubeflow.wait.polling_interval`; one **`kubeflow.trainjob.status_change` event per status transition** observed while waiting |
| `kubeflow.trainer.get_job_logs` | `kubeflow.trainjob.name`, `kubeflow.logs.step`, `kubeflow.logs.follow`, `kubeflow.logs.lines` |
| `kubeflow.trainer.get_job_events` | `kubeflow.trainjob.name`, `kubeflow.result.count` |
| `kubeflow.trainer.delete_job` | `kubeflow.trainjob.name` |
| `kubeflow.trainer.list_runtimes` / `get_runtime` / `get_runtime_packages` | `kubeflow.runtime.name`, `kubeflow.result.count` |

Every span also carries `kubeflow.operation` and `kubeflow.backend` (`kubernetes`, `localprocess` or `container`). Failed calls get status `ERROR`, an `exception` event, and `error.type`; the exception is re-raised unchanged.

### Status transitions while waiting

`wait_for_job_status` polls the backend and invokes its `callbacks` on every poll (on all three backends). The instrumentation prepends its own observer to a **copy** of the caller's callback list, so the caller's list is never mutated and the caller's callbacks still see every poll. The observer emits a span event whenever the status changes (e.g. `Created → Running → Complete`). In Jaeger these show up as markers on the wait span's timeline.

### Streaming logs

`get_job_logs` returns an iterator, and with `follow=True` it streams until the job ends. The span therefore covers the **consumption** of the iterator and ends when it's exhausted, closed or fails. It's deliberately *not* made the current span: a generator hands control back to the caller between lines, and making its span current would wrongly parent the caller's unrelated work under it.

## Metrics

| OpenTelemetry name | Prometheus name | Type | Labels | Meaning |
|---|---|---|---|---|
| `kubeflow.client.operation.duration` | `kubeflow_client_operation_duration_seconds` | histogram (s) | `kubeflow_operation`, `kubeflow_backend`, `error_type` (failures only) | Latency of each client operation. Streaming `get_job_logs(follow=True)` is excluded, since its duration is the job's lifetime, not a latency. |
| `kubeflow.trainjob.submitted` | `kubeflow_trainjob_submitted_total` | counter | `kubeflow_backend`, `kubeflow_runtime_name`, `kubeflow_trainer_type` | Jobs successfully submitted with `train()`. |
| `kubeflow.trainjob.finished` | `kubeflow_trainjob_finished_total` | counter | `kubeflow_backend`, `kubeflow_runtime_name`, `kubeflow_trainjob_status` | Jobs observed reaching `Complete` or `Failed`, **counted once per job**. |
| `kubeflow.trainjob.duration` | `kubeflow_trainjob_duration_seconds` | histogram (s) | same as above | Creation → first observed terminal status. |
| `kubeflow.trainjob.startup.duration` | `kubeflow_trainjob_startup_duration_seconds` | histogram (s) | `kubeflow_backend`, `kubeflow_runtime_name`, `kubeflow_trainjob_status` | Creation → first observed `Running` (scheduling/queueing time on Kubernetes). |

### Notes and trade-offs

- **"Observed" semantics.** The client only learns a job's status when someone asks (via `get_job`, `list_jobs` or `wait_for_job_status`). Job duration and startup are therefore *as observed*: accurate to within one polling interval when using `wait_for_job_status`, and missing for jobs nobody polls. Job-side telemetry (trace propagation into the job, next commit) gives exact timings.
- **Counted once.** A bounded record of `(backend, job name)` pairs (up to 10,000, least-recently-added evicted) prevents a finished job from being counted again each time it's listed.
- **Low-cardinality labels.** Job names appear on spans but never as metric labels. Each unique label value creates a new time series in Prometheus, and per-job series grow without bound.
- **Bucket boundaries** are chosen per metric, in seconds (see `kubeflow_otel/semconv.py`): client operations 5 ms – 10 min, job durations 1 s – 24 h.

All names are defined in [`kubeflow_otel/semconv.py`](../src/kubeflow_otel/semconv.py).
