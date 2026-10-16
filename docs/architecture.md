# Architecture

This document explains how `kubeflow-sdk-otel` is put together and why.

## Goals

1. **Zero impact when disabled.** Most SDK users will never turn telemetry on. For them the library must add no dependencies, no imports of OpenTelemetry, and negligible per-call overhead.
2. **Standard when enabled.** Emit plain OpenTelemetry (OTLP) so the data works with any collector and backend (Jaeger, Prometheus, Grafana, vendors).
3. **Never break the host application.** Bad configuration, a missing collector, or a missing optional dependency degrades telemetry to no-op. It never raises into user code.
4. **No changes to the Kubeflow SDK.** Instrumentation wraps the public client API from outside, so it works with released SDK versions. See [instrumentation.md](instrumentation.md).

## Layers

```
   client process (notebook / script)            training process (TrainJob)
   ──────────────────────────────────            ───────────────────────────
     user code                                     user's train function
        │                                                 │
   ┌────▼──────────────────────┐   trainer.env    ┌───────▼──────────────────┐
   │ instrumentation           │ ───────────────▶ │ training                 │
   │ wraps TrainerClient;      │  TRACEPARENT,    │ training_run(): job root │
   │ spans + client metrics    │  collector cfg   │ span, epochs, metrics    │
   └────┬──────────────────────┘  (propagation)   └───────┬──────────────────┘
        │        both depend only on the interfaces below │
   ┌────▼────────────────────────────────────────────────▼──────────────────┐
   │ kubeflow_otel.api: Tracer / Span / Meter / instruments + NoOp* defaults │
   └────┬───────────────────────────────────────────────────────────────────┘
        │ selected at runtime by kubeflow_otel.telemetry
   ┌────▼───────────────────────────────────────────────────────────────────┐
   │ kubeflow_otel._otel: OpenTelemetry adapters, providers, OTLP export     │
   │ (lazy import, optional dependency)                                      │
   └────────────────────────────────────────────────────────────────────────┘
```

See [instrumentation.md](instrumentation.md) for the client side and [propagation.md](propagation.md) for the job side.

### `kubeflow_otel.api`: interfaces and no-ops

Instrumented code talks to small abstract interfaces: `Tracer.start_span()`, `Span.set_attribute()`, `Meter.create_histogram()`, and so on. Each has a no-op implementation. The no-ops are **shared singletons** (one `NoOpSpan`, one no-op instrument), so the disabled path allocates nothing per call.

**Why not just use the OpenTelemetry API's built-in no-op behaviour?** The OpenTelemetry *API* package also no-ops when no SDK is configured, but it still has to be installed and imported. Our own interface lets the core ship with **no dependencies at all**, so `import kubeflow_otel` stays free for users who never enable telemetry. A test enforces this by checking `sys.modules` in a fresh interpreter. The interface is also deliberately small, which keeps the instrumentation code readable and easy to test.

`Span.is_recording` lets instrumentation skip building expensive attributes when nothing will be recorded.

### `kubeflow_otel.telemetry`: lifecycle

- `get_telemetry()` returns the process-wide `Telemetry` (tracer + meter + config). On first use it configures itself from the environment. After that it's a single global read on the hot path.
- `configure(config)` builds telemetry explicitly. Reconfiguring shuts down the previous exporters.
- `shutdown()` / `force_flush()` flush buffered data. `shutdown` is registered with `atexit` so short scripts don't lose their last spans.
- First-time initialization is guarded by a re-entrant lock, so concurrent first calls build exactly one backend.
- If OpenTelemetry isn't installed or the backend fails to start, `configure` logs a warning and returns the no-op telemetry.

### `kubeflow_otel._otel`: the OpenTelemetry backend

- **Own providers, not global ones.** The backend creates its own `TracerProvider` and `MeterProvider` and does **not** register them globally. Registering global providers is a one-shot operation in OpenTelemetry (a second `set_tracer_provider()` is ignored with a warning), so a library that grabs it can break an application's own setup. Applications that already configure OpenTelemetry can set `KUBEFLOW_OTEL_USE_GLOBAL_PROVIDERS=true` to have this library reuse their providers and exporters.
- **Export pipeline:** `BatchSpanProcessor` + OTLP/gRPC span exporter; `PeriodicExportingMetricReader` + OTLP/gRPC metric exporter. `console` and `none` exporters exist for local debugging and benchmarking.
- **Sampling:** `ParentBased(TraceIdRatioBased(ratio))`. A trace that started upstream keeps its sampling decision, and new traces are sampled at the configured ratio.
- **Resource:** `service.name` plus `telemetry.distro.name`/`version`, so telemetry produced by this library is identifiable.
- **Span adapter:** entering an `OTelSpan` attaches it to the current OpenTelemetry context. Spans started inside it, including spans from *other* OpenTelemetry instrumentation such as HTTP clients, become its children. Exceptions are recorded as span events, set `error.type`, mark the span `ERROR`, and are always re-raised.
- **Context propagation:** `Tracer.inject()` / `start_span(parent_carrier=...)` use the configured OpenTelemetry propagator (W3C Trace Context + Baggage by default). This is the basis for continuing a client-side trace inside a training job.

## Configuration

See the table in the [README](../README.md#configuration). Standard `OTEL_*` variables are honoured where they exist. Library-specific settings use the `KUBEFLOW_OTEL_` prefix. Invalid values log a warning and fall back to defaults.

## Testing strategy

- Unit tests run the real OpenTelemetry SDK with `InMemorySpanExporter` and `InMemoryMetricReader`, and assert on exported spans and metric points: names, kinds, attributes, status, parent/child links, histogram buckets.
- A subprocess test proves the disabled path never imports OpenTelemetry.
- A test simulates OpenTelemetry being absent and checks the graceful no-op fallback.
