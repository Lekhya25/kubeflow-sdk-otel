# Performance

An observability library has to justify its cost. [`benchmarks/overhead.py`](../benchmarks/overhead.py) measures it, and the latest numbers are in [`benchmarks/RESULTS.md`](../benchmarks/RESULTS.md).

```bash
python benchmarks/overhead.py
```

## Results (summary)

Measured on a laptop (Python 3.11, Windows, Intel, 16 logical CPUs). Absolute times vary with the machine; see the note on ratios below.

| | Telemetry **off** (default) | Telemetry **on** |
|---|---|---|
| Per instrumented client call | **+0.3–0.4 µs** | ~67 µs (`get_job`) / ~190 µs (`train`) |
| …relative to one bare OpenTelemetry span + metric | — | **1.4× / 2.3×** |
| Per training epoch (`run.epoch()` + `log_metrics()`) | **+3.6 µs** | ~95 µs (**1.9×**) |
| Import of `kubeflow_otel` | **+19 ms**, no OpenTelemetry imported | +179 ms once, to load the OpenTelemetry SDK |

For scale, importing the Kubeflow SDK itself took ~4.1 s, a real Kubernetes API call takes milliseconds, and a training epoch takes seconds or more.

## What the numbers mean

- **Disabled:** the cost is a wrapper call, a global read and an attribute check, well under a microsecond, so users who never turn telemetry on pay nothing measurable. The disabled import adds ~0.5% to the SDK's own import time, and OpenTelemetry is never loaded (the benchmark asserts this).
- **Enabled:** most of the cost is the OpenTelemetry Python SDK itself. One bare span plus one histogram record costs ~84 µs on this machine, and an instrumented `get_job()` (one span, one histogram record, attribute extraction) costs 1.4× that. `train()` costs more (2.3×) because it also records a submission counter, injects the trace context and copies the trainer. Export happens on background threads, so network latency never adds to these numbers.
- **In context:** ~100 µs on a client call that talks to Kubernetes (milliseconds) is noise. Per epoch, ~95 µs against even a one-second epoch is under 0.01%.

## Methodology

- **An in-memory stub backend** replaces the Kubeflow backend, so the measured time is the instrumentation itself, not network or process management. The "uninstrumented" column is the same call with the original methods.
- **Enabled = the real pipeline minus the network:** a real `BatchSpanProcessor` and real metric aggregation (with exemplar sampling), exporting to a discarding exporter.
- **Robust statistics:** each measurement is the median of 5 `timeit` runs; the whole set (client calls, reference, epochs) is repeated for 5 rounds, and the report shows medians across rounds.
- **Ratios over absolute times.** Absolute numbers drifted by up to 2× between runs on the same laptop (CPU boost, background load). Each round also measures a **reference**, the bare OpenTelemetry SDK doing one span plus one histogram record with no library code, and expresses the enabled costs relative to it *within the same round*. Those ratios are much more stable and transfer across machines.
- **Import cost is measured incrementally, in-process:** a fresh interpreter imports the Kubeflow SDK first, then times `import kubeflow_otel` and then `configure(enabled)` with `perf_counter`. An earlier version timed whole interpreters, which (a) charged `kubeflow_otel` for stdlib modules like `logging` that the SDK already imports, and (b) was swamped by ±1 s of noise in the SDK's own 4–7 s import.

## Where the enabled time goes

A profile of the enabled `get_job()` path shows no single hotspot. The time is spread across OpenTelemetry's span creation (~⅓), metric recording with exemplar sampling (~¼), attribute validation, and locks. This library's own share (argument binding, attribute dictionaries) is small. A further reduction would mean recording fewer attributes or sampling spans (`KUBEFLOW_OTEL_SAMPLE_RATIO`); metrics are always recorded.
