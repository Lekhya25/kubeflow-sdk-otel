# Observability stack

A local, one-command stack that receives everything this library emits and makes it explorable:

```
 kubeflow-sdk (client) ─┐
                        ├─ OTLP ─▶ OpenTelemetry Collector ─┬─ traces  (OTLP/gRPC) ─▶ Jaeger ─────┐
 kubeflow-trainjob (job)┘          :4317 gRPC / :4318 HTTP  └─ metrics (OTLP/HTTP) ─▶ Prometheus ─┤
                                                                                                 ▼
                                                                                              Grafana
```

| Component | Image | URL |
|---|---|---|
| OpenTelemetry Collector (contrib) | `otel/opentelemetry-collector-contrib:0.128.0` | OTLP on `localhost:4317` / `4318` |
| Jaeger v2 | `jaegertracing/jaeger:2.7.0` | http://localhost:16686 |
| Prometheus | `prom/prometheus:v3.5.0` (LTS) | http://localhost:9090 |
| Grafana | `grafana/grafana:12.0.2` | http://localhost:3000 |

## Run it

```bash
docker compose -f deploy/docker-compose.yml up -d --wait
python scripts/smoke_test.py          # sends telemetry, verifies it arrived everywhere
```

Then open Grafana at http://localhost:3000. The **Kubeflow SDK** dashboard is the home page, and login is disabled for local use. Point any instrumented program at the collector with:

```bash
export KUBEFLOW_OTEL_ENABLED=true
export OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
```

Stop with `docker compose -f deploy/docker-compose.yml down` (add `-v` to also delete stored data).

## Design notes

### One entry point: the collector
Applications export a single protocol (OTLP) to a single address. Routing, batching, memory limiting and fan-out to backends live in the collector, so swapping Jaeger for another trace store is a collector config change, not an application change. The collector runs `memory_limiter` before `batch` so a burst of telemetry can't push it out of memory.

### Metrics are pushed to Prometheus (OTLP), not scraped
Prometheus 3 can ingest OTLP directly (`--web.enable-otlp-receiver`), and the collector pushes metrics to it with the `otlphttp` exporter.

The obvious alternative is the collector's scrape-based `prometheus` exporter, and that's what this stack used first. It was replaced because it **loses exemplars**. That exporter keeps only the *latest* data point of each series. The OpenTelemetry SDK attaches an exemplar only to the export cycle in which a measurement happened, so as soon as an export arrives with no new measurement, the stored point (and its exemplar) is replaced by one without. For bursty operations like job submissions, almost no exemplars ever reached Prometheus. Pushing stores every data point with its exemplars.

`translation_strategy: UnderscoreEscapingWithSuffixes` keeps conventional Prometheus names: `kubeflow.client.operation.duration` (unit `s`) becomes `kubeflow_client_operation_duration_seconds`, counters get `_total`, dots in labels become underscores. `service.name` becomes the `job` label and `service.instance.id` becomes `instance`.

### Exemplars: from a latency spike to the exact trace
Histogram measurements made inside a sampled span carry that span's trace ID as an *exemplar*. Prometheus stores them (`--enable-feature=exemplar-storage`), and the Grafana Prometheus datasource maps the `trace_id` exemplar label to the Jaeger datasource. On the **Client call latency p95** panel, every dot is a real call; clicking one opens its trace.

### Counting correctly for short-lived processes
Each client process (a notebook kernel, a script) has its own counter series, distinguished by `instance`. PromQL's `increase()` needs two samples, so it **misses the first increment of every new series**: a script that submits one job and exits contributes 0. Prometheus' `created-timestamp-zero-ingestion` feature would fix this by inserting a zero at the series start, but in Prometheus 3.4 and 3.5 it only applies to scraped targets, not to OTLP ingestion (verified).

The dashboard therefore counts with a **start-inclusive increase**:

```promql
last_over_time(c[W]) - (last_over_time(c[W] offset W) or last_over_time(c[W]) * 0)
```

That's the value at the end of the window minus the value at its start, or minus 0 if the series didn't exist yet. Because every process has its own series and counters never reset within a process, this is exact. The same expression feeds `histogram_quantile` (bucket counts) and the bar charts (per-interval increments).

### Dashboards as code
`deploy/grafana/dashboards/kubeflow-sdk.json` is **generated** by `scripts/build_dashboard.py`, where the start-inclusive increase is defined once and reused by every panel. CI runs `build_dashboard.py --check` to make sure the committed JSON matches the generator, and the smoke test executes every panel query against live Prometheus to catch invalid PromQL.

## The dashboard

**Client (kubeflow-sdk)**
- Jobs submitted, completed and failed; job success rate; client call error rate; job duration p50/p95 (stat panels over the selected range).
- Client call latency p95 by operation, **with exemplars**, on a log scale (waits take minutes, lookups milliseconds).
- Client calls by operation (stacked bars per interval).
- Jobs submitted and finished per interval.
- Job startup p95 as observed by the client (created → first seen running). It's inflated when jobs are waited on one after another; the job-side *Time to first epoch* is exact.

**Training jobs (kubeflow-trainjob)**, filterable by the *Training job* variable:
- Training loss and accuracy per job.
- Current epoch per job.
- Mean epoch time per job (stragglers stand out).
- Time to first epoch per job (environment setup overhead, measured inside the job).

Loss uses a log scale so one diverging run can't flatten the others. Loss and accuracy are queried with `last_over_time(...[30s])`. Pushed series get no staleness marker when a job exits, so the default 5-minute lookback would draw each finished job's last value as a flat line for 5 minutes.

**Traces**
- A table of recent `train()` traces from Jaeger; each trace ID links into the trace view.

## Verifying the pipeline

`scripts/smoke_test.py` drives an instrumented `TrainerClient` (against a stub backend, so it runs anywhere) and a job-side `training_run`, then checks:

1. a trace containing both the client `train` span and the job `run` span is searchable in Jaeger;
2. the client and training metrics exist in Prometheus;
3. exemplars are stored;
4. every Grafana panel query is valid PromQL against the live Prometheus.

The `e2e` CI job runs the whole stack on GitHub Actions and executes this script on every push.

## Not for production as-is

This stack is for local development and demos:
- Grafana has anonymous **Admin** access and no login.
- Jaeger uses in-memory storage (traces are lost on restart).
- Prometheus has default retention and no authentication on its OTLP endpoint.

For a real cluster, run the collector as a DaemonSet or Deployment (e.g. via the OpenTelemetry Operator), use persistent storage for Jaeger (or Tempo), and put authentication in front of Grafana and Prometheus.
