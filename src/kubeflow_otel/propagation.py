"""Carry trace context and telemetry settings from the client into a TrainJob.

The client side builds a set of environment variables (:func:`job_environment`)
that the instrumentation merges into ``trainer.env``. Every Kubeflow backend
(local process, container, Kubernetes) passes ``trainer.env`` to the training
nodes, so the job can continue the client's trace and export to the same
collector. The job side reads them back with :func:`carrier_from_env`.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

from kubeflow_otel import semconv as sc
from kubeflow_otel.telemetry import Telemetry

# carrier key (as used by OpenTelemetry propagators) -> environment variable
_ENV_KEYS = {"traceparent": sc.ENV_TRACEPARENT, "tracestate": sc.ENV_TRACESTATE}


def job_environment(
    telemetry: Telemetry,
    *,
    job_name: str | None = None,
    runtime_name: str | None = None,
) -> dict[str, str]:
    """Environment variables that let a TrainJob join the current trace.

    Must be called while the span the job should hang under is current.
    """
    carrier: dict[str, str] = {}
    telemetry.tracer.inject(carrier)
    env = {_ENV_KEYS[k]: v for k, v in carrier.items() if k in _ENV_KEYS}

    config = telemetry.config
    env["KUBEFLOW_OTEL_ENABLED"] = "true"
    env["KUBEFLOW_OTEL_EXPORTER"] = config.exporter
    env["OTEL_SERVICE_NAME"] = sc.DEFAULT_JOB_SERVICE_NAME
    env["OTEL_METRIC_EXPORT_INTERVAL"] = str(config.metric_export_interval_ms)
    if config.otlp_endpoint:
        env["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otlp_endpoint

    resource_attrs = []
    if job_name:
        env[sc.ENV_TRAINJOB_NAME] = job_name
        resource_attrs.append(f"{sc.TRAINJOB_NAME}={job_name}")
    if runtime_name:
        env[sc.ENV_RUNTIME_NAME] = runtime_name
        resource_attrs.append(f"{sc.RUNTIME_NAME}={runtime_name}")
    if resource_attrs:
        # read automatically by the OpenTelemetry SDK's Resource detection
        env["OTEL_RESOURCE_ATTRIBUTES"] = ",".join(resource_attrs)
    return env


def merge_job_environment(
    user_env: Mapping[str, str] | None, injected: Mapping[str, str]
) -> dict[str, str]:
    """User-provided values win, except the trace context, which must be ours."""
    merged = dict(injected)
    for key, value in (user_env or {}).items():
        if key in (sc.ENV_TRACEPARENT, sc.ENV_TRACESTATE):
            continue
        merged[key] = value
    return merged


def carrier_from_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Rebuild a propagation carrier from ``TRACEPARENT``/``TRACESTATE``."""
    env = os.environ if environ is None else environ
    return {k: env[v] for k, v in _ENV_KEYS.items() if env.get(v)}
