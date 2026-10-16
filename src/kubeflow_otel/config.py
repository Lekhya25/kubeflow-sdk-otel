"""Telemetry configuration, read from environment variables.

Standard OpenTelemetry variables (``OTEL_*``) are honoured where they exist so the
library behaves like any other OpenTelemetry-instrumented component. Settings
specific to this library use the ``KUBEFLOW_OTEL_`` prefix.

Invalid values never raise: telemetry must not be able to break the host
application, so a warning is logged and the default is used instead.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

Exporter = Literal["otlp", "console", "none"]

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


@dataclass(frozen=True)
class TelemetryConfig:
    """Resolved telemetry settings.

    Attributes:
        enabled: Master switch. When False every instrument is a no-op.
        service_name: ``service.name`` resource attribute.
        exporter: ``otlp`` (gRPC, to a collector), ``console`` (stdout) or ``none``.
        otlp_endpoint: Collector endpoint; ``None`` lets the exporter apply the
            standard ``OTEL_EXPORTER_OTLP_ENDPOINT`` / ``localhost:4317`` default.
        metric_export_interval_ms: How often metrics are pushed.
        sample_ratio: Fraction of new traces to sample (parent-based).
        propagate_to_jobs: Inject trace context and telemetry settings into the
            environment of submitted TrainJobs.
        use_global_providers: Reuse tracer/meter providers the application has
            already registered globally instead of creating our own exporters.
    """

    enabled: bool = False
    service_name: str = "kubeflow-sdk"
    exporter: Exporter = "otlp"
    otlp_endpoint: str | None = None
    metric_export_interval_ms: int = 10_000
    sample_ratio: float = 1.0
    propagate_to_jobs: bool = True
    use_global_providers: bool = False

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> TelemetryConfig:
        env = os.environ if environ is None else environ
        d = cls()

        enabled = _bool(env, "KUBEFLOW_OTEL_ENABLED", d.enabled)
        if _bool(env, "OTEL_SDK_DISABLED", False):
            enabled = False

        exporter = env.get("KUBEFLOW_OTEL_EXPORTER", d.exporter).strip().lower()
        if exporter not in ("otlp", "console", "none"):
            logger.warning("Unknown KUBEFLOW_OTEL_EXPORTER=%r; using %r", exporter, d.exporter)
            exporter = d.exporter

        sample_ratio = _float(env, "KUBEFLOW_OTEL_SAMPLE_RATIO", d.sample_ratio)
        if not 0.0 <= sample_ratio <= 1.0:
            logger.warning("KUBEFLOW_OTEL_SAMPLE_RATIO must be in [0, 1]; using 1.0")
            sample_ratio = 1.0

        interval = _int(env, "OTEL_METRIC_EXPORT_INTERVAL", d.metric_export_interval_ms)
        if interval <= 0:
            logger.warning("OTEL_METRIC_EXPORT_INTERVAL must be positive; using default")
            interval = d.metric_export_interval_ms

        return cls(
            enabled=enabled,
            service_name=env.get("OTEL_SERVICE_NAME", "").strip() or d.service_name,
            exporter=exporter,  # type: ignore[arg-type]
            otlp_endpoint=env.get("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip() or None,
            metric_export_interval_ms=interval,
            sample_ratio=sample_ratio,
            propagate_to_jobs=_bool(env, "KUBEFLOW_OTEL_PROPAGATE", d.propagate_to_jobs),
            use_global_providers=_bool(
                env, "KUBEFLOW_OTEL_USE_GLOBAL_PROVIDERS", d.use_global_providers
            ),
        )


def _bool(env: Mapping[str, str], key: str, default: bool) -> bool:
    raw = env.get(key)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    logger.warning("Invalid boolean %s=%r; using %r", key, raw, default)
    return default


def _float(env: Mapping[str, str], key: str, default: float) -> float:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid number %s=%r; using %r", key, raw, default)
        return default


def _int(env: Mapping[str, str], key: str, default: int) -> int:
    raw = env.get(key)
    if raw is None or not raw.strip():
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid integer %s=%r; using %r", key, raw, default)
        return default
