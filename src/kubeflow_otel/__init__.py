"""OpenTelemetry tracing and metrics for the Kubeflow SDK.

Telemetry is off by default and costs almost nothing while off. Enable it with
``KUBEFLOW_OTEL_ENABLED=true`` (plus ``pip install kubeflow-sdk-otel[otel]``).
"""

from kubeflow_otel._version import __version__
from kubeflow_otel.config import TelemetryConfig
from kubeflow_otel.telemetry import (
    Telemetry,
    configure,
    force_flush,
    get_telemetry,
    shutdown,
)

__all__ = [
    "Telemetry",
    "TelemetryConfig",
    "__version__",
    "configure",
    "force_flush",
    "get_telemetry",
    "shutdown",
]
