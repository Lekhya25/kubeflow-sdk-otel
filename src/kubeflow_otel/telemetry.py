"""Process-wide telemetry lifecycle: configure once, fetch cheaply, shut down cleanly."""

from __future__ import annotations

import atexit
import logging
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kubeflow_otel import api
from kubeflow_otel.config import TelemetryConfig

if TYPE_CHECKING:
    from kubeflow_otel._otel import OTelBackend

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Telemetry:
    """The active tracer and meter, plus the config they were built from."""

    tracer: api.Tracer
    meter: api.Meter
    config: TelemetryConfig
    _backend: OTelBackend | None = field(default=None, repr=False, compare=False)

    @property
    def enabled(self) -> bool:
        return self._backend is not None


NOOP_TELEMETRY = Telemetry(
    tracer=api.NoOpTracer(), meter=api.NoOpMeter(), config=TelemetryConfig(enabled=False)
)

_lock = threading.RLock()
_current: Telemetry | None = None
_atexit_registered = False


def configure(config: TelemetryConfig | None = None, **backend_kwargs: Any) -> Telemetry:
    """Build the process-wide telemetry from ``config`` (default: environment).

    Calling it again replaces the previous telemetry and shuts down its
    exporters. ``backend_kwargs`` (``span_processor``, ``metric_reader``) let tests
    and advanced users plug in their own export pipeline.

    If OpenTelemetry is not installed, or the backend fails to start, telemetry
    degrades to no-op with a warning rather than raising.
    """
    global _current, _atexit_registered
    config = config or TelemetryConfig.from_env()

    new = NOOP_TELEMETRY if not config.enabled else _build(config, backend_kwargs)

    with _lock:
        previous, _current = _current, new
        if new.enabled and not _atexit_registered:
            atexit.register(shutdown)
            _atexit_registered = True
    if previous is not None and previous is not new and previous._backend is not None:
        previous._backend.shutdown()
    return new


def _build(config: TelemetryConfig, backend_kwargs: dict[str, Any]) -> Telemetry:
    try:
        from kubeflow_otel._otel import OTelBackend
    except ImportError:
        logger.warning(
            "KUBEFLOW_OTEL_ENABLED is set but OpenTelemetry is not installed; "
            "install 'kubeflow-sdk-otel[otel]'. Telemetry is disabled."
        )
        return NOOP_TELEMETRY
    try:
        backend = OTelBackend(config, **backend_kwargs)
    except Exception:
        logger.warning("Failed to start telemetry; continuing without it", exc_info=True)
        return NOOP_TELEMETRY
    return Telemetry(tracer=backend.tracer, meter=backend.meter, config=config, _backend=backend)


def get_telemetry() -> Telemetry:
    """Return the active telemetry, configuring it from the environment on first use."""
    current = _current
    if current is not None:
        return current
    with _lock:  # re-entrant: configure() takes it again
        if _current is not None:
            return _current
        return configure()


def force_flush(timeout_ms: int = 5_000) -> None:
    """Export buffered spans and metrics now (e.g. before a short-lived process exits)."""
    current = _current
    if current is not None and current._backend is not None:
        current._backend.force_flush(timeout_ms)


def shutdown() -> None:
    """Flush and stop exporters, then fall back to no-op telemetry. Idempotent."""
    global _current
    with _lock:
        current, _current = _current, None
    if current is not None and current._backend is not None:
        current._backend.shutdown()
