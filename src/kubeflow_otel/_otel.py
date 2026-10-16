"""OpenTelemetry-backed implementations of :mod:`kubeflow_otel.api`.

This module is imported lazily, only when telemetry is enabled, so the
OpenTelemetry packages are an optional dependency (``pip install
kubeflow-sdk-otel[otel]``).
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping, Sequence
from types import TracebackType

from opentelemetry import context as otel_context
from opentelemetry import metrics as otel_metrics
from opentelemetry import propagate
from opentelemetry import trace as otel_trace
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    MetricReader,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import SERVICE_NAME, Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SpanExporter,
    SpanProcessor,
)
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.trace import Status, StatusCode

from kubeflow_otel import api
from kubeflow_otel._version import __version__
from kubeflow_otel.config import TelemetryConfig

logger = logging.getLogger(__name__)

INSTRUMENTATION_SCOPE = "kubeflow_otel"

_KINDS = {
    "internal": otel_trace.SpanKind.INTERNAL,
    "client": otel_trace.SpanKind.CLIENT,
    "server": otel_trace.SpanKind.SERVER,
    "producer": otel_trace.SpanKind.PRODUCER,
    "consumer": otel_trace.SpanKind.CONSUMER,
}


class OTelSpan(api.Span):
    """Adapter exposing an OpenTelemetry span through :class:`api.Span`.

    Entering the context manager makes the span current, so spans started
    inside it (including spans from other OpenTelemetry instrumentation) become
    its children.
    """

    __slots__ = ("_span", "_token", "_ended")

    def __init__(self, span: otel_trace.Span) -> None:
        self._span = span
        self._token: object | None = None
        self._ended = False

    @property
    def otel_span(self) -> otel_trace.Span:
        return self._span

    def set_attribute(self, key: str, value: api.AttributeValue) -> None:
        self._span.set_attribute(key, value)

    def set_attributes(self, attributes: api.Attributes) -> None:
        self._span.set_attributes(attributes)

    def add_event(self, name: str, attributes: api.Attributes | None = None) -> None:
        self._span.add_event(name, attributes=attributes)

    def record_exception(self, exc: BaseException) -> None:
        self._span.record_exception(exc)
        self._span.set_attribute("error.type", type(exc).__qualname__)
        self._span.set_status(Status(StatusCode.ERROR, str(exc) or type(exc).__qualname__))

    def end(self) -> None:
        if not self._ended:
            self._ended = True
            self._span.end()

    @property
    def is_recording(self) -> bool:
        return self._span.is_recording()

    def __enter__(self) -> OTelSpan:
        self._token = otel_context.attach(otel_trace.set_span_in_context(self._span))
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        try:
            if exc is not None:
                self.record_exception(exc)
            self.end()
        finally:
            if self._token is not None:
                otel_context.detach(self._token)  # type: ignore[arg-type]
                self._token = None
        return False


class OTelTracer(api.Tracer):
    def __init__(self, tracer: otel_trace.Tracer) -> None:
        self._tracer = tracer

    def start_span(
        self,
        name: str,
        *,
        kind: api.SpanKind = "internal",
        attributes: api.Attributes | None = None,
        parent_carrier: Mapping[str, str] | None = None,
    ) -> api.Span:
        ctx = propagate.extract(parent_carrier) if parent_carrier else None
        span = self._tracer.start_span(name, context=ctx, kind=_KINDS[kind], attributes=attributes)
        return OTelSpan(span)

    def inject(self, carrier: MutableMapping[str, str]) -> None:
        propagate.inject(carrier)


class _Counter(api.Counter, api.UpDownCounter):
    __slots__ = ("_inst",)

    def __init__(self, inst: otel_metrics.Counter | otel_metrics.UpDownCounter) -> None:
        self._inst = inst

    def add(self, amount: int | float, attributes: api.Attributes | None = None) -> None:
        self._inst.add(amount, attributes)


class _Histogram(api.Histogram):
    __slots__ = ("_inst",)

    def __init__(self, inst: otel_metrics.Histogram) -> None:
        self._inst = inst

    def record(self, amount: int | float, attributes: api.Attributes | None = None) -> None:
        self._inst.record(amount, attributes)


class _Gauge(api.Gauge):
    __slots__ = ("_inst",)

    def __init__(self, inst: otel_metrics._Gauge) -> None:
        self._inst = inst

    def set(self, amount: int | float, attributes: api.Attributes | None = None) -> None:
        self._inst.set(amount, attributes)


class OTelMeter(api.Meter):
    def __init__(self, meter: otel_metrics.Meter) -> None:
        self._meter = meter

    def create_counter(self, name: str, unit: str = "", description: str = "") -> api.Counter:
        return _Counter(self._meter.create_counter(name, unit=unit, description=description))

    def create_up_down_counter(
        self, name: str, unit: str = "", description: str = ""
    ) -> api.UpDownCounter:
        return _Counter(
            self._meter.create_up_down_counter(name, unit=unit, description=description)
        )

    def create_histogram(
        self,
        name: str,
        unit: str = "",
        description: str = "",
        boundaries: Sequence[float] | None = None,
    ) -> api.Histogram:
        return _Histogram(
            self._meter.create_histogram(
                name,
                unit=unit,
                description=description,
                explicit_bucket_boundaries_advisory=boundaries,
            )
        )

    def create_gauge(self, name: str, unit: str = "", description: str = "") -> api.Gauge:
        return _Gauge(self._meter.create_gauge(name, unit=unit, description=description))


class OTelBackend:
    """Owns the tracer/meter providers and their export pipelines.

    Providers are created once per backend and are *not* registered as the
    global OpenTelemetry providers, so this library never fights with an
    application that configures OpenTelemetry itself. Set
    ``use_global_providers`` to reuse the application's providers instead.
    """

    def __init__(
        self,
        config: TelemetryConfig,
        *,
        span_processor: SpanProcessor | None = None,
        metric_reader: MetricReader | None = None,
    ) -> None:
        self._owns_providers = not config.use_global_providers
        if config.use_global_providers:
            tracer_provider = otel_trace.get_tracer_provider()
            meter_provider = otel_metrics.get_meter_provider()
            self._tracer_provider: TracerProvider | None = None
            self._meter_provider: MeterProvider | None = None
        else:
            resource = Resource.create(
                {
                    SERVICE_NAME: config.service_name,
                    "telemetry.distro.name": "kubeflow-sdk-otel",
                    "telemetry.distro.version": __version__,
                }
            )
            self._tracer_provider = TracerProvider(
                resource=resource,
                sampler=ParentBased(TraceIdRatioBased(config.sample_ratio)),
            )
            processor = span_processor or _default_span_processor(config)
            if processor is not None:
                self._tracer_provider.add_span_processor(processor)

            readers = [metric_reader or _default_metric_reader(config)]
            self._meter_provider = MeterProvider(
                resource=resource, metric_readers=[r for r in readers if r is not None]
            )
            tracer_provider, meter_provider = self._tracer_provider, self._meter_provider

        self.tracer = OTelTracer(tracer_provider.get_tracer(INSTRUMENTATION_SCOPE, __version__))
        self.meter = OTelMeter(meter_provider.get_meter(INSTRUMENTATION_SCOPE, __version__))

    def force_flush(self, timeout_ms: int = 5_000) -> None:
        if self._tracer_provider is not None:
            self._tracer_provider.force_flush(timeout_ms)
        if self._meter_provider is not None:
            self._meter_provider.force_flush(timeout_ms)

    def shutdown(self) -> None:
        if not self._owns_providers:
            return
        for provider in (self._tracer_provider, self._meter_provider):
            if provider is None:
                continue
            try:
                provider.shutdown()
            except Exception:  # pragma: no cover - exporter failures must not propagate
                logger.debug("Error shutting down telemetry provider", exc_info=True)


def _default_span_processor(config: TelemetryConfig) -> SpanProcessor | None:
    exporter: SpanExporter
    if config.exporter == "none":
        return None
    if config.exporter == "console":
        exporter = ConsoleSpanExporter()
    else:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

        exporter = OTLPSpanExporter(endpoint=config.otlp_endpoint)
    return BatchSpanProcessor(exporter)


def _default_metric_reader(config: TelemetryConfig) -> MetricReader | None:
    if config.exporter == "none":
        return None
    if config.exporter == "console":
        exporter = ConsoleMetricExporter()
    else:
        from opentelemetry.exporter.otlp.proto.grpc.metric_exporter import OTLPMetricExporter

        exporter = OTLPMetricExporter(endpoint=config.otlp_endpoint)
    return PeriodicExportingMetricReader(
        exporter, export_interval_millis=config.metric_export_interval_ms
    )
