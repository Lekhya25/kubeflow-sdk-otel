"""Backend-neutral telemetry interfaces and their no-op implementations.

Instrumented code depends only on the interfaces in this module. When telemetry
is disabled (the default) the no-op implementations are used, so OpenTelemetry
does not need to be installed and an instrumented call costs a handful of
attribute lookups. When telemetry is enabled, :mod:`kubeflow_otel._otel`
provides OpenTelemetry-backed implementations of the same interfaces.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, MutableMapping, Sequence
from types import TracebackType
from typing import Literal

AttributeValue = str | bool | int | float
Attributes = Mapping[str, AttributeValue]
SpanKind = Literal["internal", "client", "server", "producer", "consumer"]


class Span(abc.ABC):
    """A unit of work. Usable as a context manager that makes it the current span."""

    @abc.abstractmethod
    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Attach a key/value attribute to the span."""

    def set_attributes(self, attributes: Attributes) -> None:
        for key, value in attributes.items():
            self.set_attribute(key, value)

    @abc.abstractmethod
    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        """Record a timestamped event (e.g. a job status transition) on the span."""

    @abc.abstractmethod
    def record_exception(self, exc: BaseException) -> None:
        """Record an exception and mark the span as failed."""

    @abc.abstractmethod
    def end(self) -> None:
        """End the span. Ending twice is a no-op."""

    @property
    @abc.abstractmethod
    def is_recording(self) -> bool:
        """False for no-op spans; lets callers skip building expensive attributes."""

    @abc.abstractmethod
    def __enter__(self) -> Span: ...

    @abc.abstractmethod
    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool: ...


class Tracer(abc.ABC):
    """Creates spans and propagates trace context across process boundaries."""

    @abc.abstractmethod
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "internal",
        attributes: Attributes | None = None,
        parent_carrier: Mapping[str, str] | None = None,
    ) -> Span:
        """Start a span.

        The parent is the current span unless ``parent_carrier`` is given, in which
        case the parent is the remote span encoded in it (W3C ``traceparent``).
        """

    @abc.abstractmethod
    def inject(self, carrier: MutableMapping[str, str]) -> None:
        """Write the current trace context into ``carrier`` (W3C Trace Context keys)."""


class Counter(abc.ABC):
    @abc.abstractmethod
    def add(self, amount: int | float, attributes: Attributes | None = None) -> None: ...


class UpDownCounter(abc.ABC):
    @abc.abstractmethod
    def add(self, amount: int | float, attributes: Attributes | None = None) -> None: ...


class Histogram(abc.ABC):
    @abc.abstractmethod
    def record(self, amount: int | float, attributes: Attributes | None = None) -> None: ...


class Gauge(abc.ABC):
    @abc.abstractmethod
    def set(self, amount: int | float, attributes: Attributes | None = None) -> None: ...


class Meter(abc.ABC):
    """Creates metric instruments."""

    @abc.abstractmethod
    def create_counter(self, name: str, unit: str = "", description: str = "") -> Counter: ...

    @abc.abstractmethod
    def create_up_down_counter(
        self, name: str, unit: str = "", description: str = ""
    ) -> UpDownCounter: ...

    @abc.abstractmethod
    def create_histogram(
        self,
        name: str,
        unit: str = "",
        description: str = "",
        boundaries: Sequence[float] | None = None,
    ) -> Histogram: ...

    @abc.abstractmethod
    def create_gauge(self, name: str, unit: str = "", description: str = "") -> Gauge: ...


# --------------------------------------------------------------------------- no-op


class NoOpSpan(Span):
    """Span that records nothing and never suppresses exceptions."""

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        pass

    def set_attributes(self, attributes: Attributes) -> None:
        pass

    def add_event(self, name: str, attributes: Attributes | None = None) -> None:
        pass

    def record_exception(self, exc: BaseException) -> None:
        pass

    def end(self) -> None:
        pass

    @property
    def is_recording(self) -> bool:
        return False

    def __enter__(self) -> NoOpSpan:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        return False


_NOOP_SPAN = NoOpSpan()


class NoOpTracer(Tracer):
    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = "internal",
        attributes: Attributes | None = None,
        parent_carrier: Mapping[str, str] | None = None,
    ) -> Span:
        return _NOOP_SPAN

    def inject(self, carrier: MutableMapping[str, str]) -> None:
        pass


class _NoOpInstrument(Counter, UpDownCounter, Histogram, Gauge):
    def add(self, amount: int | float, attributes: Attributes | None = None) -> None:
        pass

    def record(self, amount: int | float, attributes: Attributes | None = None) -> None:
        pass

    def set(self, amount: int | float, attributes: Attributes | None = None) -> None:
        pass


_NOOP_INSTRUMENT = _NoOpInstrument()


class NoOpMeter(Meter):
    def create_counter(self, name: str, unit: str = "", description: str = "") -> Counter:
        return _NOOP_INSTRUMENT

    def create_up_down_counter(
        self, name: str, unit: str = "", description: str = ""
    ) -> UpDownCounter:
        return _NOOP_INSTRUMENT

    def create_histogram(
        self,
        name: str,
        unit: str = "",
        description: str = "",
        boundaries: Sequence[float] | None = None,
    ) -> Histogram:
        return _NOOP_INSTRUMENT

    def create_gauge(self, name: str, unit: str = "", description: str = "") -> Gauge:
        return _NOOP_INSTRUMENT
