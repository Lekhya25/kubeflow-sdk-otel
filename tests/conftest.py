from __future__ import annotations

from dataclasses import dataclass

import pytest
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

import kubeflow_otel
from kubeflow_otel import Telemetry, TelemetryConfig


@dataclass
class OTelHarness:
    telemetry: Telemetry
    spans: InMemorySpanExporter
    metrics: InMemoryMetricReader

    def finished_spans(self):
        return self.spans.get_finished_spans()

    def metric_points(self, name: str) -> list:
        """All data points of metric ``name`` from a fresh collection."""
        data = self.metrics.get_metrics_data()
        points = []
        for rm in data.resource_metrics if data else []:
            for sm in rm.scope_metrics:
                for m in sm.metrics:
                    if m.name == name:
                        points.extend(m.data.data_points)
        return points


@pytest.fixture(autouse=True)
def _reset_telemetry(monkeypatch):
    for var in ("KUBEFLOW_OTEL_ENABLED", "OTEL_SDK_DISABLED", "KUBEFLOW_OTEL_EXPORTER"):
        monkeypatch.delenv(var, raising=False)
    kubeflow_otel.shutdown()
    yield
    kubeflow_otel.shutdown()


@pytest.fixture
def otel() -> OTelHarness:
    """Telemetry enabled with in-memory exporters."""
    spans = InMemorySpanExporter()
    reader = InMemoryMetricReader()
    telemetry = kubeflow_otel.configure(
        TelemetryConfig(enabled=True, service_name="test-service"),
        span_processor=SimpleSpanProcessor(spans),
        metric_reader=reader,
    )
    assert telemetry.enabled
    return OTelHarness(telemetry=telemetry, spans=spans, metrics=reader)
