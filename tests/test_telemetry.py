import os
import subprocess
import sys
import textwrap

import pytest
from opentelemetry.trace import SpanKind, StatusCode

import kubeflow_otel
from kubeflow_otel import TelemetryConfig, api
from kubeflow_otel.telemetry import NOOP_TELEMETRY

# --------------------------------------------------------------------- disabled


def test_disabled_by_default_returns_noop():
    telemetry = kubeflow_otel.get_telemetry()
    assert telemetry is NOOP_TELEMETRY
    assert isinstance(telemetry.tracer, api.NoOpTracer)
    assert isinstance(telemetry.meter, api.NoOpMeter)


def test_noop_span_never_swallows_exceptions():
    tracer = api.NoOpTracer()
    with pytest.raises(RuntimeError), tracer.start_span("op") as span:
        assert span.is_recording is False
        raise RuntimeError("boom")


def test_noop_instruments_accept_calls():
    meter = api.NoOpMeter()
    meter.create_counter("c").add(1, {"a": "b"})
    meter.create_histogram("h", boundaries=[1, 2]).record(1.5)
    meter.create_gauge("g").set(3)
    meter.create_up_down_counter("u").add(-1)
    carrier: dict[str, str] = {}
    api.NoOpTracer().inject(carrier)
    assert carrier == {}


def test_disabled_path_does_not_import_opentelemetry():
    code = textwrap.dedent(
        """
        import sys, kubeflow_otel
        t = kubeflow_otel.get_telemetry()
        with t.tracer.start_span("x"):
            pass
        print(any(m.startswith("opentelemetry") for m in sys.modules))
        """
    )
    env = {k: v for k, v in os.environ.items() if not k.startswith("KUBEFLOW_OTEL")}
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    assert out.stdout.strip() == "False"


def test_missing_opentelemetry_degrades_to_noop(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "kubeflow_otel._otel", None)  # import -> ImportError
    telemetry = kubeflow_otel.configure(TelemetryConfig(enabled=True))
    assert telemetry is NOOP_TELEMETRY
    assert "not installed" in caplog.text


# ---------------------------------------------------------------------- enabled


def test_span_is_exported_with_kind_and_attributes(otel):
    with otel.telemetry.tracer.start_span(
        "kubeflow.trainer.train", kind="client", attributes={"kubeflow.runtime.name": "torch"}
    ) as span:
        assert span.is_recording
        span.add_event("submitted", {"n": 1})

    (exported,) = otel.finished_spans()
    assert exported.name == "kubeflow.trainer.train"
    assert exported.kind == SpanKind.CLIENT
    assert exported.attributes["kubeflow.runtime.name"] == "torch"
    assert exported.events[0].name == "submitted"
    assert exported.resource.attributes["service.name"] == "test-service"
    assert exported.status.status_code == StatusCode.UNSET


def test_exception_marks_span_failed_and_propagates(otel):
    with pytest.raises(ValueError), otel.telemetry.tracer.start_span("op"):
        raise ValueError("bad runtime")

    (exported,) = otel.finished_spans()
    assert exported.status.status_code == StatusCode.ERROR
    assert exported.attributes["error.type"] == "ValueError"
    assert any(e.name == "exception" for e in exported.events)


def test_nested_spans_are_parented(otel):
    tracer = otel.telemetry.tracer
    with tracer.start_span("outer"), tracer.start_span("inner"):
        pass
    inner, outer = otel.finished_spans()
    assert inner.parent.span_id == outer.context.span_id
    assert inner.context.trace_id == outer.context.trace_id


def test_context_round_trips_through_a_carrier(otel):
    tracer = otel.telemetry.tracer
    carrier: dict[str, str] = {}
    with tracer.start_span("client-side"):
        tracer.inject(carrier)
    assert carrier["traceparent"].startswith("00-")

    # e.g. in another process, rebuilt from the job's environment
    with tracer.start_span("job-side", parent_carrier=carrier):
        pass
    client, job = otel.finished_spans()
    assert job.context.trace_id == client.context.trace_id
    assert job.parent.span_id == client.context.span_id
    assert job.parent.is_remote


def test_metrics_are_recorded(otel):
    meter = otel.telemetry.meter
    meter.create_counter("jobs", unit="{job}").add(2, {"status": "ok"})
    meter.create_histogram("latency", unit="s", boundaries=[0.1, 1.0]).record(0.5)
    meter.create_gauge("loss").set(0.25, {"job": "a"})

    (jobs,) = otel.metric_points("jobs")
    assert jobs.value == 2 and jobs.attributes == {"status": "ok"}
    (latency,) = otel.metric_points("latency")
    assert list(latency.explicit_bounds) == [0.1, 1.0]
    assert latency.bucket_counts[1] == 1
    (loss,) = otel.metric_points("loss")
    assert loss.value == 0.25


def test_reconfigure_and_shutdown_are_safe(otel):
    kubeflow_otel.configure(TelemetryConfig(enabled=True, exporter="none"))
    kubeflow_otel.shutdown()
    kubeflow_otel.shutdown()  # idempotent
    kubeflow_otel.force_flush()  # no-op when nothing is configured
    assert kubeflow_otel.get_telemetry() is NOOP_TELEMETRY
