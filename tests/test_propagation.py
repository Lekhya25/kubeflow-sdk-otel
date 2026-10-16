import os
import re
import subprocess
import sys
import textwrap

import pytest
from fake_backend import FakeBackend
from kubeflow.trainer import CustomTrainer, LocalProcessBackendConfig, TrainerClient
from kubeflow.trainer.options.common import Name
from opentelemetry.trace import SpanKind, StatusCode

import kubeflow_otel
from kubeflow_otel import TelemetryConfig, TrainerClientInstrumentor
from kubeflow_otel import semconv as sc
from kubeflow_otel.propagation import carrier_from_env, merge_job_environment
from kubeflow_otel.training import training_run


def train_fn():
    pass


@pytest.fixture
def instrumentor():
    inst = TrainerClientInstrumentor()
    inst.instrument()
    yield inst
    inst.uninstrument()


@pytest.fixture
def backend():
    return FakeBackend()


@pytest.fixture
def client(backend):
    c = TrainerClient(backend_config=LocalProcessBackendConfig())
    c.backend = backend
    return c


def submit(client, **trainer_kwargs):
    trainer = CustomTrainer(func=train_fn, **trainer_kwargs)
    client.train(runtime="torch-distributed", trainer=trainer, options=[Name("run-7")])
    return trainer


def train_span(otel):
    (span,) = [s for s in otel.finished_spans() if s.name == "kubeflow.trainer.train"]
    return span


# ------------------------------------------------------------------ client side


def test_train_injects_trace_context_into_job_env(otel, instrumentor, client, backend):
    trainer = submit(client, env={"MY_VAR": "1"})

    (env,) = backend.received_envs
    span = train_span(otel)
    version, trace_id, parent_id, flags = env[sc.ENV_TRACEPARENT].split("-")
    assert trace_id == format(span.context.trace_id, "032x")
    assert parent_id == format(span.context.span_id, "016x")  # job hangs under train()
    assert int(flags, 16) & 0x01  # "sampled" bit (0x02 is the Level 2 "random trace-id" flag)

    assert env["MY_VAR"] == "1"  # user variables preserved
    assert env["KUBEFLOW_OTEL_ENABLED"] == "true"
    assert env["OTEL_SERVICE_NAME"] == sc.DEFAULT_JOB_SERVICE_NAME
    assert env[sc.ENV_TRAINJOB_NAME] == "run-7"
    assert env[sc.ENV_RUNTIME_NAME] == "torch-distributed"
    assert "kubeflow.trainjob.name=run-7" in env["OTEL_RESOURCE_ATTRIBUTES"]
    assert trainer.env == {"MY_VAR": "1"}  # caller's trainer object untouched


def test_user_values_win_except_trace_context():
    merged = merge_job_environment(
        {"OTEL_SERVICE_NAME": "custom", sc.ENV_TRACEPARENT: "stale"},
        {"OTEL_SERVICE_NAME": "kubeflow-trainjob", sc.ENV_TRACEPARENT: "00-fresh"},
    )
    assert merged["OTEL_SERVICE_NAME"] == "custom"
    assert merged[sc.ENV_TRACEPARENT] == "00-fresh"


def test_no_injection_when_propagation_is_off(instrumentor, client, backend):
    kubeflow_otel.configure(TelemetryConfig(enabled=True, exporter="none", propagate_to_jobs=False))
    submit(client)
    assert backend.received_envs == [None]


def test_no_injection_when_telemetry_is_disabled(instrumentor, client, backend):
    submit(client)
    assert backend.received_envs == [None]


# --------------------------------------------------------------------- job side


def test_training_run_continues_the_client_trace(otel, instrumentor, client, backend):
    submit(client)
    (job_env,) = backend.received_envs

    # what the training function does inside the job
    with training_run(environ=job_env) as run:
        for epoch in range(2):
            with run.epoch(epoch):
                run.log_metrics({"loss": 1.0 / (epoch + 1), "accuracy": 0.5 + epoch / 10})

    spans = {s.name: s for s in otel.finished_spans()}
    client_span = spans["kubeflow.trainer.train"]
    run_span = spans["kubeflow.trainjob.run"]
    assert run_span.context.trace_id == client_span.context.trace_id
    assert run_span.parent.span_id == client_span.context.span_id
    assert run_span.parent.is_remote
    assert run_span.kind == SpanKind.SERVER
    assert run_span.attributes[sc.TRAINJOB_NAME] == "run-7"

    epochs = [s for s in otel.finished_spans() if s.name == "kubeflow.trainjob.epoch"]
    assert [s.attributes[sc.TRAINING_EPOCH] for s in epochs] == [0, 1]
    assert all(s.parent.span_id == run_span.context.span_id for s in epochs)

    metric_events = [e for e in run_span.events if e.name == sc.EVENT_TRAINING_METRICS]
    assert [e.attributes["step"] for e in metric_events] == [0, 1]

    loss = {
        p.attributes[sc.TRAINING_METRIC_NAME]: p.value
        for p in otel.metric_points(sc.METRIC_TRAINING_METRIC)
    }
    assert loss == {"loss": 0.5, "accuracy": 0.6}
    (duration,) = otel.metric_points(sc.METRIC_EPOCH_DURATION)
    assert duration.count == 2
    assert duration.attributes == {sc.TRAINJOB_NAME: "run-7"}


def test_training_run_without_context_starts_a_new_trace(otel):
    with training_run(environ={"RANK": "1", "WORLD_SIZE": "4"}) as run:
        run.log_metrics({"loss": 0.3}, step=10)
    (span,) = otel.finished_spans()
    assert span.parent is None
    assert span.attributes[sc.TRAINING_RANK] == 1
    assert span.attributes[sc.TRAINING_WORLD_SIZE] == 4
    (point,) = otel.metric_points(sc.METRIC_TRAINING_METRIC)
    assert point.attributes[sc.TRAINING_RANK] == 1


def test_failure_inside_an_epoch_marks_the_run_failed(otel):
    with pytest.raises(FloatingPointError), training_run(environ={}) as run, run.epoch(3):
        raise FloatingPointError("loss is NaN")
    spans = {s.name: s for s in otel.finished_spans()}
    assert spans["kubeflow.trainjob.epoch"].status.status_code == StatusCode.ERROR
    assert spans["kubeflow.trainjob.run"].status.status_code == StatusCode.ERROR


def test_training_run_is_a_noop_when_disabled():
    with training_run(environ={}) as run, run.epoch(0):
        run.log_metrics({"loss": 1.0})
    assert run.span.is_recording is False


def test_carrier_from_env_ignores_missing_values():
    assert carrier_from_env({}) == {}
    assert carrier_from_env({sc.ENV_TRACEPARENT: "00-a-b-01"}) == {"traceparent": "00-a-b-01"}


# ------------------------------------------------------------ across processes


def test_job_in_a_separate_process_joins_the_trace(otel, instrumentor, client, backend):
    """A real child process, configured only by the injected environment."""
    submit(client)
    (job_env,) = backend.received_envs
    job_env = {**job_env, "KUBEFLOW_OTEL_EXPORTER": "console"}  # print spans to stdout

    code = textwrap.dedent(
        """
        from kubeflow_otel.training import training_run
        with training_run() as run:
            with run.epoch(0):
                run.log_metrics({"loss": 0.42})
        """
    )
    base_env = {k: v for k, v in os.environ.items() if not k.startswith(("KUBEFLOW_", "OTEL_"))}
    out = subprocess.run(
        [sys.executable, "-c", code],
        env={**base_env, **job_env},
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )

    client_span = train_span(otel)
    trace_ids = set(re.findall(r'"trace_id": "0x([0-9a-f]{32})"', out.stdout))
    parent_ids = set(re.findall(r'"parent_id": "0x([0-9a-f]{16})"', out.stdout))
    assert trace_ids == {format(client_span.context.trace_id, "032x")}
    assert format(client_span.context.span_id, "016x") in parent_ids
    assert '"kubeflow.trainjob.name": "run-7"' in out.stdout  # from OTEL_RESOURCE_ATTRIBUTES


def test_job_reports_time_from_submission_to_first_code(otel, instrumentor, client, backend):
    import time

    submit(client)
    (job_env,) = backend.received_envs
    assert job_env[sc.ENV_SUBMIT_TIME].isdigit()

    # pretend the job spent 5 s on scheduling and dependency installation
    job_env = {**job_env, sc.ENV_SUBMIT_TIME: str(time.time_ns() - 5_000_000_000)}
    with training_run(environ=job_env):
        pass

    run_span = next(s for s in otel.finished_spans() if s.name == "kubeflow.trainjob.run")
    assert 5.0 <= run_span.attributes[sc.TRAINING_STARTUP_DELAY] < 6.0
    (startup,) = otel.metric_points(sc.METRIC_TRAINING_STARTUP)
    assert startup.count == 1 and 5.0 <= startup.sum < 6.0
    assert startup.attributes == {sc.TRAINJOB_NAME: "run-7"}


@pytest.mark.parametrize("raw", ["", "not-a-number", str(10**30)])
def test_missing_or_bad_submit_time_is_ignored(otel, raw):
    with training_run(environ={sc.ENV_SUBMIT_TIME: raw}):
        pass
    (span,) = otel.finished_spans()
    assert sc.TRAINING_STARTUP_DELAY not in span.attributes
    assert otel.metric_points(sc.METRIC_TRAINING_STARTUP) == []
