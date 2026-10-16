import pytest
from fake_backend import FakeBackend, make_job
from kubeflow.trainer import CustomTrainer, LocalProcessBackendConfig, TrainerClient
from kubeflow.trainer.constants import constants as kf_constants
from kubeflow.trainer.options.common import Name
from opentelemetry.trace import SpanKind, StatusCode

from kubeflow_otel import TrainerClientInstrumentor
from kubeflow_otel import semconv as sc


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


def span_named(otel, name):
    matches = [s for s in otel.finished_spans() if s.name == name]
    assert len(matches) == 1, [s.name for s in otel.finished_spans()]
    return matches[0]


def test_status_constants_match_the_sdk():
    assert sc.STATUS_RUNNING == kf_constants.TRAINJOB_RUNNING
    assert sc.STATUS_COMPLETE == kf_constants.TRAINJOB_COMPLETE
    assert sc.STATUS_FAILED == kf_constants.TRAINJOB_FAILED


# ------------------------------------------------------------------ lifecycle


def test_disabled_telemetry_delegates_without_spans(instrumentor, client):
    # no `otel` fixture: telemetry stays disabled
    assert (
        client.train(runtime="torch-distributed", trainer=CustomTrainer(func=train_fn)) == "job-1"
    )
    assert hasattr(TrainerClient.train, "_kubeflow_otel_original")


def test_uninstrument_restores_original_methods(client):
    original = TrainerClient.train
    inst = TrainerClientInstrumentor()
    inst.instrument()
    assert TrainerClient.train is not original
    inst.uninstrument()
    assert TrainerClient.train is original


def test_double_instrument_does_not_double_wrap(instrumentor):
    wrapped = TrainerClient.train
    TrainerClientInstrumentor().instrument()  # second instrumentor is ignored
    assert TrainerClient.train is wrapped
    assert not hasattr(wrapped._kubeflow_otel_original, "_kubeflow_otel_original")


# ---------------------------------------------------------------------- train


def test_train_span_and_metrics(otel, instrumentor, client):
    name = client.train(
        runtime="torch-distributed", trainer=CustomTrainer(func=train_fn, num_nodes=2)
    )

    span = span_named(otel, "kubeflow.trainer.train")
    assert span.kind == SpanKind.CLIENT
    assert span.attributes[sc.OPERATION] == "train"
    assert span.attributes[sc.BACKEND] == "fake"
    assert span.attributes[sc.RUNTIME_NAME] == "torch-distributed"
    assert span.attributes[sc.TRAINER_TYPE] == "CustomTrainer"
    assert span.attributes[sc.TRAINJOB_NUM_NODES] == 2
    assert span.attributes[sc.TRAINJOB_NAME] == name

    (submitted,) = otel.metric_points(sc.METRIC_JOBS_SUBMITTED)
    assert submitted.value == 1
    assert submitted.attributes == {
        sc.BACKEND: "fake",
        sc.RUNTIME_NAME: "torch-distributed",
        sc.TRAINER_TYPE: "CustomTrainer",
    }
    (duration,) = otel.metric_points(sc.METRIC_OPERATION_DURATION)
    assert duration.count == 1
    assert duration.attributes == {sc.OPERATION: "train", sc.BACKEND: "fake"}


def test_train_name_option_is_recorded_before_submission(otel, instrumentor, client):
    client.train(
        runtime="torch-distributed", trainer=CustomTrainer(func=train_fn), options=[Name("my-run")]
    )
    assert span_named(otel, "kubeflow.trainer.train").attributes[sc.TRAINJOB_NAME] == "my-run"


def test_train_failure_is_recorded_and_propagated(otel, instrumentor, client, backend):
    backend.train_error = ValueError("Runtime 'nope' not found")
    with pytest.raises(ValueError, match="not found"):
        client.train(runtime="nope", trainer=CustomTrainer(func=train_fn))

    span = span_named(otel, "kubeflow.trainer.train")
    assert span.status.status_code == StatusCode.ERROR
    assert span.attributes[sc.ERROR_TYPE] == "ValueError"
    assert otel.metric_points(sc.METRIC_JOBS_SUBMITTED) == []
    (duration,) = otel.metric_points(sc.METRIC_OPERATION_DURATION)
    assert duration.attributes[sc.ERROR_TYPE] == "ValueError"


def test_client_span_nests_under_the_callers_span(otel, instrumentor, client):
    with otel.telemetry.tracer.start_span("notebook-cell"):
        client.train(runtime="torch-distributed", trainer=CustomTrainer(func=train_fn))
    train = span_named(otel, "kubeflow.trainer.train")
    parent = span_named(otel, "notebook-cell")
    assert train.parent.span_id == parent.context.span_id


# --------------------------------------------------------- job observation


def test_terminal_job_is_counted_once(otel, instrumentor, client, backend):
    backend.jobs["done"] = make_job("done", "Complete", age_s=42)
    client.get_job("done")
    client.get_job("done")
    client.list_jobs()

    (finished,) = otel.metric_points(sc.METRIC_JOBS_FINISHED)
    assert finished.value == 1
    assert finished.attributes[sc.TRAINJOB_STATUS] == "Complete"
    assert sc.TRAINJOB_NAME not in finished.attributes  # keep label cardinality low
    (job_duration,) = otel.metric_points(sc.METRIC_JOB_DURATION)
    assert job_duration.count == 1
    assert 40 <= job_duration.sum <= 60

    get_spans = [s for s in otel.finished_spans() if s.name == "kubeflow.trainer.get_job"]
    assert get_spans[0].attributes[sc.TRAINJOB_STATUS] == "Complete"
    assert span_named(otel, "kubeflow.trainer.list_jobs").attributes[sc.RESULT_COUNT] == 1


def test_wait_records_status_transitions_and_startup(otel, instrumentor, client, backend):
    backend.jobs["job-1"] = make_job("job-1", "Created", age_s=3)
    backend.wait_sequence = ["Created", "Running", "Running", "Complete"]
    seen = []
    user_callbacks = [lambda job: seen.append(job.status)]

    job = client.wait_for_job_status("job-1", callbacks=user_callbacks, polling_interval=1)

    assert job.status == "Complete"
    assert seen == ["Created", "Running", "Running", "Complete"]  # user callback untouched
    assert len(user_callbacks) == 1  # caller's list not mutated

    span = span_named(otel, "kubeflow.trainer.wait_for_job_status")
    events = [
        (e.attributes.get(sc.TRAINJOB_PREVIOUS_STATUS), e.attributes[sc.TRAINJOB_STATUS])
        for e in span.events
        if e.name == sc.EVENT_STATUS_CHANGE
    ]
    assert events == [(None, "Created"), ("Created", "Running"), ("Running", "Complete")]
    assert span.attributes[sc.WAIT_TARGET_STATUS] == "Complete"
    assert span.attributes[sc.WAIT_POLL_INTERVAL] == 1

    (startup,) = otel.metric_points(sc.METRIC_JOB_STARTUP)
    assert startup.count == 1
    (finished,) = otel.metric_points(sc.METRIC_JOBS_FINISHED)
    assert finished.attributes[sc.TRAINJOB_STATUS] == "Complete"


def test_wait_on_failed_job(otel, instrumentor, client, backend):
    backend.jobs["job-1"] = make_job("job-1", "Running")
    backend.wait_sequence = ["Running", "Failed"]
    with pytest.raises(RuntimeError):
        client.wait_for_job_status("job-1")

    span = span_named(otel, "kubeflow.trainer.wait_for_job_status")
    assert span.status.status_code == StatusCode.ERROR
    (finished,) = otel.metric_points(sc.METRIC_JOBS_FINISHED)
    assert finished.attributes[sc.TRAINJOB_STATUS] == "Failed"


def test_broken_telemetry_never_breaks_the_call(otel, instrumentor, client, backend, monkeypatch):
    calls = []

    def explode(*_args, **_kwargs):
        calls.append(1)
        raise RuntimeError("exporter on fire")

    inst = instrumentor._instruments_for(otel.telemetry)
    monkeypatch.setattr(inst.job_startup, "record", explode)
    monkeypatch.setattr(inst.jobs_finished, "add", explode)
    monkeypatch.setattr(inst.jobs_submitted, "add", explode)

    backend.wait_sequence = ["Running", "Complete"]
    assert client.train(runtime="torch-distributed", trainer=CustomTrainer(func=train_fn))
    assert client.wait_for_job_status("job-1").status == "Complete"
    assert len(calls) == 3  # submitted.add, startup.record, finished.add all blew up


# ----------------------------------------------------------------------- logs


def test_logs_span_covers_iteration_and_is_not_current(otel, instrumentor, client, backend):
    backend.jobs["job-1"] = make_job("job-1", "Running")
    lines = []
    for line in client.get_job_logs("job-1"):
        # spans started while iterating must not become children of the logs span
        with otel.telemetry.tracer.start_span("user-work"):
            lines.append(line)

    assert len(lines) == 3
    logs = span_named(otel, "kubeflow.trainer.get_job_logs")
    assert logs.attributes[sc.LOGS_LINES] == 3
    assert logs.attributes[sc.LOGS_FOLLOW] is False
    for user_span in (s for s in otel.finished_spans() if s.name == "user-work"):
        assert user_span.parent is None or user_span.parent.span_id != logs.context.span_id


# ------------------------------------------------------------- other methods


def test_remaining_operations_are_traced(otel, instrumentor, client, backend):
    backend.jobs["job-1"] = make_job("job-1", "Running")
    client.list_runtimes()
    client.get_runtime("torch-distributed")
    client.get_job_events("job-1")
    client.delete_job("job-1")
    with pytest.raises(ValueError):
        client.get_runtime("missing")

    names = [s.name for s in otel.finished_spans()]
    assert names == [
        "kubeflow.trainer.list_runtimes",
        "kubeflow.trainer.get_runtime",
        "kubeflow.trainer.get_job_events",
        "kubeflow.trainer.delete_job",
        "kubeflow.trainer.get_runtime",
    ]
    ops = {p.attributes[sc.OPERATION] for p in otel.metric_points(sc.METRIC_OPERATION_DURATION)}
    assert ops == {"list_runtimes", "get_runtime", "get_job_events", "delete_job"}


def test_auto_module_instruments_on_import():
    import subprocess
    import sys

    code = (
        "import kubeflow_otel.auto\n"
        "from kubeflow.trainer import TrainerClient\n"
        "print(hasattr(TrainerClient.train, '_kubeflow_otel_original'))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "True"
