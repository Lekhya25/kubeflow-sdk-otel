"""Tracing and metrics for ``kubeflow.trainer.TrainerClient``, added from outside.

    from kubeflow_otel.instrumentation import TrainerClientInstrumentor
    TrainerClientInstrumentor().instrument()

``instrument()`` wraps the public ``TrainerClient`` methods at class level, the
same pattern as the ``opentelemetry-instrumentation-*`` packages. Every client
(including ones created by other libraries) is covered without code changes,
and ``uninstrument()`` restores the original methods.

When telemetry is disabled, a wrapped call costs one global read and one
attribute check before delegating to the original method.
"""

from __future__ import annotations

import copy
import dataclasses
import functools
import inspect
import logging
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from kubeflow_otel import api
from kubeflow_otel import semconv as sc
from kubeflow_otel.propagation import job_environment, merge_job_environment
from kubeflow_otel.telemetry import Telemetry, get_telemetry

logger = logging.getLogger(__name__)

_ORIGINAL_ATTR = "_kubeflow_otel_original"
_MAX_TRACKED_JOBS = 10_000


@dataclass(frozen=True)
class _Instruments:
    telemetry: Telemetry
    operation_duration: api.Histogram
    jobs_submitted: api.Counter
    jobs_finished: api.Counter
    job_duration: api.Histogram
    job_startup: api.Histogram

    @classmethod
    def build(cls, telemetry: Telemetry) -> _Instruments:
        m = telemetry.meter
        return cls(
            telemetry=telemetry,
            operation_duration=m.create_histogram(
                sc.METRIC_OPERATION_DURATION,
                unit="s",
                description="Duration of TrainerClient operations.",
                boundaries=sc.OPERATION_BUCKETS,
            ),
            jobs_submitted=m.create_counter(
                sc.METRIC_JOBS_SUBMITTED,
                unit="{job}",
                description="TrainJobs successfully submitted through TrainerClient.train().",
            ),
            jobs_finished=m.create_counter(
                sc.METRIC_JOBS_FINISHED,
                unit="{job}",
                description="TrainJobs observed reaching a terminal status (counted once per job).",
            ),
            job_duration=m.create_histogram(
                sc.METRIC_JOB_DURATION,
                unit="s",
                description="Time from TrainJob creation until a terminal status was observed.",
                boundaries=sc.JOB_DURATION_BUCKETS,
            ),
            job_startup=m.create_histogram(
                sc.METRIC_JOB_STARTUP,
                unit="s",
                description="Time from TrainJob creation until it was observed Running.",
                boundaries=sc.JOB_STARTUP_BUCKETS,
            ),
        )


class TrainerClientInstrumentor:
    """Adds spans and metrics to every ``TrainerClient`` operation.

    Args to :meth:`instrument`:
        telemetry: Use this telemetry instead of the process-wide one from
            :func:`kubeflow_otel.get_telemetry` (mainly for tests).
    """

    _lock = threading.Lock()
    _active: TrainerClientInstrumentor | None = None

    def __init__(self) -> None:
        self._telemetry: Telemetry | None = None
        self._instruments: _Instruments | None = None
        self._instruments_lock = threading.Lock()
        self._finished_jobs: OrderedDict[tuple[str, str], None] = OrderedDict()
        self._finished_lock = threading.Lock()
        self._client_cls: type | None = None

    # ------------------------------------------------------------ lifecycle

    @property
    def is_instrumented(self) -> bool:
        return TrainerClientInstrumentor._active is self

    def instrument(self, telemetry: Telemetry | None = None) -> None:
        from kubeflow.trainer import TrainerClient

        with TrainerClientInstrumentor._lock:
            if TrainerClientInstrumentor._active is not None:
                logger.warning("TrainerClient is already instrumented; ignoring")
                return
            self._telemetry = telemetry
            self._client_cls = TrainerClient
            for name, spec in _SPECS.items():
                original = getattr(TrainerClient, name, None)
                if original is None or hasattr(original, _ORIGINAL_ATTR):
                    continue
                setattr(TrainerClient, name, self._wrap(spec, original))
            TrainerClientInstrumentor._active = self

    def uninstrument(self) -> None:
        with TrainerClientInstrumentor._lock:
            if TrainerClientInstrumentor._active is not self or self._client_cls is None:
                return
            for name in _SPECS:
                current = getattr(self._client_cls, name, None)
                original = getattr(current, _ORIGINAL_ATTR, None)
                if original is not None:
                    setattr(self._client_cls, name, original)
            TrainerClientInstrumentor._active = None

    # ------------------------------------------------------------- wrapping

    def _wrap(self, spec: _Spec, original: Callable[..., Any]) -> Callable[..., Any]:
        signature = inspect.signature(original)
        invoke = self._invoke_logs if spec.streams else self._invoke

        @functools.wraps(original)
        def wrapper(client: Any, *args: Any, **kwargs: Any) -> Any:
            telemetry = self._telemetry or get_telemetry()
            if not telemetry.enabled:
                return original(client, *args, **kwargs)
            return invoke(telemetry, spec, signature, original, client, args, kwargs)

        setattr(wrapper, _ORIGINAL_ATTR, original)
        return wrapper

    def _instruments_for(self, telemetry: Telemetry) -> _Instruments:
        inst = self._instruments
        if inst is not None and inst.telemetry is telemetry:
            return inst
        with self._instruments_lock:
            if self._instruments is None or self._instruments.telemetry is not telemetry:
                self._instruments = _Instruments.build(telemetry)
            return self._instruments

    def _invoke(
        self,
        telemetry: Telemetry,
        spec: _Spec,
        signature: inspect.Signature,
        original: Callable[..., Any],
        client: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        inst = self._instruments_for(telemetry)
        ctx = _CallContext(self, inst, client, _bind(signature, client, args, kwargs))
        attributes = {sc.OPERATION: spec.operation, sc.BACKEND: ctx.backend}
        if spec.before is not None:
            attributes.update(_safely(spec.before, ctx) or {})

        error_type: str | None = None
        start = time.perf_counter()
        with telemetry.tracer.start_span(
            spec.span_name, kind="client", attributes=attributes
        ) as span:
            ctx.span = span
            call_args, call_kwargs = (client, *args), kwargs
            if spec.prepare is not None and ctx.bound is not None:
                prepared = _safely(spec.prepare, ctx)
                if prepared is not None:
                    call_args, call_kwargs = prepared
            try:
                result = original(*call_args, **call_kwargs)
            except BaseException as exc:
                error_type = type(exc).__qualname__
                raise
            finally:
                ctx.record_operation(spec.operation, time.perf_counter() - start, error_type)
            if spec.after is not None:
                _safely(spec.after, ctx, result)
            return result

    def _invoke_logs(
        self,
        telemetry: Telemetry,
        spec: _Spec,
        signature: inspect.Signature,
        original: Callable[..., Any],
        client: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Iterator[str]:
        """get_job_logs returns a (possibly endless) iterator: trace its consumption.

        The span is *not* made current, because a generator yields control back
        to the caller and must not leak its context into the caller's code.
        """
        inst = self._instruments_for(telemetry)
        ctx = _CallContext(self, inst, client, _bind(signature, client, args, kwargs))
        attributes = {sc.OPERATION: spec.operation, sc.BACKEND: ctx.backend}
        attributes.update(_safely(_logs_before, ctx) or {})
        follow = bool(attributes.get(sc.LOGS_FOLLOW, False))

        span = telemetry.tracer.start_span(spec.span_name, kind="client", attributes=attributes)
        start = time.perf_counter()
        try:
            iterator = iter(original(client, *args, **kwargs))
        except BaseException as exc:
            span.record_exception(exc)
            span.end()
            ctx.record_operation(
                spec.operation, time.perf_counter() - start, type(exc).__qualname__
            )
            raise
        return _traced_lines(iterator, span, ctx, spec.operation, start, record_duration=not follow)

    # -------------------------------------------------------- job tracking

    def _observe_job(self, ctx: _CallContext, job: Any) -> None:
        """Count a job once when it is first seen in a terminal status."""
        status = getattr(job, "status", None)
        name = getattr(job, "name", None)
        if status not in sc.TERMINAL_STATUSES or not name:
            return
        key = (ctx.backend, name)
        with self._finished_lock:
            if key in self._finished_jobs:
                return
            self._finished_jobs[key] = None
            if len(self._finished_jobs) > _MAX_TRACKED_JOBS:
                self._finished_jobs.popitem(last=False)
        attrs = _job_metric_attributes(ctx.backend, job)
        ctx.inst.jobs_finished.add(1, attrs)
        age = _job_age_seconds(job)
        if age is not None:
            ctx.inst.job_duration.record(age, attrs)


@dataclass
class _CallContext:
    instrumentor: TrainerClientInstrumentor
    inst: _Instruments
    client: Any
    bound: inspect.BoundArguments | None
    span: api.Span | None = None

    def __post_init__(self) -> None:
        self.backend = _backend_name(self.client)

    def arg(self, name: str, default: Any = None) -> Any:
        if self.bound is None:
            return default
        return self.bound.arguments.get(name, default)

    def record_operation(self, operation: str, seconds: float, error_type: str | None) -> None:
        attrs = {sc.OPERATION: operation, sc.BACKEND: self.backend}
        if error_type:
            attrs[sc.ERROR_TYPE] = error_type
        _safely(self.inst.operation_duration.record, seconds, attrs)


@dataclass(frozen=True)
class _Spec:
    operation: str
    before: Callable[[_CallContext], dict[str, Any] | None] | None = None
    after: Callable[[_CallContext, Any], None] | None = None
    # returns replacement (args, kwargs) for the call, or None to leave them unchanged
    prepare: Callable[[_CallContext], tuple[tuple[Any, ...], dict[str, Any]] | None] | None = None
    streams: bool = False

    @property
    def span_name(self) -> str:
        return f"kubeflow.trainer.{self.operation}"


# ------------------------------------------------------------------ helpers


def _safely(fn: Callable[..., Any], *args: Any) -> Any:
    """Run telemetry code; never let its failure reach the user's call."""
    try:
        return fn(*args)
    except Exception:
        logger.debug("kubeflow_otel instrumentation hook failed", exc_info=True)
        return None


def _bind(
    signature: inspect.Signature, client: Any, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> inspect.BoundArguments | None:
    try:
        bound = signature.bind(client, *args, **kwargs)
    except TypeError:
        return None  # the original call will raise the real TypeError
    bound.apply_defaults()
    return bound


def _backend_name(client: Any) -> str:
    name = type(getattr(client, "backend", None)).__name__
    return name.removesuffix("Backend").lower() or "unknown"


def _runtime_name(runtime: Any) -> str | None:
    if runtime is None or isinstance(runtime, str):
        return runtime
    return getattr(runtime, "name", None)


def _job_attributes(job: Any) -> dict[str, Any]:
    attrs: dict[str, Any] = {}
    for key, value in (
        (sc.TRAINJOB_NAME, getattr(job, "name", None)),
        (sc.TRAINJOB_STATUS, getattr(job, "status", None)),
        (sc.TRAINJOB_NUM_NODES, getattr(job, "num_nodes", None)),
        (sc.RUNTIME_NAME, _runtime_name(getattr(job, "runtime", None))),
    ):
        if isinstance(value, (str, int, float, bool)):
            attrs[key] = value
    return attrs


def _job_metric_attributes(backend: str, job: Any) -> dict[str, Any]:
    # Low-cardinality labels only: never the job name.
    attrs = {sc.BACKEND: backend, sc.TRAINJOB_STATUS: str(getattr(job, "status", "Unknown"))}
    runtime = _runtime_name(getattr(job, "runtime", None))
    if runtime:
        attrs[sc.RUNTIME_NAME] = runtime
    return attrs


def _job_age_seconds(job: Any) -> float | None:
    created = getattr(job, "creation_timestamp", None)
    if not isinstance(created, datetime):
        return None
    now = datetime.now(timezone.utc) if created.tzinfo else datetime.now()
    return max((now - created).total_seconds(), 0.0)


def _option_job_name(options: Any) -> str | None:
    for option in options or ():
        if type(option).__name__ == "Name" and isinstance(getattr(option, "name", None), str):
            return option.name
    return None


def _traced_lines(
    iterator: Iterator[str],
    span: api.Span,
    ctx: _CallContext,
    operation: str,
    start: float,
    record_duration: bool,
) -> Iterator[str]:
    lines = 0
    error_type: str | None = None
    try:
        for line in iterator:
            lines += 1
            yield line
    except GeneratorExit:
        raise
    except BaseException as exc:
        error_type = type(exc).__qualname__
        span.record_exception(exc)
        raise
    finally:
        span.set_attribute(sc.LOGS_LINES, lines)
        span.end()
        if record_duration or error_type:
            ctx.record_operation(operation, time.perf_counter() - start, error_type)


# ------------------------------------------------------- per-operation hooks


def _train_before(ctx: _CallContext) -> dict[str, Any]:
    trainer = ctx.arg("trainer")
    attrs: dict[str, Any] = {}
    runtime = _runtime_name(ctx.arg("runtime"))
    if runtime:
        attrs[sc.RUNTIME_NAME] = runtime
    if trainer is not None:
        attrs[sc.TRAINER_TYPE] = type(trainer).__name__
        num_nodes = getattr(trainer, "num_nodes", None)
        if isinstance(num_nodes, int):
            attrs[sc.TRAINJOB_NUM_NODES] = num_nodes
    name = _option_job_name(ctx.arg("options"))
    if name:
        attrs[sc.TRAINJOB_NAME] = name
    return attrs


def _train_prepare(ctx: _CallContext) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
    """Give the job the current trace context via a *copy* of ``trainer.env``."""
    assert ctx.bound is not None
    telemetry = ctx.inst.telemetry
    trainer = ctx.arg("trainer")
    if not telemetry.config.propagate_to_jobs or trainer is None or not hasattr(trainer, "env"):
        return None  # e.g. BuiltinTrainer has no env

    injected = job_environment(
        telemetry,
        job_name=_option_job_name(ctx.arg("options")),
        runtime_name=_runtime_name(ctx.arg("runtime")),
    )
    env = merge_job_environment(trainer.env, injected)
    if dataclasses.is_dataclass(trainer):
        trainer_copy = dataclasses.replace(trainer, env=env)
    else:
        trainer_copy = copy.copy(trainer)
        trainer_copy.env = env
    ctx.bound.arguments["trainer"] = trainer_copy
    return ctx.bound.args, ctx.bound.kwargs


def _train_after(ctx: _CallContext, job_name: Any) -> None:
    if ctx.span is not None and isinstance(job_name, str):
        ctx.span.set_attribute(sc.TRAINJOB_NAME, job_name)
    attrs: dict[str, Any] = {sc.BACKEND: ctx.backend}
    runtime = _runtime_name(ctx.arg("runtime"))
    if runtime:
        attrs[sc.RUNTIME_NAME] = runtime
    trainer = ctx.arg("trainer")
    if trainer is not None:
        attrs[sc.TRAINER_TYPE] = type(trainer).__name__
    ctx.inst.jobs_submitted.add(1, attrs)


def _name_before(ctx: _CallContext) -> dict[str, Any]:
    name = ctx.arg("name")
    return {sc.TRAINJOB_NAME: name} if isinstance(name, str) else {}


def _job_after(ctx: _CallContext, job: Any) -> None:
    if ctx.span is not None:
        ctx.span.set_attributes(_job_attributes(job))
    ctx.instrumentor._observe_job(ctx, job)


def _list_jobs_before(ctx: _CallContext) -> dict[str, Any]:
    runtime = _runtime_name(ctx.arg("runtime"))
    return {sc.RUNTIME_NAME: runtime} if runtime else {}


def _list_jobs_after(ctx: _CallContext, jobs: Any) -> None:
    if ctx.span is not None:
        ctx.span.set_attribute(sc.RESULT_COUNT, len(jobs))
    for job in jobs:
        ctx.instrumentor._observe_job(ctx, job)


def _count_after(ctx: _CallContext, items: Any) -> None:
    if ctx.span is not None:
        ctx.span.set_attribute(sc.RESULT_COUNT, len(items))


def _runtime_before(ctx: _CallContext) -> dict[str, Any]:
    name = _runtime_name(ctx.arg("name") or ctx.arg("runtime"))
    return {sc.RUNTIME_NAME: name} if name else {}


def _wait_before(ctx: _CallContext) -> dict[str, Any]:
    attrs = _name_before(ctx)
    status = ctx.arg("status")
    if status:
        attrs[sc.WAIT_TARGET_STATUS] = ",".join(sorted(status))
    for key, arg in ((sc.WAIT_TIMEOUT, "timeout"), (sc.WAIT_POLL_INTERVAL, "polling_interval")):
        value = ctx.arg(arg)
        if isinstance(value, (int, float)):
            attrs[key] = value
    return attrs


def _wait_prepare(ctx: _CallContext) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Add an observer callback (without touching the caller's list) to see every poll."""
    assert ctx.bound is not None
    previous: list[str | None] = [None]
    startup_recorded = [False]

    def observe(job: Any) -> None:
        try:
            status = getattr(job, "status", None)
            if status != previous[0] and ctx.span is not None:
                event = {sc.TRAINJOB_STATUS: str(status)}
                if previous[0] is not None:
                    event[sc.TRAINJOB_PREVIOUS_STATUS] = previous[0]
                ctx.span.add_event(sc.EVENT_STATUS_CHANGE, event)
                previous[0] = status
            if status == sc.STATUS_RUNNING and not startup_recorded[0]:
                startup_recorded[0] = True
                age = _job_age_seconds(job)
                if age is not None:
                    ctx.inst.job_startup.record(age, _job_metric_attributes(ctx.backend, job))
            ctx.instrumentor._observe_job(ctx, job)
        except Exception:
            logger.debug("kubeflow_otel wait observer failed", exc_info=True)

    user_callbacks = ctx.bound.arguments.get("callbacks") or []
    ctx.bound.arguments["callbacks"] = [observe, *user_callbacks]
    return ctx.bound.args, ctx.bound.kwargs


def _logs_before(ctx: _CallContext) -> dict[str, Any]:
    attrs = _name_before(ctx)
    step = ctx.arg("step")
    if isinstance(step, str):
        attrs[sc.LOGS_STEP] = step
    attrs[sc.LOGS_FOLLOW] = bool(ctx.arg("follow"))
    return attrs


_SPECS: dict[str, _Spec] = {
    "train": _Spec("train", before=_train_before, after=_train_after, prepare=_train_prepare),
    "get_job": _Spec("get_job", before=_name_before, after=_job_after),
    "list_jobs": _Spec("list_jobs", before=_list_jobs_before, after=_list_jobs_after),
    "wait_for_job_status": _Spec(
        "wait_for_job_status", before=_wait_before, after=_job_after, prepare=_wait_prepare
    ),
    "get_job_logs": _Spec("get_job_logs", streams=True),
    "get_job_events": _Spec("get_job_events", before=_name_before, after=_count_after),
    "delete_job": _Spec("delete_job", before=_name_before),
    "list_runtimes": _Spec("list_runtimes", after=_count_after),
    "get_runtime": _Spec("get_runtime", before=_runtime_before),
    "get_runtime_packages": _Spec("get_runtime_packages", before=_runtime_before),
}
