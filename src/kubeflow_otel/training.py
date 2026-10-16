"""Telemetry for code running *inside* a TrainJob.

When the client is instrumented, each submitted job receives the client's
trace context and collector settings in its environment. ``training_run``
picks them up so the job's spans join the same trace as ``client.train()``:

    def train_func():
        from kubeflow_otel.training import training_run   # import inside the function

        with training_run() as run:
            for epoch in range(epochs):
                with run.epoch(epoch):
                    loss = train_one_epoch()
                    run.log_metrics({"loss": loss})

Without the injected environment it still works: it starts a new trace, or does
nothing at all if telemetry is disabled.
"""

from __future__ import annotations

import os
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from types import TracebackType

from kubeflow_otel import api
from kubeflow_otel import semconv as sc
from kubeflow_otel.propagation import carrier_from_env
from kubeflow_otel.telemetry import Telemetry, force_flush, get_telemetry


class TrainingRun:
    """Handle for the active training run; created by :func:`training_run`."""

    def __init__(self, telemetry: Telemetry, span: api.Span, labels: dict[str, str | int]) -> None:
        self._telemetry = telemetry
        self.span = span
        self._labels = labels
        meter = telemetry.meter
        self._metric = meter.create_gauge(
            sc.METRIC_TRAINING_METRIC, description="Latest value of a training metric."
        )
        self._epoch_gauge = meter.create_gauge(
            sc.METRIC_TRAINING_EPOCH, unit="{epoch}", description="Current epoch."
        )
        self._epoch_duration = meter.create_histogram(
            sc.METRIC_EPOCH_DURATION,
            unit="s",
            description="Wall-clock duration of training epochs.",
            boundaries=sc.EPOCH_BUCKETS,
        )
        self._startup = meter.create_histogram(
            sc.METRIC_TRAINING_STARTUP,
            unit="s",
            description="Time from client submission until the training code started.",
            boundaries=sc.JOB_STARTUP_BUCKETS,
        )
        self.current_epoch: int | None = None

    def epoch(self, epoch: int) -> _EpochScope:
        """Context manager timing one epoch as a child span of the run."""
        return _EpochScope(self, epoch)

    def log_metrics(self, metrics: Mapping[str, float], step: int | None = None) -> None:
        """Record metric values as gauges and as an event on the run span."""
        step = self.current_epoch if step is None else step
        event: dict[str, float | int] = {}
        for name, value in metrics.items():
            value = float(value)
            self._metric.set(value, {**self._labels, sc.TRAINING_METRIC_NAME: name})
            event[name] = value
        if step is not None:
            event["step"] = step
        self.span.add_event(sc.EVENT_TRAINING_METRICS, event)

    def _epoch_finished(self, epoch: int, seconds: float) -> None:
        self._epoch_gauge.set(epoch, self._labels)
        self._epoch_duration.record(seconds, self._labels)


class _EpochScope:
    __slots__ = ("_run", "_epoch", "_span", "_start")

    def __init__(self, run: TrainingRun, epoch: int) -> None:
        self._run = run
        self._epoch = epoch

    def __enter__(self) -> api.Span:
        self._run.current_epoch = self._epoch
        self._start = time.perf_counter()
        self._span = self._run._telemetry.tracer.start_span(
            "kubeflow.trainjob.epoch", attributes={sc.TRAINING_EPOCH: self._epoch}
        )
        return self._span.__enter__()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        try:
            self._run._epoch_finished(self._epoch, time.perf_counter() - self._start)
        finally:
            self._span.__exit__(exc_type, exc, tb)
        return False


@contextmanager
def training_run(
    name: str = "kubeflow.trainjob.run",
    *,
    environ: Mapping[str, str] | None = None,
    telemetry: Telemetry | None = None,
    flush: bool = True,
) -> Iterator[TrainingRun]:
    """Trace a training run, continuing the submitting client's trace if present.

    Args:
        name: Name of the root span for this run.
        environ: Where to read the injected context from (default ``os.environ``).
        telemetry: Explicit telemetry (default: configured from the environment).
        flush: Export buffered telemetry when the run ends. Training processes
            usually exit right after, so this defaults to True.
    """
    env = os.environ if environ is None else environ
    telemetry = telemetry or get_telemetry()

    labels: dict[str, str | int] = {
        sc.TRAINJOB_NAME: env.get(sc.ENV_TRAINJOB_NAME) or "unknown",
    }
    attributes: dict[str, str | int] = dict(labels)
    if env.get(sc.ENV_RUNTIME_NAME):
        attributes[sc.RUNTIME_NAME] = env[sc.ENV_RUNTIME_NAME]
    # torchrun sets RANK / WORLD_SIZE for each worker process
    for key, attr in (("RANK", sc.TRAINING_RANK), ("WORLD_SIZE", sc.TRAINING_WORLD_SIZE)):
        if env.get(key, "").isdigit():
            attributes[attr] = int(env[key])
    if sc.TRAINING_RANK in attributes:
        labels[sc.TRAINING_RANK] = attributes[sc.TRAINING_RANK]

    delay = _startup_delay_seconds(env)
    if delay is not None:
        attributes[sc.TRAINING_STARTUP_DELAY] = delay

    span = telemetry.tracer.start_span(
        name, kind="server", attributes=attributes, parent_carrier=carrier_from_env(env) or None
    )
    try:
        with span:
            run = TrainingRun(telemetry, span, labels)
            if delay is not None:
                run._startup.record(delay, labels)
            yield run
    finally:
        if flush:
            force_flush()


def _startup_delay_seconds(env: Mapping[str, str]) -> float | None:
    """Seconds from client submission to training code starting.

    Covers scheduling, image pulls and dependency installation. Compares the
    client's clock with the job's, so it assumes reasonably synchronized clocks
    (always true for the local backend; NTP-synced nodes on Kubernetes).
    """
    raw = env.get(sc.ENV_SUBMIT_TIME, "")
    if not raw.isdigit():
        return None
    delay = (time.time_ns() - int(raw)) / 1e9
    return round(delay, 3) if delay >= 0 else None
