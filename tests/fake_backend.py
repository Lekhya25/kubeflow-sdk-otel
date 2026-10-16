"""An in-memory stand-in for a Kubeflow Trainer backend.

Lets the instrumentation tests drive a real ``TrainerClient`` deterministically,
on any OS, without Kubernetes, containers or subprocesses.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import datetime, timedelta

from kubeflow.trainer.backends.localprocess.constants import local_runtimes
from kubeflow.trainer.types import types

RUNTIME = local_runtimes[0]


def make_job(name: str, status: str, age_s: float = 0.0) -> types.TrainJob:
    return types.TrainJob(
        name=name,
        runtime=RUNTIME,
        steps=[],
        num_nodes=1,
        creation_timestamp=datetime.now() - timedelta(seconds=age_s),
        status=status,
    )


class FakeBackend:
    def __init__(self) -> None:
        self.jobs: dict[str, types.TrainJob] = {}
        self.train_error: BaseException | None = None
        self.wait_sequence: list[str] = []
        self.received_envs: list[dict[str, str] | None] = []

    def list_runtimes(self) -> list[types.Runtime]:
        return [RUNTIME]

    def get_runtime(self, name: str) -> types.Runtime:
        if name != RUNTIME.name:
            raise ValueError(f"Runtime '{name}' not found")
        return RUNTIME

    def get_runtime_packages(self, runtime: types.Runtime) -> None:
        return None

    def train(self, runtime=None, initializer=None, trainer=None, options=None) -> str:
        if self.train_error is not None:
            raise self.train_error
        self.received_envs.append(getattr(trainer, "env", None))
        name = "job-1"
        for option in options or []:
            name = getattr(option, "name", name)
        self.jobs[name] = make_job(name, "Created")
        return name

    def get_job(self, name: str) -> types.TrainJob:
        if name not in self.jobs:
            raise ValueError(f"No TrainJob with name {name}")
        return self.jobs[name]

    def list_jobs(self, runtime=None) -> list[types.TrainJob]:
        return list(self.jobs.values())

    def wait_for_job_status(
        self, name, status=frozenset({"Complete"}), timeout=600, polling_interval=2, callbacks=None
    ) -> types.TrainJob:
        job = self.jobs[name]
        for current in self.wait_sequence:
            job.status = current
            for callback in callbacks or []:
                callback(job)
            if current in status:
                return job
            if current == "Failed":
                raise RuntimeError(f"TrainJob {name} failed")
        raise TimeoutError(f"Timeout waiting for TrainJob {name}")

    def get_job_logs(self, name, step="node-0", follow=False) -> Iterator[str]:
        yield from ("epoch 1 loss=0.9", "epoch 2 loss=0.5", "done")

    def get_job_events(self, name) -> list:
        return []

    def delete_job(self, name) -> None:
        self.jobs.pop(name, None)
