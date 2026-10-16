# Demo: a hyperparameter sweep of real training jobs

The demo runs **real PyTorch training jobs through the Kubeflow SDK** and sends everything to the [observability stack](observability-stack.md). Nothing is simulated: each job is created with `TrainerClient.train()`, runs on the SDK's local-process backend in its own virtualenv, is launched with `torchrun`, and reports epochs, loss and accuracy with `kubeflow_otel.training`.

```bash
docker compose -f deploy/docker-compose.yml up -d --wait
docker compose -f deploy/docker-compose.yml --profile demo run --rm demo
# longer training, nicer curves:
docker compose -f deploy/docker-compose.yml --profile demo run --rm demo --epochs 30 --epoch-pause 2
```

Then open Grafana (http://localhost:3000) and Jaeger (http://localhost:16686, service `kubeflow-sdk`, operation `kubeflow.demo.sweep`).

## What it runs

[`examples/demo/run_demo.py`](../examples/demo/run_demo.py) submits a six-trial sweep, three jobs at a time:

| Trial | Learning rate | Hidden units | Expected outcome |
|---|---|---|---|
| 1 | 0.001 | 64 | completes |
| 2 | 0.01 | 64 | completes |
| 3 | 0.05 | 128 | completes |
| 4 | 0.1 | 128 | completes |
| 5 | **8.0** | 64 | **diverges**: loss becomes NaN, the job raises `FloatingPointError` and is marked Failed |
| 6 | 0.05 | **1024** | completes, but each epoch is about 2–3× slower (a straggler) |

Each job trains a small MLP on a synthetic, learnable classification task (no dataset download), so the demo is self-contained.

The whole sweep runs under one parent span, `kubeflow.demo.sweep`. Every client call and every job, with all its epochs, therefore lands in **a single trace**.

## How it's packaged

The local-process backend needs Linux and bash, so the demo runs in a container ([`examples/demo/Dockerfile`](../examples/demo/Dockerfile)):

- **Wheelhouse.** For every job, the SDK creates a fresh virtualenv and pip-installs `torch` plus whatever the trainer asks for. The image pre-downloads torch (CPU) and `kubeflow-sdk-otel[otel]` into `/wheels`, and jobs install from there with `PIP_NO_INDEX=1 PIP_FIND_LINKS=/wheels`: offline, with no per-job ~200 MB downloads.
- **Explicit job environment.** The local backend starts each job with *only* `trainer.env` as its environment, so the demo passes everything a job needs there (`HOME`, pip settings, thread count). `kubeflow_otel` adds the trace context and collector settings on top.
- **No port clashes between concurrent jobs.** The SDK launches each job with a bare `torchrun <file>`, which uses the fixed rendezvous port 29500 by default. `PET_STANDALONE=1` (torchrun reads its options from `PET_*` variables) switches to `--standalone`, which picks a free port per job.

## What it showed

![Grafana dashboard after a demo run](images/grafana-dashboard.png)

![Jaeger: the whole sweep as one trace](images/jaeger-sweep-trace.png)

From a 30-epoch run (6 jobs, 3 concurrent):

- **Outcomes:** 5 Complete, 1 Failed (the lr=8.0 trial), 83.3% success. The failed job is red in Jaeger; its `run` and `epoch 0` spans carry the `FloatingPointError` with its stack trace.
- **Stragglers:** mean epoch time was 2.1–2.2 s for the small models and **4.4 s for the 1024-unit model**. Its run span is visibly twice as long.
- **Environment setup dominates short jobs.** The gap in the trace between each `train()` call and its job's first span is the SDK creating a virtualenv and installing torch. The *Time to first epoch* panel measures it: **37–42 s per job** with a warm cache, and **67–120 s** in earlier runs with more contention. With 15 epochs of training (~20 s), setup was **77–87% of each job's wall time**. None of this was visible before: the client-side status reports `Running` as soon as the process starts.

## Kubeflow SDK behaviors found along the way

Running real jobs surfaced two issues in the Kubeflow SDK's local-process backend (version 0.5.0). This project works around them rather than patching the SDK.

1. **A working-directory race between concurrent jobs.** `LocalJob.run()` runs on a thread per job but calls `os.getcwd()` / `os.chdir()`, and the working directory belongs to the whole process. When jobs overlap, one job saves another job's directory as "the directory to return to". After that directory is deleted by cleanup, the restore fails with `FileNotFoundError: /tmp/<other job's venv>` in the job's thread, and the client process can be left in a deleted directory. Job statuses are unaffected. The natural fix is `subprocess.Popen(..., cwd=self.execution_dir)` instead of `os.chdir`.
2. **`wait_for_job_status` doesn't raise when a job fails** on the local backend, unlike the container and Kubernetes backends. Waiting for `{"Complete"}` on a failed job keeps polling until the timeout (10 minutes by default). The demo waits for `{"Complete", "Failed"}` and checks the result.

## Regenerating the screenshots

```bash
pip install -e ".[docs]" && playwright install chromium
python scripts/screenshots.py 45m                 # last 45 minutes
python scripts/screenshots.py <from-ms> <to-ms>   # an exact window, e.g. one demo run
```
