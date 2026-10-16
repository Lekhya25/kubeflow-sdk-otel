"""Run a small hyperparameter sweep of real PyTorch training jobs through the Kubeflow SDK.

Each job runs via the SDK's local-process backend (its own virtualenv, launched with
torchrun) and reports epochs, loss and accuracy with kubeflow_otel.training. The
client side is instrumented, so every job's trace starts at TrainerClient.train().

Normally started by Docker Compose (see docs/demo.md):

    docker compose -f deploy/docker-compose.yml --profile demo run --rm demo
"""

from __future__ import annotations

import argparse
import time

from kubeflow.trainer import CustomTrainer, LocalProcessBackendConfig, TrainerClient
from kubeflow.trainer.options.common import Name

import kubeflow_otel
from kubeflow_otel import TrainerClientInstrumentor

# Environment for the training processes. The local backend gives a job *only*
# trainer.env, so everything it needs is listed here (kubeflow_otel adds the
# trace context and collector settings on top).
JOB_ENV = {
    "HOME": "/tmp",
    "PIP_NO_INDEX": "1",  # install from the image's wheelhouse, never the network
    "PIP_FIND_LINKS": "/wheels",
    "PIP_NO_CACHE_DIR": "1",
    "PET_STANDALONE": "1",  # torchrun --standalone: a free rendezvous port per job
    "OMP_NUM_THREADS": "2",
}

# lr, hidden units: one config diverges (fails), one is deliberately slow (straggler)
SWEEP = [
    {"lr": 0.001, "hidden": 64},
    {"lr": 0.01, "hidden": 64},
    {"lr": 0.05, "hidden": 128},
    {"lr": 0.1, "hidden": 128},
    {"lr": 8.0, "hidden": 64},  # diverges -> FloatingPointError -> job Failed
    {"lr": 0.05, "hidden": 1024},  # same lr, much larger model -> slow epochs
]


def train_mlp(params):
    """Training function executed inside each TrainJob (imports must live inside)."""
    import math
    import time

    import torch

    from kubeflow_otel.training import training_run

    torch.manual_seed(params["seed"])
    torch.set_num_threads(2)

    # Synthetic but learnable task: a hidden nonlinear rule over 20 features.
    n, d, hidden, bs = params["samples"], 20, params["hidden"], params["batch_size"]
    x = torch.randn(n, d)
    w = torch.randn(d)
    y = ((x @ w + 1.5 * torch.sin(3 * x[:, 0]) * x[:, 1]) > 0).long()
    split = int(0.8 * n)
    x_train, y_train, x_val, y_val = x[:split], y[:split], x[split:], y[split:]

    model = torch.nn.Sequential(
        torch.nn.Linear(d, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, hidden),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden, 2),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    with training_run() as run:
        for epoch in range(params["epochs"]):
            with run.epoch(epoch):
                model.train()
                order = torch.randperm(split)
                total = 0.0
                for i in range(0, split, bs):
                    idx = order[i : i + bs]
                    optimizer.zero_grad()
                    loss = loss_fn(model(x_train[idx]), y_train[idx])
                    loss.backward()
                    optimizer.step()
                    total += loss.item() * len(idx)
                train_loss = total / split
                if not math.isfinite(train_loss):
                    raise FloatingPointError(
                        f"training diverged at epoch {epoch} (lr={params['lr']})"
                    )
                model.eval()
                with torch.no_grad():
                    accuracy = (model(x_val).argmax(1) == y_val).float().mean().item()
                run.log_metrics({"loss": train_loss, "accuracy": accuracy})
                print(f"epoch {epoch}: loss={train_loss:.4f} accuracy={accuracy:.3f}", flush=True)
                time.sleep(params["epoch_pause"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--concurrency", type=int, default=3, help="jobs running at once")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--samples", type=int, default=40_000)
    parser.add_argument("--epoch-pause", type=float, default=1.0, help="seconds between epochs")
    args = parser.parse_args()

    telemetry = kubeflow_otel.get_telemetry()
    print(f"telemetry enabled: {telemetry.enabled} -> {telemetry.config.otlp_endpoint}")
    TrainerClientInstrumentor().instrument()
    client = TrainerClient(backend_config=LocalProcessBackendConfig())

    run_id = time.strftime("%H%M%S")
    configs = [
        {
            **cfg,
            "name": f"lr{cfg['lr']}-h{cfg['hidden']}-{run_id}".replace(".", "p"),
            "seed": i,
            "epochs": args.epochs,
            "samples": args.samples,
            "batch_size": 256,
            "epoch_pause": args.epoch_pause,
        }
        for i, cfg in enumerate(SWEEP)
    ]

    # One parent span for the whole sweep: every client call and every job (with
    # its epochs) lands in a single trace, with the jobs as parallel branches.
    with telemetry.tracer.start_span(
        "kubeflow.demo.sweep", attributes={"demo.trials": len(configs), "demo.run_id": run_id}
    ):
        results = run_sweep(client, configs, args.concurrency)

    kubeflow_otel.shutdown()
    print("\nsummary:")
    for job, status in results.items():
        print(f"  {status:9s} {job}")


def run_sweep(client: TrainerClient, configs: list[dict], concurrency: int) -> dict[str, str]:
    results: dict[str, str] = {}
    for start in range(0, len(configs), concurrency):
        batch = configs[start : start + concurrency]
        submitted = []
        for cfg in batch:
            job = client.train(
                runtime="torch-distributed",
                trainer=CustomTrainer(
                    func=train_mlp,
                    func_args=cfg,
                    packages_to_install=["kubeflow-sdk-otel[otel]"],
                    env=JOB_ENV,
                ),
                options=[Name(cfg["name"])],
            )
            print(f"submitted {job}")
            submitted.append(job)

        for job in submitted:
            # The local backend does not raise on failure (other backends do), so
            # wait for either terminal status and inspect it.
            final = client.wait_for_job_status(
                job, status={"Complete", "Failed"}, timeout=1200, polling_interval=2
            )
            results[job] = final.status
            print(f"{job}: {final.status}")
            if final.status == "Failed":
                tail = list(client.get_job_logs(job))[-3:]
                print("  last log lines:", *tail, sep="\n    ")

        client.list_jobs()
        for job in submitted:
            client.delete_job(job)
    return results


if __name__ == "__main__":
    main()
