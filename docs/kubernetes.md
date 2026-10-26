# Running on Kubernetes

The trace propagation is designed around `trainer.env`, which every Kubeflow backend passes to the training process. This example verifies it on the **Kubernetes backend**: a real `TrainJob` on a local [kind](https://kind.sigs.k8s.io/) cluster running Kubeflow Trainer, traced from the client's `train()` call into the training pod.

## Setup

```
laptop (Windows/macOS/Linux)                     kind cluster (Kubernetes 1.35)
 run_on_kind.py                                   ┌───────────────────────────────────────────┐
  TrainerClient(KubernetesBackendConfig) ─────────▶ API server ─▶ Kubeflow Trainer ─▶ JobSet ─▶ pod
  OTLP → localhost:4317 ──port-forward──┐          │                                       │
                                        └──────────▶ otel-collector (Service, :4317) ◀─────┘
  Jaeger API ← localhost:16686 ─port-forward──────▶ jaeger (Service, :16686)   ▲   OTLP via cluster DNS
                                                   │    traces ────────────────┘
                                                   └───────────────────────────────────────────┘
```

| Piece | File | What it is |
|---|---|---|
| Cluster | `examples/kubernetes/setup.sh` | kind v0.31 with node image Kubernetes v1.35.0, pinned by digest |
| Kubeflow Trainer v2.3.0 | installed from the upstream `manifests/overlays/manager` | The controller, JobSet, and the `TrainJob` / `TrainingRuntime` / `ClusterTrainingRuntime` CRDs. v2.3 matches `kubeflow==0.5.0`. |
| Training image | `examples/kubernetes/Dockerfile` | `python:3.11-slim` + CPU PyTorch 2.5.1 + `kubeflow-sdk-otel[otel]`, loaded into the node with `kind load docker-image` (never pulled) |
| Runtime | `examples/kubernetes/runtime.yaml` | A `ClusterTrainingRuntime` named `torch-otel`, mirroring upstream `torch-distributed` but using the small image instead of the multi-GB CUDA one |
| Collector | `examples/kubernetes/collector.yaml` | OTel Collector Deployment, Service and ConfigMap. Traces go to Jaeger; metrics to the debug exporter (Jaeger is traces-only). |
| Jaeger | `examples/kubernetes/jaeger.yaml` | Jaeger v2 Deployment and Service in the `observability` namespace |
| Client | `examples/kubernetes/run_on_kind.py` | Submits a 5-epoch TrainJob, waits, reads logs, then checks Jaeger for one trace spanning both services |

```bash
bash examples/kubernetes/setup.sh
kubectl -n observability port-forward svc/otel-collector 4317:4317   # terminal 2
kubectl -n observability port-forward svc/jaeger 16686:16686         # terminal 3
python examples/kubernetes/run_on_kind.py
kind delete cluster --name kfotel                                    # clean up
```

## Result

```
submitted TrainJob k8s-otel-211854
k8s-otel-211854: Complete
trace fb7f8be1119d55ca470e232728b8c777: 10 spans
  +   0.0s  48.41s  kubeflow-sdk       kubeflow.example.kubernetes
  +   0.0s   0.17s  kubeflow-sdk       kubeflow.trainer.train
  +   0.2s  48.12s  kubeflow-sdk       kubeflow.trainer.wait_for_job_status
  +  41.9s   0.32s  kubeflow-trainjob  kubeflow.trainjob.run
  +  41.9s   0.22s  kubeflow-trainjob  kubeflow.trainjob.epoch   (x5)
  +  48.3s   0.12s  kubeflow-sdk       kubeflow.trainer.get_job_logs
PASS
```

- **One trace across the client and the pod:** the pod's `kubeflow.trainjob.run` span is a child of the client's `kubeflow.trainer.train` span. The `TrainJob`'s container environment carried `TRACEPARENT`, `OTEL_SERVICE_NAME=kubeflow-trainjob`, the job name and the in-cluster collector endpoint.
- **Metrics from both sides** reached the collector (client latency and job lifecycle; in-pod epochs, loss, accuracy, and time to first epoch).
- **Start-up dominates a short job:** submission to the first line of training code took **41.9 s** on the first run and **25.2 s** on a warm rerun (pod scheduling, container start, `torchrun` bootstrap, importing torch), while each epoch took about 0.02–0.2 s.

## What this surfaced

1. **The client and the pods need different collector addresses** (`localhost` through port-forward vs cluster DNS). The library forwards the client's endpoint by default, and user-set `trainer.env` values take precedence, so the pods' endpoint is set there.
2. **`func_args` calling conventions differ between SDK backends:** Kubernetes uses `train(**func_args)`, local-process uses `train(func_args)`. A function written for the local backend crashed on Kubernetes with an "unexpected keyword argument" error until it accepted both conventions.
3. **A traces-only backend rejects OTLP metrics.** Exporting straight to Jaeger produced `UNIMPLEMENTED … MetricsService` errors for every metric batch, which is one more reason to put a collector in front of the backends.
4. **Client spans that end last are exported at shutdown,** so a check that queries Jaeger right away can see a partial trace. The script waits for the final client span.

## Kubernetes concepts used
- **Pod, Deployment, Service:** the collector and Jaeger are Deployments (keep N pods running) behind Services (a stable DNS name and virtual IP). The training pods find the collector at `otel-collector.observability.svc.cluster.local`.
- **Namespace:** observability components live in `observability`, TrainJobs in `default`, and Kubeflow Trainer in `kubeflow-system`.
- **ConfigMap:** the collector's config is mounted as a file.
- **CRDs and controllers:** `TrainJob` is a custom resource. The Trainer controller turns it into a JobSet, and JobSet creates batch Jobs and pods. `ClusterTrainingRuntime` is a cluster-scoped template that jobs reference by name.
- **`kubectl port-forward`:** tunnels a Service port to localhost so the laptop-side client can reach in-cluster services.
- **Image loading:** `kind load docker-image` copies a local image into the node's containerd, and `imagePullPolicy: IfNotPresent` stops Kubernetes from trying to pull it.
- **Debugging a failed job:** `kubectl get trainjob,jobset,jobs,pods`, `kubectl get events --sort-by=.lastTimestamp`, and reading the generated command from `kubectl get trainjob -o json`. The crashed pod had already been deleted after its backoff limit, so its logs were gone.
