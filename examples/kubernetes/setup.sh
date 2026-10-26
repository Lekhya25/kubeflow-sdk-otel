#!/usr/bin/env bash
# Create a local Kubernetes cluster (kind) with Kubeflow Trainer, the torch-otel runtime,
# an OpenTelemetry Collector and Jaeger. Run from the repository root.
#
# Requires: docker, kind (>= 0.31), kubectl (within one minor version of Kubernetes 1.35).
set -euo pipefail

CLUSTER=kfotel
NODE_IMAGE="kindest/node:v1.35.0@sha256:452d707d4862f52530247495d180205e029056831160e22870e37e3f6c1ac31f"
TRAINER_REF=v2.3.0   # matches kubeflow==0.5.0 (kubeflow-trainer-api >=2.3,<2.4)
RUNTIME_IMAGE=kubeflow-sdk-otel-runtime:0.1.0

kind create cluster --name "$CLUSTER" --image "$NODE_IMAGE" --wait 120s

# Kubeflow Trainer controller (installs the TrainJob/TrainingRuntime CRDs and JobSet)
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=$TRAINER_REF"
kubectl -n kubeflow-system wait --for=condition=Available deployment --all --timeout=300s

# Small CPU training image with kubeflow-sdk-otel baked in, loaded straight into the node
docker build -f examples/kubernetes/Dockerfile -t "$RUNTIME_IMAGE" .
kind load docker-image "$RUNTIME_IMAGE" --name "$CLUSTER"

kubectl apply -f examples/kubernetes/runtime.yaml
kubectl apply -f examples/kubernetes/jaeger.yaml -f examples/kubernetes/collector.yaml
kubectl -n observability wait --for=condition=Available deploy/jaeger deploy/otel-collector --timeout=300s

echo
echo "Ready. In two other terminals:"
echo "  kubectl -n observability port-forward svc/otel-collector 4317:4317"
echo "  kubectl -n observability port-forward svc/jaeger 16686:16686"
echo "then: python examples/kubernetes/run_on_kind.py"
