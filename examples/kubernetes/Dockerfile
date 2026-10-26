# Training-node image for the Kubernetes example: CPU PyTorch plus kubeflow-sdk-otel.
# kubeflow-sdk-otel isn't on PyPI, so it's baked in rather than pip-installed by each pod.
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir --retries 10 --timeout 120 torch==2.5.1 \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[otel]" && rm -rf /src

WORKDIR /workspace
