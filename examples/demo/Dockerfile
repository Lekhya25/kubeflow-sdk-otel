# Demo runner: submits real training jobs through the Kubeflow SDK's local-process
# backend (which needs Linux and bash) with kubeflow-sdk-otel instrumentation.
FROM python:3.11-slim

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Wheelhouse for the training jobs. Every job gets a fresh virtualenv and pip-installs
# torch + kubeflow-sdk-otel into it; serving those from /wheels keeps jobs offline
# and fast instead of downloading ~200 MB per job.
RUN pip download --no-cache-dir --dest /wheels \
      --index-url https://download.pytorch.org/whl/cpu \
      --extra-index-url https://pypi.org/simple \
      torch==2.5.1

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip wheel --no-cache-dir --wheel-dir /wheels ".[otel]" \
 && pip install --no-cache-dir "kubeflow==0.5.0" ".[otel]"

COPY examples/demo/run_demo.py /demo/run_demo.py
WORKDIR /demo
ENTRYPOINT ["python", "/demo/run_demo.py"]
