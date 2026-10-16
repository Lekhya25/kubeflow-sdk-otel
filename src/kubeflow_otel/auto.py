"""Instrument the Kubeflow SDK as a side effect of importing this module.

    import kubeflow_otel.auto  # noqa: F401

Telemetry is still controlled by ``KUBEFLOW_OTEL_ENABLED``; with it unset the
wrappers simply delegate to the original methods.
"""

from kubeflow_otel.instrumentation import TrainerClientInstrumentor

instrumentor = TrainerClientInstrumentor()
instrumentor.instrument()
