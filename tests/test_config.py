import pytest

from kubeflow_otel import TelemetryConfig


def test_defaults_are_disabled():
    cfg = TelemetryConfig.from_env({})
    assert cfg == TelemetryConfig()
    assert cfg.enabled is False
    assert cfg.exporter == "otlp"
    assert cfg.propagate_to_jobs is True


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
def test_enabled_truthy_values(value):
    assert TelemetryConfig.from_env({"KUBEFLOW_OTEL_ENABLED": value}).enabled is True


def test_otel_sdk_disabled_wins():
    env = {"KUBEFLOW_OTEL_ENABLED": "true", "OTEL_SDK_DISABLED": "true"}
    assert TelemetryConfig.from_env(env).enabled is False


def test_standard_otel_variables_are_honoured():
    cfg = TelemetryConfig.from_env(
        {
            "OTEL_SERVICE_NAME": "my-notebook",
            "OTEL_EXPORTER_OTLP_ENDPOINT": "http://collector:4317",
            "OTEL_METRIC_EXPORT_INTERVAL": "2000",
        }
    )
    assert cfg.service_name == "my-notebook"
    assert cfg.otlp_endpoint == "http://collector:4317"
    assert cfg.metric_export_interval_ms == 2000


@pytest.mark.parametrize(
    "env",
    [
        {"KUBEFLOW_OTEL_ENABLED": "maybe"},
        {"KUBEFLOW_OTEL_EXPORTER": "zipkin"},
        {"KUBEFLOW_OTEL_SAMPLE_RATIO": "1.5"},
        {"KUBEFLOW_OTEL_SAMPLE_RATIO": "abc"},
        {"OTEL_METRIC_EXPORT_INTERVAL": "-5"},
    ],
)
def test_invalid_values_fall_back_to_defaults_without_raising(env, caplog):
    cfg = TelemetryConfig.from_env(env)
    defaults = TelemetryConfig()
    assert cfg.enabled == defaults.enabled
    assert cfg.exporter == defaults.exporter
    assert cfg.sample_ratio == defaults.sample_ratio
    assert cfg.metric_export_interval_ms == defaults.metric_export_interval_ms
    assert caplog.records, "a warning should explain the ignored value"
