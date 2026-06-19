from __future__ import annotations

import pytest

from service.settings import Settings


def _clear_env(monkeypatch, names: list[str]) -> None:
    for name in names:
        monkeypatch.delenv(name, raising=False)


def test_observability_defaults(monkeypatch):
    _clear_env(
        monkeypatch,
        [
            "LOG_LEVEL",
            "LOG_FORMAT",
            "OTEL_ENABLED",
            "OTEL_METRICS_ENABLED",
            "OTEL_EXPORTER_ENDPOINT",
            "OTEL_TRACE_SAMPLE_RATIO",
        ],
    )
    settings = Settings.from_env()
    obs = settings.observability
    assert obs.logging.log_format == "json"
    assert obs.logging.level == "INFO"
    assert obs.tracing.enabled is False
    assert obs.metrics.enabled is False


def test_model_and_tag_path_defaults(monkeypatch):
    _clear_env(monkeypatch, ["MODEL_PATH", "TAGS_PATH"])
    settings = Settings.from_env()
    assert str(settings.model_path) == "model/swinv2-v3.onnx"
    assert str(settings.tags_path) == "csv/Tags-cn_2024_ver-1.0.csv"
    assert settings.max_concurrent_inference == 1
    assert settings.inference_acquire_timeout_seconds == 30.0
    assert settings.startup_self_check is True
    assert settings.require_cuda is False


def test_invalid_trace_sample_ratio(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_ENDPOINT", "http://collector:4318")
    monkeypatch.setenv("OTEL_TRACE_SAMPLE_RATIO", "2")
    settings = Settings.from_env()
    with pytest.raises(ValueError):
        settings.validate()


def test_tracing_requires_exporter_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "1")
    monkeypatch.delenv("OTEL_EXPORTER_ENDPOINT", raising=False)
    settings = Settings.from_env()
    with pytest.raises(ValueError):
        settings.validate()


def test_metrics_can_reuse_trace_exporter_endpoint(monkeypatch):
    monkeypatch.setenv("OTEL_ENABLED", "1")
    monkeypatch.setenv("OTEL_METRICS_ENABLED", "1")
    monkeypatch.setenv("OTEL_EXPORTER_ENDPOINT", "http://collector:4318")
    monkeypatch.delenv("OTEL_METRICS_EXPORTER_ENDPOINT", raising=False)
    settings = Settings.from_env()
    settings.validate()
    assert settings.observability.metrics.exporter_endpoint == "http://collector:4318"


def test_invalid_stability_settings(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCE", "0")
    settings = Settings.from_env()
    with pytest.raises(ValueError, match="MAX_CONCURRENT_INFERENCE"):
        settings.validate()

    monkeypatch.setenv("MAX_CONCURRENT_INFERENCE", "1")
    monkeypatch.setenv("INFERENCE_ACQUIRE_TIMEOUT_SECONDS", "-1")
    settings = Settings.from_env()
    with pytest.raises(ValueError, match="INFERENCE_ACQUIRE_TIMEOUT_SECONDS"):
        settings.validate()


def test_stability_settings_from_env(monkeypatch):
    monkeypatch.setenv("MAX_CONCURRENT_INFERENCE", "2")
    monkeypatch.setenv("INFERENCE_ACQUIRE_TIMEOUT_SECONDS", "0.5")
    monkeypatch.setenv("STARTUP_SELF_CHECK", "false")
    monkeypatch.setenv("REQUIRE_CUDA", "true")

    settings = Settings.from_env()

    assert settings.max_concurrent_inference == 2
    assert settings.inference_acquire_timeout_seconds == 0.5
    assert settings.startup_self_check is False
    assert settings.require_cuda is True
