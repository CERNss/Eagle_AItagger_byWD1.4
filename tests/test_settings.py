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
