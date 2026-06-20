from __future__ import annotations

import os

import pytest

from service import config as config_module


@pytest.fixture(autouse=True)
def _reset_config_cache():
    config_module.reset_config_cache()
    yield
    config_module.reset_config_cache()


def test_missing_files_is_noop(monkeypatch, tmp_path):
    monkeypatch.setenv("CONFIG_PATH", str(tmp_path / "nope.yaml"))
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.delenv("FOO_BAR", raising=False)
    assert config_module.config_value("FOO_BAR") is None


def test_yaml_value_used_when_env_absent(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("default_threshold: 0.37\nuse_chinese_name: true\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.delenv("DEFAULT_THRESHOLD", raising=False)
    assert config_module.config_value("DEFAULT_THRESHOLD") == "0.37"
    assert config_module.config_value("USE_CHINESE_NAME") == "true"


def test_env_overrides_yaml(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("default_threshold: 0.37\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.setenv("DEFAULT_THRESHOLD", "0.9")
    assert config_module.config_value("DEFAULT_THRESHOLD") == "0.9"


def test_interpolation_default_then_env(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("otel_exporter_headers: ${OTEL_TOKEN:-none}\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.delenv("OTEL_EXPORTER_HEADERS", raising=False)
    monkeypatch.delenv("OTEL_TOKEN", raising=False)
    assert config_module.config_value("OTEL_EXPORTER_HEADERS") == "none"

    config_module.reset_config_cache()
    monkeypatch.setenv("OTEL_TOKEN", "Authorization=Bearer x")
    assert config_module.config_value("OTEL_EXPORTER_HEADERS") == "Authorization=Bearer x"


def test_dotenv_populates_environment_for_interpolation(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("SECRET_HEADER=abc123\n", encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text("otel_exporter_headers: ${SECRET_HEADER}\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    monkeypatch.setenv("ENV_FILE", str(env_file))
    monkeypatch.delenv("OTEL_EXPORTER_HEADERS", raising=False)
    monkeypatch.delenv("SECRET_HEADER", raising=False)
    try:
        assert config_module.config_value("OTEL_EXPORTER_HEADERS") == "abc123"
    finally:
        os.environ.pop("SECRET_HEADER", None)


def test_list_values_flatten_to_csv(monkeypatch, tmp_path):
    cfg = tmp_path / "config.yaml"
    cfg.write_text("exclude_tags:\n  - a\n  - b c\n", encoding="utf-8")
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "nope.env"))
    monkeypatch.delenv("EXCLUDE_TAGS", raising=False)
    assert config_module.config_value("EXCLUDE_TAGS") == "a,b c"


def test_settings_load_from_yaml(monkeypatch, tmp_path):
    from service.settings import Settings

    cfg = tmp_path / "config.yaml"
    cfg.write_text(
        "default_top_k: 7\nliveness_failure_threshold: 9\nmax_image_pixels: 1000\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("CONFIG_PATH", str(cfg))
    monkeypatch.setenv("ENV_FILE", str(tmp_path / "nope.env"))
    for name in ("DEFAULT_TOP_K", "LIVENESS_FAILURE_THRESHOLD", "MAX_IMAGE_PIXELS"):
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()
    assert settings.default_top_k == 7
    assert settings.liveness_failure_threshold == 9
    assert settings.max_image_pixels == 1000
