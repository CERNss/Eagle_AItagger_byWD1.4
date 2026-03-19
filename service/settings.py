from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return default if value is None else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return default if value is None else float(value)


def _env_list(name: str) -> tuple[str, ...]:
    value = os.getenv(name, "")
    items = [item.strip() for item in value.split(",") if item.strip()]
    return tuple(items)


def _env_headers(name: str) -> tuple[tuple[str, str], ...]:
    value = os.getenv(name, "")
    headers: list[tuple[str, str]] = []
    for raw_pair in value.split(","):
        if not raw_pair.strip():
            continue
        key, _, header_value = raw_pair.partition("=")
        key = key.strip()
        if not key or not header_value:
            continue
        headers.append((key, header_value.strip()))
    return tuple(headers)


@dataclass(frozen=True)
class LoggingSettings:
    level: str
    log_format: str
    include_trace_context: bool
    hash_image_paths: bool


@dataclass(frozen=True)
class TracingSettings:
    enabled: bool
    exporter_endpoint: str | None
    headers: tuple[tuple[str, str], ...]
    sample_ratio: float


@dataclass(frozen=True)
class MetricsSettings:
    enabled: bool
    exporter_endpoint: str | None
    headers: tuple[tuple[str, str], ...]
    export_interval_seconds: int


@dataclass(frozen=True)
class ObservabilitySettings:
    logging: LoggingSettings
    tracing: TracingSettings
    metrics: MetricsSettings
    service_name: str
    service_version: str
    deployment_environment: str

    @property
    def logging_mode(self) -> str:
        return self.logging.log_format


@dataclass(frozen=True)
class Settings:
    host: str
    port: int
    model_path: Path
    tags_path: Path
    image_root: Path | None
    default_threshold: float
    default_use_chinese_name: bool
    default_top_k: int
    batch_limit: int
    replace_underscore: bool
    underscore_excludes: tuple[str, ...]
    escape_tags: bool
    additional_tags: tuple[str, ...]
    exclude_tags: tuple[str, ...]
    sort_alphabetically: bool
    observability: ObservabilitySettings

    @classmethod
    def from_env(cls) -> "Settings":
        image_root = os.getenv("IMAGE_ROOT")
        exporter_endpoint = os.getenv("OTEL_EXPORTER_ENDPOINT")
        headers = _env_headers("OTEL_EXPORTER_HEADERS")
        logging_settings = LoggingSettings(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            log_format=os.getenv("LOG_FORMAT", "json").lower(),
            include_trace_context=_env_bool("LOG_INCLUDE_TRACE", True),
            hash_image_paths=_env_bool("LOG_HASH_IMAGE_PATHS", True),
        )
        tracing_settings = TracingSettings(
            enabled=_env_bool("OTEL_ENABLED", False),
            exporter_endpoint=exporter_endpoint,
            headers=headers,
            sample_ratio=_env_float("OTEL_TRACE_SAMPLE_RATIO", 0.1),
        )
        metrics_settings = MetricsSettings(
            enabled=_env_bool("OTEL_METRICS_ENABLED", False),
            exporter_endpoint=os.getenv("OTEL_METRICS_EXPORTER_ENDPOINT", exporter_endpoint),
            headers=headers,
            export_interval_seconds=_env_int("OTEL_METRIC_EXPORT_INTERVAL", 60),
        )
        observability = ObservabilitySettings(
            logging=logging_settings,
            tracing=tracing_settings,
            metrics=metrics_settings,
            service_name=os.getenv("SERVICE_NAME", "eagle-ai-tagger"),
            service_version=os.getenv("SERVICE_VERSION", "dev"),
            deployment_environment=os.getenv("DEPLOYMENT_ENVIRONMENT", "development"),
        )
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=_env_int("PORT", 8000),
            model_path=Path(os.getenv("MODEL_PATH", "model/swinv2-v3.onnx")),
            tags_path=Path(os.getenv("TAGS_PATH", "csv/Tags-cn_2024_ver-1.0.csv")),
            image_root=Path(image_root).expanduser() if image_root else None,
            default_threshold=_env_float("DEFAULT_THRESHOLD", 0.5),
            default_use_chinese_name=_env_bool("USE_CHINESE_NAME", True),
            default_top_k=_env_int("DEFAULT_TOP_K", 50),
            batch_limit=_env_int("BATCH_LIMIT", 64),
            replace_underscore=_env_bool("REPLACE_UNDERSCORE", True),
            underscore_excludes=_env_list("UNDERSCORE_EXCLUDES"),
            escape_tags=_env_bool("ESCAPE_TAGS", False),
            additional_tags=_env_list("ADDITIONAL_TAGS"),
            exclude_tags=_env_list("EXCLUDE_TAGS"),
            sort_alphabetically=_env_bool("SORT_ALPHABETICALLY", False),
            observability=observability,
        )

    def validate(self) -> None:
        if not 0 <= self.default_threshold <= 1:
            raise ValueError("DEFAULT_THRESHOLD must be between 0 and 1")
        if self.default_top_k <= 0:
            raise ValueError("DEFAULT_TOP_K must be greater than 0")
        if self.batch_limit <= 0:
            raise ValueError("BATCH_LIMIT must be greater than 0")
        log_format = self.observability.logging.log_format
        if log_format not in {"json", "text"}:
            raise ValueError("LOG_FORMAT must be 'json' or 'text'")
        sample_ratio = self.observability.tracing.sample_ratio
        if not 0 <= sample_ratio <= 1:
            raise ValueError("OTEL_TRACE_SAMPLE_RATIO must be between 0 and 1")
        interval = self.observability.metrics.export_interval_seconds
        if interval <= 0:
            raise ValueError("OTEL_METRIC_EXPORT_INTERVAL must be greater than 0")
        for endpoint in (
            self.observability.tracing.exporter_endpoint,
            self.observability.metrics.exporter_endpoint,
        ):
            if endpoint is None:
                continue
            if not endpoint.startswith(("http://", "https://")):
                raise ValueError("OTEL exporter endpoints must start with http:// or https://")
        if self.observability.tracing.enabled and not self.observability.tracing.exporter_endpoint:
            raise ValueError("OTEL_ENABLED requires OTEL_EXPORTER_ENDPOINT to be set")
        if self.observability.metrics.enabled and not self.observability.metrics.exporter_endpoint:
            raise ValueError("OTEL_METRICS_ENABLED requires OTEL_METRICS_EXPORTER_ENDPOINT or OTEL_EXPORTER_ENDPOINT")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
