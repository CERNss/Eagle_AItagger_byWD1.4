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

    @classmethod
    def from_env(cls) -> "Settings":
        image_root = os.getenv("IMAGE_ROOT")
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
        )

    def validate(self) -> None:
        if not 0 <= self.default_threshold <= 1:
            raise ValueError("DEFAULT_THRESHOLD must be between 0 and 1")
        if self.default_top_k <= 0:
            raise ValueError("DEFAULT_TOP_K must be greater than 0")
        if self.batch_limit <= 0:
            raise ValueError("BATCH_LIMIT must be greater than 0")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
