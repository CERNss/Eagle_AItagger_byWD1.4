"""Layered configuration loader: config.yaml + .env + ${} interpolation.

Resolution precedence for every setting key (UPPER_SNAKE_CASE):

1. An explicit process environment variable (this also covers values injected
   from ``.env`` via python-dotenv, and anything set by docker-compose).
2. The matching key from ``config.yaml`` (the canonical, non-secret config).
3. The built-in default baked into :mod:`service.settings`.

``config.yaml`` string values may embed ``${VAR}`` or ``${VAR:-default}``
placeholders. They are resolved against the process environment, so secrets
live in ``.env`` (kept out of version control) while ``config.yaml`` only
references them by name.

If neither ``config.yaml`` nor ``.env`` is present the loader is a no-op and
behaviour is identical to a pure environment-variable deployment, which keeps
existing Docker/CI setups and the test-suite working unchanged.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

try:  # optional dependency; degrade to env-only when missing
    import yaml
except ImportError:  # pragma: no cover - exercised only without pyyaml
    yaml = None  # type: ignore[assignment]

try:  # optional dependency; degrade to env-only when missing
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - exercised only without python-dotenv
    load_dotenv = None  # type: ignore[assignment]


_INTERPOLATION = re.compile(r"\$\{([^}:]+?)(?::-([^}]*))?\}")

_loaded = False
_flat_config: dict[str, str] = {}


def _interpolate(value: str) -> str:
    def _replace(match: re.Match[str]) -> str:
        name = match.group(1).strip()
        default = match.group(2)
        env_value = os.environ.get(name)
        if env_value is not None:
            return env_value
        return default if default is not None else ""

    return _INTERPOLATION.sub(_replace, value)


def _coerce(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return _interpolate(value)
    if isinstance(value, (list, tuple)):
        return ",".join(_coerce(item) for item in value)
    return str(value)


def _flatten(data: dict[str, Any], flat: dict[str, str]) -> None:
    for key, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            # Nested sections are purely cosmetic; leaves map by their own key.
            _flatten(value, flat)
            continue
        flat[str(key).upper()] = _coerce(value)


def load_config() -> dict[str, str]:
    """Load ``.env`` into the environment and parse ``config.yaml`` once."""
    global _loaded, _flat_config
    if _loaded:
        return _flat_config

    env_file = os.environ.get("ENV_FILE", ".env")
    if load_dotenv is not None and env_file and Path(env_file).is_file():
        load_dotenv(env_file, override=False)

    flat: dict[str, str] = {}
    config_path = os.environ.get("CONFIG_PATH", "config.yaml")
    if yaml is not None and config_path and Path(config_path).is_file():
        with open(config_path, "r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if isinstance(data, dict):
            _flatten(data, flat)

    _flat_config = flat
    _loaded = True
    return _flat_config


def config_value(name: str) -> str | None:
    """Return the resolved value for ``name`` or ``None`` if unset."""
    load_config()
    env_value = os.environ.get(name)
    if env_value is not None:
        return env_value
    return _flat_config.get(name)


def reset_config_cache() -> None:
    """Drop the cached config so the next access reloads files (tests only)."""
    global _loaded, _flat_config
    _loaded = False
    _flat_config = {}
