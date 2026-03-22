from __future__ import annotations

import hashlib
import logging
import sys
from contextvars import ContextVar
from pathlib import Path
from typing import Any

from loguru import logger
from opentelemetry import trace

from .settings import Settings

_CONFIGURED = False
_INCLUDE_TRACE = True
_HASH_IMAGE_PATHS = True
_SERVICE_METADATA: dict[str, str] = {}

_REQUEST_ID = ContextVar("request_id", default=None)
_IMAGE_HASH = ContextVar("image_hash", default=None)


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno
        logger.bind(logger_name=record.name).opt(
            depth=6, exception=record.exc_info
        ).log(level, record.getMessage())


def setup_logging(settings: Settings) -> None:
    global _CONFIGURED, _SERVICE_METADATA, _INCLUDE_TRACE, _HASH_IMAGE_PATHS
    if _CONFIGURED:
        return
    observability = settings.observability
    _SERVICE_METADATA = {
        "service.name": observability.service_name,
        "service.version": observability.service_version,
        "deployment.environment": observability.deployment_environment,
    }
    _INCLUDE_TRACE = observability.logging.include_trace_context
    _HASH_IMAGE_PATHS = observability.logging.hash_image_paths

    logger.remove()
    logger.configure(
        patcher=_patch_record,
        extra={
            "service.name": observability.service_name,
            "service.version": observability.service_version,
            "deployment.environment": observability.deployment_environment,
        }
    )
    intercept = _InterceptHandler()
    logger.add(
        sys.stdout,
        level=observability.logging.level,
        serialize=observability.logging.log_format == "json",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )

    logging.basicConfig(handlers=[intercept], level=0, force=True)
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "fastapi"):
        logging.getLogger(name).handlers = [intercept]
    _CONFIGURED = True


def _patch_record(record: dict[str, Any]) -> None:
    request_id = _REQUEST_ID.get()
    image_hash = _IMAGE_HASH.get()
    if request_id:
        record["extra"]["request_id"] = request_id
    if image_hash:
        record["extra"]["image_hash"] = image_hash
    if _INCLUDE_TRACE:
        span = trace.get_current_span()
        span_ctx = span.get_span_context()
        if span_ctx and span_ctx.is_valid:
            record["extra"]["trace_id"] = f"{span_ctx.trace_id:032x}"
            record["extra"]["span_id"] = f"{span_ctx.span_id:016x}"
    for key, value in _SERVICE_METADATA.items():
        record["extra"][key] = value


def set_request_context(request_id: str | None):
    return _REQUEST_ID.set(request_id)


def reset_request_context(token) -> None:
    if token is not None:
        _REQUEST_ID.reset(token)


def set_image_path_context(path: str | Path | None):
    hashed = None
    if path:
        value = str(path)
        hashed = hashlib.sha256(value.encode("utf-8")).hexdigest() if _HASH_IMAGE_PATHS else value
    return _IMAGE_HASH.set(hashed)


def reset_image_path_context(token) -> None:
    if token is not None:
        _IMAGE_HASH.reset(token)
