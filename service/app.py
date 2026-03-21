from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request, status
from loguru import logger

from .runtime import TaggerRuntime
from .schemas import (
    BatchTagRequest,
    BatchTagResponse,
    HealthResponse,
    ReadyResponse,
    TagRequest,
    TagResponse,
)
from .settings import get_settings
from .logging import reset_request_context, set_request_context
from .observability import metrics_recorder, observability_status, setup_observability, shutdown_observability

SETTINGS = get_settings()
RUNTIME = TaggerRuntime(SETTINGS)


def _http_exception_for_error(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(exc, PermissionError):
        return HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc))
    if isinstance(exc, OSError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    return HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@asynccontextmanager
async def lifespan(_: FastAPI):
    try:
        RUNTIME.load()
    except Exception:
        shutdown_observability()
        raise
    try:
        yield
    finally:
        logger.info("service.shutdown")
        RUNTIME.shutdown()
        shutdown_observability()


app = FastAPI(title="Eagle AI Tagger Service", lifespan=lifespan)
setup_observability(app, SETTINGS)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    token = set_request_context(request_id)
    started = time.perf_counter()
    logger.bind(path=request.url.path, method=request.method, request_id=request_id).info("request.start")
    try:
        response = await call_next(request)
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.bind(
            path=request.url.path,
            method=request.method,
            request_id=request_id,
            elapsed_ms=elapsed_ms,
        ).info("request.complete")
    except Exception:
        logger.exception(
            "request.failed",
            path=request.url.path,
            method=request.method,
            request_id=request_id,
        )
        raise
    finally:
        reset_request_context(token)
    response.headers["x-request-id"] = request_id
    return response


def _observability_payload() -> dict[str, bool | str | None]:
    status_info = observability_status()
    return {
        "logging_mode": status_info.logging_mode,
        "tracing_enabled": status_info.tracing_enabled,
        "metrics_enabled": status_info.metrics_enabled,
    }


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    payload = _observability_payload()
    return HealthResponse(
        status="ok",
        model_loaded=RUNTIME.is_loaded,
        provider=RUNTIME.provider,
        model_path=str(SETTINGS.model_path),
        tags_path=str(SETTINGS.tags_path),
        **payload,
    )


@app.get("/readyz", response_model=ReadyResponse)
def readyz() -> ReadyResponse:
    if not RUNTIME.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="runtime is not ready",
        )
    payload = _observability_payload()
    return ReadyResponse(status="ready", provider=RUNTIME.provider, **payload)


@app.post("/tag", response_model=TagResponse)
def tag_image(request: TagRequest) -> TagResponse:
    started = time.perf_counter()
    try:
        resolved_path, tags, elapsed_ms = RUNTIME.predict(
            image_path=request.image_path,
            threshold=request.threshold,
            use_chinese_name=request.use_chinese_name,
            top_k=request.top_k,
        )
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        metrics_recorder().record_error("/tag", RUNTIME.provider, exc.__class__.__name__)
        metrics_recorder().record_request("/tag", "error", RUNTIME.provider, elapsed)
        raise _http_exception_for_error(exc) from exc

    total_elapsed = int((time.perf_counter() - started) * 1000)
    metrics_recorder().record_request("/tag", "success", RUNTIME.provider, total_elapsed)
    return TagResponse(
        provider=RUNTIME.provider,
        image_path=str(resolved_path),
        tags=tags,
        elapsed_ms=elapsed_ms,
    )


@app.post("/tag/batch", response_model=BatchTagResponse)
def tag_batch(request: BatchTagRequest) -> BatchTagResponse:
    started = time.perf_counter()
    try:
        results = RUNTIME.predict_batch(
            image_paths=request.image_paths,
            threshold=request.threshold,
            use_chinese_name=request.use_chinese_name,
            top_k=request.top_k,
        )
    except Exception as exc:
        elapsed = int((time.perf_counter() - started) * 1000)
        metrics_recorder().record_error("/tag/batch", RUNTIME.provider, exc.__class__.__name__)
        metrics_recorder().record_request(
            "/tag/batch",
            "error",
            RUNTIME.provider,
            elapsed,
            batch_size=len(request.image_paths),
        )
        raise _http_exception_for_error(exc) from exc

    total_elapsed = int((time.perf_counter() - started) * 1000)
    metrics_recorder().record_request(
        "/tag/batch",
        "success",
        RUNTIME.provider,
        total_elapsed,
        batch_size=len(request.image_paths),
    )
    for item in results:
        if item.get("success") is False:
            error_class = item.get("error_type") or "UnknownError"
            metrics_recorder().record_error("/tag/batch", RUNTIME.provider, error_class)
    return BatchTagResponse(provider=RUNTIME.provider, results=results)
