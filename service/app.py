from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, status

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
    RUNTIME.load()
    yield


app = FastAPI(title="Eagle AI Tagger Service", lifespan=lifespan)


@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=RUNTIME.is_loaded,
        provider=RUNTIME.provider,
        model_path=str(SETTINGS.model_path),
        tags_path=str(SETTINGS.tags_path),
    )


@app.get("/readyz", response_model=ReadyResponse)
def readyz() -> ReadyResponse:
    if not RUNTIME.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="runtime is not ready",
        )
    return ReadyResponse(status="ready", provider=RUNTIME.provider)


@app.post("/tag", response_model=TagResponse)
def tag_image(request: TagRequest) -> TagResponse:
    try:
        resolved_path, tags, elapsed_ms = RUNTIME.predict(
            image_path=request.image_path,
            threshold=request.threshold,
            use_chinese_name=request.use_chinese_name,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise _http_exception_for_error(exc) from exc

    return TagResponse(
        provider=RUNTIME.provider,
        image_path=str(resolved_path),
        tags=tags,
        elapsed_ms=elapsed_ms,
    )


@app.post("/tag/batch", response_model=BatchTagResponse)
def tag_batch(request: BatchTagRequest) -> BatchTagResponse:
    try:
        results = RUNTIME.predict_batch(
            image_paths=request.image_paths,
            threshold=request.threshold,
            use_chinese_name=request.use_chinese_name,
            top_k=request.top_k,
        )
    except Exception as exc:
        raise _http_exception_for_error(exc) from exc

    return BatchTagResponse(provider=RUNTIME.provider, results=results)
