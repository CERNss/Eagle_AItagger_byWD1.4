from __future__ import annotations

from pydantic import BaseModel, Field


class TagRecord(BaseModel):
    name: str
    score: float


class TagRequest(BaseModel):
    image_path: str
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    use_chinese_name: bool | None = None
    top_k: int | None = Field(default=None, gt=0, le=4096)


class TagResponse(BaseModel):
    provider: str
    image_path: str
    tags: list[TagRecord]
    elapsed_ms: int


class BatchTagRequest(BaseModel):
    image_paths: list[str] = Field(min_length=1)
    threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    use_chinese_name: bool | None = None
    top_k: int | None = Field(default=None, gt=0, le=4096)


class BatchTagItem(BaseModel):
    image_path: str
    success: bool
    tags: list[TagRecord] = Field(default_factory=list)
    elapsed_ms: int | None = None
    error: str | None = None


class BatchTagResponse(BaseModel):
    provider: str
    results: list[BatchTagItem]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    provider: str
    model_path: str
    tags_path: str


class ReadyResponse(BaseModel):
    status: str
    provider: str
