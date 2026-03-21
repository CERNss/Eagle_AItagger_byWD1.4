from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any
import warnings

from PIL import Image
import pandas as pd
from loguru import logger
from opentelemetry import trace as otel_trace
from opentelemetry.trace import Status, StatusCode

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover - platform dependent in dev envs
    ort = None
    _ORT_IMPORT_ERROR = exc
else:
    _ORT_IMPORT_ERROR = None

from .image_utils import ImageUtils
from .logging import reset_image_path_context, set_image_path_context
from .observability import metrics_recorder
from .settings import Settings

_TRACER = otel_trace.get_tracer("eagle.ai.tagger.runtime")
_PYNVML_AVAILABLE: bool | None = None  # None = not yet checked
_PYNVML_MODULE: Any = None


class TaggerRuntime:
    RATING_TAG_COUNT = 4
    TAG_ESCAPE_PATTERN = re.compile(r"([\\()])")

    def __init__(self, settings: Settings):
        self.settings = settings
        self.session: Any | None = None
        self.provider = "uninitialized"
        self.input_name = ""
        self.output_name = ""
        self.target_size = 0
        self.is_loaded = False
        self.has_chinese_names = False
        self.english_names: list[str] = []
        self.chinese_names: list[str] = []
        self._nvml_initialized = False
        self._nvml_handle = None

    def load(self) -> None:
        with _TRACER.start_as_current_span("runtime.load") as span:
            self.settings.validate()
            if ort is None:
                raise RuntimeError(
                    "onnxruntime is not installed; install onnxruntime/onnxruntime-gpu for this platform"
                ) from _ORT_IMPORT_ERROR
            if not self.settings.model_path.exists():
                raise FileNotFoundError(f"model file not found: {self.settings.model_path}")
            if not self.settings.tags_path.exists():
                raise FileNotFoundError(f"tags file not found: {self.settings.tags_path}")

            available_providers = ort.get_available_providers()
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available_providers
                else ["CPUExecutionProvider"]
            )

            span.set_attribute("runtime.providers.available", ",".join(available_providers))
            self.session = ort.InferenceSession(str(self.settings.model_path), providers=providers)
            self.provider = self.session.get_providers()[0]
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            self.target_size = self._resolve_target_size(self.session.get_inputs()[0].shape)
            span.set_attribute("runtime.provider", self.provider)
            logger.info(
                "runtime.loaded",
                provider=self.provider,
                model=self.settings.model_path.name,
                tags_csv=self.settings.tags_path.name,
            )
            if "CUDA" in self.provider.upper():
                self._init_nvml()

            tags_df = pd.read_csv(self.settings.tags_path)
            if "name" not in tags_df.columns:
                raise ValueError("tags csv must include a 'name' column")

            self.has_chinese_names = "right_tag_cn" in tags_df.columns
            content_tags = tags_df.iloc[self.RATING_TAG_COUNT :]
            self.english_names = content_tags["name"].fillna("").astype(str).tolist()
            if self.has_chinese_names:
                self.chinese_names = content_tags["right_tag_cn"].fillna("").astype(str).tolist()
            else:
                self.chinese_names = self.english_names[:]

            self.is_loaded = True
            span.set_attribute("runtime.loaded", True)

    def _resolve_target_size(self, shape: list[Any]) -> int:
        for dim in shape[1:3]:
            if isinstance(dim, int):
                return dim
        raise ValueError(f"unable to determine target image size from model input shape: {shape}")

    def resolve_image_path(self, image_path: str | Path) -> Path:
        candidate = Path(image_path).expanduser()
        if not candidate.is_absolute() and self.settings.image_root is not None:
            candidate = self.settings.image_root / candidate

        resolved = candidate.resolve(strict=False)
        if self.settings.image_root is not None:
            root = self.settings.image_root.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as exc:
                raise PermissionError(f"image path is outside IMAGE_ROOT: {resolved}") from exc

        if not resolved.exists():
            raise FileNotFoundError(f"image not found: {resolved}")
        if not resolved.is_file():
            raise ValueError(f"path is not a file: {resolved}")
        return resolved

    def predict(
        self,
        image_path: str | Path,
        threshold: float | None = None,
        use_chinese_name: bool | None = None,
        top_k: int | None = None,
    ) -> tuple[Path, list[dict[str, Any]], int]:
        if not self.is_loaded or self.session is None:
            raise RuntimeError("runtime is not loaded")

        resolved_path = self.resolve_image_path(image_path)
        image_ctx = set_image_path_context(resolved_path)
        started = time.perf_counter()
        try:
            with _TRACER.start_as_current_span("runtime.predict") as predict_span:
                predict_span.set_attribute("runtime.provider", self.provider)
                try:
                    with _TRACER.start_as_current_span("image.preprocess"):
                        with Image.open(resolved_path) as image:
                            processed = ImageUtils.preprocess_image(image, self.target_size)
                    with _TRACER.start_as_current_span("onnx.inference"):
                        scores = self.session.run([self.output_name], {self.input_name: processed})[0][0]
                    with _TRACER.start_as_current_span("tags.postprocess"):
                        tags = self._postprocess_scores(
                            scores=scores,
                            threshold=self.settings.default_threshold if threshold is None else threshold,
                            use_chinese_name=(
                                self.settings.default_use_chinese_name
                                if use_chinese_name is None
                                else use_chinese_name
                            ),
                            top_k=self.settings.default_top_k if top_k is None else top_k,
                        )
                    elapsed_ms = int((time.perf_counter() - started) * 1000)
                    predict_span.set_attribute("runtime.elapsed_ms", elapsed_ms)
                    predict_span.set_attribute("runtime.tag_count", len(tags))
                    self._record_gpu_metrics()
                    logger.info(
                        "runtime.predict.complete",
                        provider=self.provider,
                        elapsed_ms=elapsed_ms,
                        tag_count=len(tags),
                    )
                    return resolved_path, tags, elapsed_ms
                except Exception as exc:
                    predict_span.record_exception(exc)
                    predict_span.set_status(Status(StatusCode.ERROR, str(exc)))
                    raise
        finally:
            reset_image_path_context(image_ctx)

    def predict_batch(
        self,
        image_paths: list[str],
        threshold: float | None = None,
        use_chinese_name: bool | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        if len(image_paths) > self.settings.batch_limit:
            raise ValueError(
                f"batch size {len(image_paths)} exceeds configured limit {self.settings.batch_limit}"
            )

        with _TRACER.start_as_current_span("runtime.predict_batch") as span:
            span.set_attribute("runtime.batch_size", len(image_paths))
            results: list[dict[str, Any]] = []
            for image_path in image_paths:
                try:
                    resolved_path, tags, elapsed_ms = self.predict(
                        image_path=image_path,
                        threshold=threshold,
                        use_chinese_name=use_chinese_name,
                        top_k=top_k,
                    )
                    results.append(
                        {
                            "image_path": str(resolved_path),
                            "success": True,
                            "tags": tags,
                            "elapsed_ms": elapsed_ms,
                            "error": None,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "image_path": str(image_path),
                            "success": False,
                            "tags": [],
                            "elapsed_ms": None,
                            "error": str(exc),
                            "error_type": exc.__class__.__name__,
                        }
                    )
                    logger.warning(
                        "runtime.predict.failed",
                        error=str(exc),
                        error_type=exc.__class__.__name__,
                    )
            failure_count = sum(1 for item in results if not item["success"])
            span.set_attribute("runtime.failures", failure_count)
            logger.info(
                "runtime.predict_batch.complete",
                total=len(image_paths),
                failures=failure_count,
                provider=self.provider,
            )
            return results

    def _postprocess_scores(
        self,
        scores: list[float],
        threshold: float,
        use_chinese_name: bool,
        top_k: int,
    ) -> list[dict[str, Any]]:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        names = (
            self.chinese_names
            if use_chinese_name and self.has_chinese_names
            else self.english_names
        )
        content_scores = scores[self.RATING_TAG_COUNT :]
        if len(content_scores) != len(names):
            raise ValueError(
                "model output does not match the number of configured tag labels"
            )
        raw_tags = {
            name: float(score)
            for name, score in zip(names, content_scores)
            if name
        }

        for tag in self.settings.additional_tags:
            raw_tags.setdefault(tag, 1.0)

        filtered = {
            tag: score
            for tag, score in raw_tags.items()
            if score >= threshold and tag not in self.settings.exclude_tags
        }

        illegal_chars = {"[", "]", ",", "(", ")", "\\"}
        normalized: list[tuple[str, float]] = []
        for tag, score in filtered.items():
            if len(tag) == 1 and tag in illegal_chars:
                continue
            normalized_tag = tag
            if self.settings.replace_underscore and tag not in self.settings.underscore_excludes:
                normalized_tag = normalized_tag.replace("_", " ")
            if self.settings.escape_tags:
                normalized_tag = self.TAG_ESCAPE_PATTERN.sub(r"\\\1", normalized_tag)
            normalized.append((normalized_tag, score))

        if self.settings.sort_alphabetically:
            normalized.sort(key=lambda item: item[0])
        else:
            normalized.sort(key=lambda item: (-item[1], item[0]))

        return [
            {"name": tag, "score": round(score, 6)}
            for tag, score in normalized[:top_k]
        ]

    def _init_nvml(self) -> None:
        pynvml = self._get_pynvml_module()
        if pynvml is None or self._nvml_initialized:
            return
        try:
            pynvml.nvmlInit()
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self._nvml_initialized = True
        except Exception as exc:  # pragma: no cover - NVML optional
            logger.warning("nvml.init.failed", error=str(exc))

    def _read_gpu_memory(self) -> int | None:
        pynvml = self._get_pynvml_module()
        if pynvml is None or not self._nvml_initialized or self._nvml_handle is None:
            return None
        try:
            info = pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
            return int(info.used)
        except Exception as exc:  # pragma: no cover
            logger.warning("nvml.read.failed", error=str(exc))
            return None

    def _record_gpu_metrics(self) -> None:
        if "CUDA" not in self.provider.upper():
            return
        memory_bytes = self._read_gpu_memory()
        if memory_bytes is None:
            return
        metrics_recorder().record_gpu(self.provider, memory_bytes)

    def shutdown(self) -> None:
        pynvml = self._get_pynvml_module()
        if self._nvml_initialized and pynvml is not None:
            try:
                pynvml.nvmlShutdown()
                logger.info("nvml.shutdown.ok")
            except Exception as exc:
                logger.warning("nvml.shutdown.failed", error=str(exc))
            finally:
                self._nvml_initialized = False
                self._nvml_handle = None

    @staticmethod
    def _get_pynvml_module() -> Any | None:
        global _PYNVML_AVAILABLE, _PYNVML_MODULE
        if _PYNVML_AVAILABLE is False:
            return None
        if _PYNVML_AVAILABLE is None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    import pynvml as pynvml_module
                _PYNVML_MODULE = pynvml_module
                _PYNVML_AVAILABLE = True
            except ImportError:  # pragma: no cover - optional dependency
                _PYNVML_AVAILABLE = False
                return None
        return _PYNVML_MODULE
