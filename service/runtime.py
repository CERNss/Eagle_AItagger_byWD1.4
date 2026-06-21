from __future__ import annotations

import math
import os
import re
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
import warnings

from PIL import Image, UnidentifiedImageError
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

from .image_utils import ImageUtils, configure_image_limits
from .logging import reset_image_path_context, set_image_path_context
from .observability import metrics_recorder
from .settings import Settings

_TRACER = otel_trace.get_tracer("eagle.ai.tagger.runtime")
_PYNVML_AVAILABLE: bool | None = None  # None = not yet checked
_PYNVML_MODULE: Any = None

# Errors caused by the request/input itself. They are mapped to 4xx and never
# count against runtime liveness (a bad image must not make the service look
# unhealthy). Everything else is treated as an infrastructure failure.
_CLIENT_ERROR_TYPES = (
    FileNotFoundError,
    PermissionError,
    IsADirectoryError,
    NotADirectoryError,
    UnidentifiedImageError,
    # A decompression bomb is a property of the input image, not the runtime, so
    # it must map to 4xx and never count against liveness. It subclasses plain
    # Exception (not OSError), so it has to be listed explicitly or it would be
    # misclassified as an infra failure and trigger a needless session reload.
    Image.DecompressionBombError,
    ValueError,
)


class InferenceBusyError(RuntimeError):
    """Raised when all inference slots stay busy past the configured timeout."""


def classify_error(exc: Exception) -> str:
    """Return ``"busy"``, ``"client"`` or ``"infra"`` for an inference error."""
    if isinstance(exc, InferenceBusyError):
        return "busy"
    if isinstance(exc, _CLIENT_ERROR_TYPES):
        return "client"
    if isinstance(exc, OSError):
        # Truncated/corrupt image reads surface as OSError; treat as input error.
        return "client"
    return "infra"


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
        self._inference_slots = threading.BoundedSemaphore(settings.max_concurrent_inference)
        # Liveness / self-healing state
        self._state_lock = threading.Lock()
        self._consecutive_failures = 0
        # Start time of every in-flight inference, keyed by a monotonic token.
        # A single shared timestamp would be wrong under MAX_CONCURRENT_INFERENCE
        # > 1: a fast call finishing would clear a slow call's clock and hide a
        # hang. The watchdog/liveness judge by the oldest entry instead.
        self._inflight: dict[int, float] = {}
        self._inflight_seq = 0
        self._reload_lock = threading.Lock()
        # None = never reloaded. Do NOT use 0.0 as a sentinel: the cooldown check
        # compares against time.monotonic(), whose epoch is system boot, so on a
        # freshly-booted host (e.g. a CI runner) monotonic() can be < cooldown and
        # 0.0 would wrongly gate the very first reload.
        self._last_reload_monotonic: float | None = None
        self._watchdog_thread: threading.Thread | None = None
        self._watchdog_stop = threading.Event()
        # Action taken when an inference is detected as hung. Injectable for tests.
        self._fatal_handler: Callable[[int], Any] = os._exit

    # ------------------------------------------------------------------ load

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

            configure_image_limits(self.settings.max_image_pixels)

            available_providers = ort.get_available_providers()
            span.set_attribute("runtime.providers.available", ",".join(available_providers))
            self._build_session_with_retry()
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

            if self.settings.startup_self_check:
                self._validate_model_output_shape()

            self.is_loaded = True
            self._start_watchdog()
            span.set_attribute("runtime.loaded", True)

    def _build_session(self) -> Any:
        """Create (or recreate) the ONNX inference session and cache its I/O."""
        if ort is None:
            raise RuntimeError("onnxruntime is not installed")
        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )
        session = ort.InferenceSession(str(self.settings.model_path), providers=providers)
        provider = session.get_providers()[0]
        if self.settings.require_cuda and "CUDA" not in provider.upper():
            raise RuntimeError(
                f"CUDAExecutionProvider is required but runtime selected {provider}"
            )
        self.session = session
        self.provider = provider
        self.input_name = session.get_inputs()[0].name
        self.output_name = session.get_outputs()[0].name
        self.target_size = self._resolve_target_size(session.get_inputs()[0].shape)
        return session

    def _build_session_with_retry(self) -> None:
        attempts = self.settings.startup_load_retries + 1
        delay = self.settings.startup_load_retry_delay_seconds
        last_exc: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                self._build_session()
                return
            except Exception as exc:  # transient GPU/provider init failures
                last_exc = exc
                logger.warning(
                    "runtime.load.attempt_failed",
                    attempt=attempt,
                    max_attempts=attempts,
                    error=str(exc),
                )
                if attempt < attempts:
                    time.sleep(delay)
        assert last_exc is not None
        raise last_exc

    def _resolve_target_size(self, shape: list[Any]) -> int:
        for dim in shape[1:3]:
            if isinstance(dim, int):
                return dim
        raise ValueError(f"unable to determine target image size from model input shape: {shape}")

    def _validate_model_output_shape(self) -> None:
        if self.session is None:
            raise RuntimeError("runtime session is not initialized")
        outputs = self.session.get_outputs()
        if not outputs:
            raise ValueError("model must expose at least one output")
        output_shape = outputs[0].shape
        label_count = len(self.english_names) + self.RATING_TAG_COUNT
        output_count = self._resolve_output_count(output_shape)
        if output_count is None:
            logger.warning(
                "runtime.self_check.output_shape_dynamic",
                output_shape=str(output_shape),
                expected_labels=label_count,
            )
            return
        if output_count != label_count:
            raise ValueError(
                "model output label count mismatch: "
                f"output has {output_count} scores, tags csv expects {label_count}"
            )

    @staticmethod
    def _resolve_output_count(shape: list[Any]) -> int | None:
        if shape and isinstance(shape[-1], int):
            return shape[-1]
        return None

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

    # --------------------------------------------------------------- predict

    def predict(
        self,
        image_path: str | Path,
        threshold: float | None = None,
        use_chinese_name: bool | None = None,
        top_k: int | None = None,
    ) -> tuple[Path, list[dict[str, Any]], int]:
        session = self.session
        if not self.is_loaded or session is None:
            raise RuntimeError("runtime is not loaded")
        input_name = self.input_name
        output_name = self.output_name
        target_size = self.target_size

        resolved_path = self.resolve_image_path(image_path)
        image_ctx = set_image_path_context(resolved_path)
        started = time.perf_counter()
        acquired = False
        try:
            acquired = self._inference_slots.acquire(timeout=self.settings.inference_acquire_timeout_seconds)
            if not acquired:
                raise InferenceBusyError(
                    "inference capacity is exhausted; retry after the current request completes"
                )
            inflight_token = self._mark_inflight_start()
            try:
                with _TRACER.start_as_current_span("runtime.predict") as predict_span:
                    predict_span.set_attribute("runtime.provider", self.provider)
                    try:
                        with _TRACER.start_as_current_span("image.preprocess"):
                            with Image.open(resolved_path) as image:
                                processed = ImageUtils.preprocess_image(image, target_size)
                        with _TRACER.start_as_current_span("onnx.inference"):
                            scores = session.run([output_name], {input_name: processed})[0][0]
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
                        self._record_success()
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
                        self._record_failure(exc)
                        raise
            finally:
                self._mark_inflight_end(inflight_token)
        finally:
            if acquired:
                self._inference_slots.release()
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
        raw_tags: dict[str, float] = {}
        for name, score in zip(names, content_scores):
            if not name:
                continue
            value = float(score)
            if not math.isfinite(value):  # drop NaN/Inf so JSON stays valid
                continue
            raw_tags[name] = value

        for tag in self.settings.additional_tags:
            raw_tags[tag] = max(raw_tags.get(tag, 0.0), 1.0)

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

    # ----------------------------------------------------- liveness / healing

    def _mark_inflight_start(self) -> int:
        with self._state_lock:
            self._inflight_seq += 1
            token = self._inflight_seq
            self._inflight[token] = time.monotonic()
            return token

    def _mark_inflight_end(self, token: int) -> None:
        with self._state_lock:
            self._inflight.pop(token, None)

    def _oldest_inflight_start(self) -> float | None:
        # Caller must hold ``_state_lock``.
        if not self._inflight:
            return None
        return min(self._inflight.values())

    def _record_success(self) -> None:
        with self._state_lock:
            self._consecutive_failures = 0

    def _record_failure(self, exc: Exception) -> None:
        if classify_error(exc) != "infra":
            return
        with self._state_lock:
            self._consecutive_failures += 1
            failures = self._consecutive_failures
        logger.warning(
            "runtime.inference.infra_failure",
            error=str(exc),
            error_type=exc.__class__.__name__,
            consecutive_failures=failures,
        )
        if self.settings.session_auto_reload:
            self._maybe_reload_session()

    @property
    def consecutive_failures(self) -> int:
        with self._state_lock:
            return self._consecutive_failures

    def liveness(self) -> tuple[bool, str]:
        """Report dynamic health: loaded, not stuck, not in a failure storm."""
        if not self.is_loaded:
            return False, "runtime is not loaded"
        with self._state_lock:
            failures = self._consecutive_failures
            started = self._oldest_inflight_start()
        threshold = self.settings.liveness_failure_threshold
        if failures >= threshold:
            return False, f"{failures} consecutive inference failures (>= {threshold})"
        hard = self.settings.inference_hard_timeout_seconds
        if hard > 0 and started is not None:
            elapsed = time.monotonic() - started
            if elapsed >= hard:
                return False, f"inference stuck for {elapsed:.1f}s (>= {hard:.0f}s)"
        return True, "ok"

    def _maybe_reload_session(self) -> None:
        """Best-effort in-process session rebuild, rate-limited by a cooldown."""
        if not self._reload_lock.acquire(blocking=False):
            return  # another thread is already reloading
        try:
            now = time.monotonic()
            if (
                self._last_reload_monotonic is not None
                and now - self._last_reload_monotonic < self.settings.session_reload_cooldown_seconds
            ):
                return
            self._last_reload_monotonic = now
            logger.warning("runtime.session.reload.start", provider=self.provider)
            try:
                self._build_session()
                logger.info("runtime.session.reload.ok", provider=self.provider)
            except Exception as exc:
                # Reload failed: leave the failure counter climbing so liveness
                # trips and the orchestrator restarts the container.
                logger.error("runtime.session.reload.failed", error=str(exc))
        finally:
            self._reload_lock.release()

    def _start_watchdog(self) -> None:
        if self.settings.inference_hard_timeout_seconds <= 0 or self._watchdog_thread is not None:
            return
        self._watchdog_stop.clear()
        thread = threading.Thread(
            target=self._watchdog_loop,
            name="inference-watchdog",
            daemon=True,
        )
        self._watchdog_thread = thread
        thread.start()

    def _watchdog_loop(self) -> None:
        hard = self.settings.inference_hard_timeout_seconds
        interval = min(5.0, hard / 4) if hard > 0 else 5.0
        while not self._watchdog_stop.wait(interval):
            if self._check_watchdog():
                return

    def _check_watchdog(self) -> bool:
        """Return True (and trigger the fatal handler) if an inference is hung."""
        hard = self.settings.inference_hard_timeout_seconds
        if hard <= 0:
            return False
        with self._state_lock:
            started = self._oldest_inflight_start()
        if started is None:
            return False
        elapsed = time.monotonic() - started
        if elapsed < hard:
            return False
        logger.critical(
            "runtime.watchdog.inference_stuck",
            elapsed_seconds=round(elapsed, 1),
            hard_timeout_seconds=hard,
        )
        # Loguru's async queue is not flushed by os._exit, so also write the
        # reason synchronously to stderr — this is the operator's only breadcrumb.
        try:
            sys.stderr.write(
                f"FATAL runtime.watchdog.inference_stuck elapsed={elapsed:.1f}s "
                f"hard_timeout={hard:.0f}s; exiting for supervisor restart\n"
            )
            sys.stderr.flush()
            logger.complete()
        except Exception:  # pragma: no cover - never block the exit path
            pass
        # A native ONNX call cannot be interrupted from Python; exit the process
        # so the supervisor (docker restart / k8s) brings up a fresh one.
        self._fatal_handler(1)
        return True

    # ------------------------------------------------------------- gpu / nvml

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
        self._watchdog_stop.set()
        thread = self._watchdog_thread
        if thread is not None:
            thread.join(timeout=2.0)
            self._watchdog_thread = None
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
