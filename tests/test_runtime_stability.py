from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from service.runtime import InferenceBusyError, TaggerRuntime
from service.settings import (
    LoggingSettings,
    MetricsSettings,
    ObservabilitySettings,
    Settings,
    TracingSettings,
)


class _FakeIO:
    def __init__(self, name: str = "io", shape: list[object] | None = None) -> None:
        self.name = name
        self.shape = shape or [1, 448, 448, 3]


class _FakeSession:
    def __init__(
        self,
        scores: list[float] | None = None,
        output_shape: list[object] | None = None,
        sleep_seconds: float = 0,
    ) -> None:
        self.scores = scores or [0.0, 0.0, 0.0, 0.0, 0.9]
        self.output_shape = output_shape or [1, len(self.scores)]
        self.sleep_seconds = sleep_seconds

    def get_inputs(self) -> list[_FakeIO]:
        return [_FakeIO("input", [1, 448, 448, 3])]

    def get_outputs(self) -> list[_FakeIO]:
        return [_FakeIO("output", self.output_shape)]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, _: list[str], __: dict[str, object]) -> list[list[list[float]]]:
        if self.sleep_seconds:
            time.sleep(self.sleep_seconds)
        return [[self.scores]]


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    observability = ObservabilitySettings(
        logging=LoggingSettings(
            level="INFO",
            log_format="json",
            include_trace_context=True,
            hash_image_paths=True,
        ),
        tracing=TracingSettings(
            enabled=False,
            exporter_endpoint=None,
            headers=(),
            sample_ratio=0.1,
        ),
        metrics=MetricsSettings(
            enabled=False,
            exporter_endpoint=None,
            headers=(),
            export_interval_seconds=60,
        ),
        service_name="test",
        service_version="test",
        deployment_environment="test",
    )
    base = Settings(
        host="127.0.0.1",
        port=8000,
        model_path=tmp_path / "model.onnx",
        tags_path=tmp_path / "tags.csv",
        image_root=None,
        default_threshold=0.5,
        default_use_chinese_name=True,
        default_top_k=50,
        batch_limit=2,
        max_concurrent_inference=1,
        inference_acquire_timeout_seconds=30.0,
        startup_self_check=True,
        require_cuda=False,
        replace_underscore=True,
        underscore_excludes=(),
        escape_tags=False,
        additional_tags=(),
        exclude_tags=(),
        sort_alphabetically=False,
        observability=observability,
    )
    return replace(base, **overrides)


def _write_tags(path: Path, content_names: list[str], chinese_names: list[str] | None = None) -> None:
    lines = ["name,right_tag_cn"]
    for idx in range(4):
        lines.append(f"rating_{idx},评级_{idx}")
    for idx, name in enumerate(content_names):
        chinese = chinese_names[idx] if chinese_names is not None else f"中文_{idx}"
        lines.append(f"{name},{chinese}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_resolve_image_path_rejects_paths_outside_image_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = tmp_path / "outside.png"
    outside.write_text("not an image", encoding="utf-8")
    runtime = TaggerRuntime(_settings(tmp_path, image_root=root))

    with pytest.raises(PermissionError):
        runtime.resolve_image_path("../outside.png")


def test_resolve_image_path_allows_relative_paths_inside_image_root(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    image = root / "image.png"
    image.write_text("not an image", encoding="utf-8")
    runtime = TaggerRuntime(_settings(tmp_path, image_root=root))

    assert runtime.resolve_image_path("image.png") == image.resolve()


def test_predict_batch_rejects_requests_over_limit(tmp_path):
    runtime = TaggerRuntime(_settings(tmp_path, batch_limit=1))

    with pytest.raises(ValueError, match="batch size 2 exceeds configured limit 1"):
        runtime.predict_batch(["a.png", "b.png"])


def test_postprocess_additional_tags_override_low_model_score(tmp_path):
    settings = _settings(
        tmp_path,
        additional_tags=("must_have",),
        default_threshold=0.5,
        replace_underscore=False,
    )
    runtime = TaggerRuntime(settings)
    runtime.english_names = ["must_have", "regular"]
    runtime.chinese_names = runtime.english_names[:]

    tags = runtime._postprocess_scores(
        scores=[0.0, 0.0, 0.0, 0.0, 0.1, 0.8],
        threshold=0.5,
        use_chinese_name=False,
        top_k=10,
    )

    assert {"name": "must_have", "score": 1.0} in tags
    assert {"name": "regular", "score": 0.8} in tags


def test_postprocess_rejects_label_count_mismatch(tmp_path):
    runtime = TaggerRuntime(_settings(tmp_path))
    runtime.english_names = ["one"]
    runtime.chinese_names = runtime.english_names[:]

    with pytest.raises(ValueError, match="model output does not match"):
        runtime._postprocess_scores(
            scores=[0.0, 0.0, 0.0, 0.0, 0.9, 0.8],
            threshold=0.5,
            use_chinese_name=False,
            top_k=10,
        )


def test_startup_self_check_rejects_output_label_count_mismatch(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    settings.model_path.write_bytes(b"fake")
    _write_tags(settings.tags_path, ["tag_a", "tag_b"])

    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(_: str, providers: list[str]) -> _FakeSession:
            assert providers == ["CPUExecutionProvider"]
            return _FakeSession(scores=[0.0] * 5, output_shape=[1, 5])

    monkeypatch.setattr("service.runtime.ort", FakeOrt)

    with pytest.raises(ValueError, match="model output label count mismatch"):
        TaggerRuntime(settings).load()


def test_require_cuda_rejects_cpu_provider(tmp_path, monkeypatch):
    settings = _settings(tmp_path, require_cuda=True)
    settings.model_path.write_bytes(b"fake")
    _write_tags(settings.tags_path, ["tag_a"])

    class FakeOrt:
        @staticmethod
        def get_available_providers() -> list[str]:
            return ["CPUExecutionProvider"]

        @staticmethod
        def InferenceSession(_: str, providers: list[str]) -> _FakeSession:
            assert providers == ["CPUExecutionProvider"]
            return _FakeSession()

    monkeypatch.setattr("service.runtime.ort", FakeOrt)

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider is required"):
        TaggerRuntime(settings).load()


def test_predict_raises_busy_when_inference_slot_timeout_expires(tmp_path, monkeypatch):
    image = tmp_path / "image.png"
    image.write_bytes(b"fake image")
    settings = _settings(tmp_path, inference_acquire_timeout_seconds=0.01)
    runtime = TaggerRuntime(settings)
    runtime.session = _FakeSession(sleep_seconds=0.1)
    runtime.provider = "CPUExecutionProvider"
    runtime.input_name = "input"
    runtime.output_name = "output"
    runtime.target_size = 448
    runtime.english_names = ["tag"]
    runtime.chinese_names = ["标签"]
    runtime.has_chinese_names = True
    runtime.is_loaded = True

    monkeypatch.setattr("service.runtime.Image.open", lambda _: object())
    monkeypatch.setattr("service.runtime.ImageUtils.preprocess_image", lambda *_: "processed")

    assert runtime._inference_slots.acquire(blocking=False)
    try:
        with pytest.raises(InferenceBusyError):
            runtime.predict(image)
    finally:
        runtime._inference_slots.release()
