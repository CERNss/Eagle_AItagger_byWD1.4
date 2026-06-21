from __future__ import annotations

import time

import pytest
from PIL import Image

from service.runtime import InferenceBusyError, TaggerRuntime, classify_error
from service.settings import Settings


def _runtime(monkeypatch, **env: object) -> TaggerRuntime:
    for key, value in env.items():
        monkeypatch.setenv(key, str(value))
    return TaggerRuntime(Settings.from_env())


def test_classify_error_buckets():
    assert classify_error(InferenceBusyError("x")) == "busy"
    assert classify_error(FileNotFoundError()) == "client"
    assert classify_error(PermissionError()) == "client"
    assert classify_error(ValueError()) == "client"
    assert classify_error(OSError("truncated")) == "client"
    # A decompression bomb is an input problem, not an infra failure: it must not
    # count toward liveness or trigger a session reload.
    assert classify_error(Image.DecompressionBombError("too many pixels")) == "client"
    assert classify_error(RuntimeError("cuda")) == "infra"
    assert classify_error(MemoryError()) == "infra"


def test_liveness_requires_loaded(monkeypatch):
    runtime = _runtime(monkeypatch)
    healthy, reason = runtime.liveness()
    assert healthy is False
    assert "not loaded" in reason


def test_liveness_trips_after_consecutive_infra_failures(monkeypatch):
    runtime = _runtime(monkeypatch, LIVENESS_FAILURE_THRESHOLD=3, SESSION_AUTO_RELOAD="false")
    runtime.is_loaded = True
    assert runtime.liveness()[0] is True
    for _ in range(3):
        runtime._record_failure(RuntimeError("cuda failure"))
    healthy, reason = runtime.liveness()
    assert healthy is False
    assert "consecutive" in reason


def test_client_errors_do_not_trip_liveness(monkeypatch):
    runtime = _runtime(monkeypatch, LIVENESS_FAILURE_THRESHOLD=2)
    runtime.is_loaded = True
    for _ in range(5):
        runtime._record_failure(FileNotFoundError("missing image"))
    assert runtime.consecutive_failures == 0
    assert runtime.liveness()[0] is True


def test_success_resets_failure_streak(monkeypatch):
    runtime = _runtime(monkeypatch, LIVENESS_FAILURE_THRESHOLD=3, SESSION_AUTO_RELOAD="false")
    runtime.is_loaded = True
    runtime._record_failure(RuntimeError("x"))
    runtime._record_failure(RuntimeError("x"))
    assert runtime.consecutive_failures == 2
    runtime._record_success()
    assert runtime.consecutive_failures == 0
    assert runtime.liveness()[0] is True


def test_liveness_detects_stuck_inference(monkeypatch):
    runtime = _runtime(monkeypatch, INFERENCE_HARD_TIMEOUT_SECONDS=0.05)
    runtime.is_loaded = True
    token = runtime._mark_inflight_start()
    time.sleep(0.06)
    healthy, reason = runtime.liveness()
    assert healthy is False
    assert "stuck" in reason
    runtime._mark_inflight_end(token)
    assert runtime.liveness()[0] is True


def test_liveness_tracks_oldest_inflight_under_concurrency(monkeypatch):
    # Regression: with MAX_CONCURRENT_INFERENCE > 1, a fast call finishing must
    # not clear a slow call's clock and hide its hang.
    runtime = _runtime(
        monkeypatch,
        INFERENCE_HARD_TIMEOUT_SECONDS=0.05,
        MAX_CONCURRENT_INFERENCE=2,
    )
    runtime.is_loaded = True
    fatal_calls: list[int] = []
    runtime._fatal_handler = fatal_calls.append  # don't os._exit the test runner
    stuck = runtime._mark_inflight_start()  # the call that will hang
    time.sleep(0.06)
    quick = runtime._mark_inflight_start()  # a fresh, fast call
    runtime._mark_inflight_end(quick)       # ...finishes immediately
    healthy, reason = runtime.liveness()
    assert healthy is False                 # the stuck call must still trip
    assert "stuck" in reason
    assert runtime._check_watchdog() is True
    assert fatal_calls == [1]
    runtime._mark_inflight_end(stuck)
    assert runtime.liveness()[0] is True


def test_watchdog_triggers_fatal_handler_on_hang(monkeypatch):
    runtime = _runtime(monkeypatch, INFERENCE_HARD_TIMEOUT_SECONDS=0.05)
    runtime.is_loaded = True
    calls: list[int] = []
    runtime._fatal_handler = calls.append
    runtime._mark_inflight_start()
    time.sleep(0.06)
    assert runtime._check_watchdog() is True
    assert calls == [1]


def test_watchdog_idle_does_not_fire(monkeypatch):
    runtime = _runtime(monkeypatch, INFERENCE_HARD_TIMEOUT_SECONDS=0.05)
    runtime.is_loaded = True
    calls: list[int] = []
    runtime._fatal_handler = calls.append
    assert runtime._check_watchdog() is False
    assert calls == []


def test_session_auto_reload_on_infra_failure(monkeypatch):
    runtime = _runtime(monkeypatch, SESSION_AUTO_RELOAD="true", SESSION_RELOAD_COOLDOWN_SECONDS="0")
    runtime.is_loaded = True
    reloads: list[bool] = []
    monkeypatch.setattr(runtime, "_build_session", lambda: reloads.append(True))
    runtime._record_failure(RuntimeError("cuda failure"))
    assert reloads == [True]


def test_session_reload_respects_cooldown(monkeypatch):
    runtime = _runtime(monkeypatch, SESSION_AUTO_RELOAD="true", SESSION_RELOAD_COOLDOWN_SECONDS="100")
    runtime.is_loaded = True
    reloads: list[bool] = []
    monkeypatch.setattr(runtime, "_build_session", lambda: reloads.append(True))
    runtime._record_failure(RuntimeError("x"))
    runtime._record_failure(RuntimeError("x"))
    assert len(reloads) == 1  # second failure within cooldown does not reload


def test_postprocess_drops_non_finite_scores(monkeypatch):
    runtime = _runtime(monkeypatch)
    runtime.english_names = ["a", "b", "c"]
    runtime.chinese_names = runtime.english_names[:]
    tags = runtime._postprocess_scores(
        scores=[0.0, 0.0, 0.0, 0.0, float("nan"), 0.9, float("inf")],
        threshold=0.5,
        use_chinese_name=False,
        top_k=10,
    )
    assert {tag["name"] for tag in tags} == {"b"}
