from __future__ import annotations

from fastapi import status

from service.app import _http_exception_for_error
from service.runtime import InferenceBusyError


def test_http_exception_maps_inference_busy_to_503():
    exc = _http_exception_for_error(InferenceBusyError("busy"))

    assert exc.status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert exc.detail == "busy"


def test_http_exception_maps_path_and_validation_errors():
    assert _http_exception_for_error(FileNotFoundError("missing")).status_code == status.HTTP_404_NOT_FOUND
    assert _http_exception_for_error(PermissionError("denied")).status_code == status.HTTP_403_FORBIDDEN
    assert _http_exception_for_error(ValueError("bad")).status_code == status.HTTP_400_BAD_REQUEST
    assert _http_exception_for_error(OSError("bad path")).status_code == status.HTTP_400_BAD_REQUEST


def test_http_exception_maps_infra_errors_to_retryable_503():
    # Infrastructure/unknown failures (e.g. a dead ONNX session) are retryable
    # and also drive runtime liveness, so they map to 503 rather than 500.
    assert _http_exception_for_error(RuntimeError("boom")).status_code == status.HTTP_503_SERVICE_UNAVAILABLE
    assert _http_exception_for_error(MemoryError()).status_code == status.HTTP_503_SERVICE_UNAVAILABLE
