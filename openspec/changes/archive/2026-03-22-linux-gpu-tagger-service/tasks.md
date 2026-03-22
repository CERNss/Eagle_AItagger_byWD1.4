## 1. Service Runtime

- [x] 1.1 Create a Linux-oriented service package with runtime settings, request/response schemas, and a reusable ONNX tagging runtime.
- [x] 1.2 Refactor image preprocessing and tag post-processing so the service can load the model once and serve single-image and batch predictions.

## 2. HTTP API

- [x] 2.1 Add a FastAPI application with startup initialization plus `/healthz` and `/readyz` endpoints.
- [x] 2.2 Implement `/tag` and `/tag/batch` endpoints that validate image paths and return per-request inference results without mutating Eagle metadata.
- [x] 2.3 Remove hardcoded `max_length` cap from `BatchTagRequest` schema so batch size is governed exclusively by the runtime `BATCH_LIMIT` setting, surfacing violations as HTTP 400.

## 3. Linux Packaging

- [x] 3.1 Replace the current runtime dependency setup with Linux service dependency manifests and normalize package file encoding for Linux tooling.
- [x] 3.2 Add Docker-focused deployment assets including a Dockerfile, Compose example, and runtime environment configuration.
- [x] 3.3 Remove unused packages (`packaging`, `psutil`) from `requirements.txt` and move script-only dependency (`requests`) to `requirements-dev.txt`.

## 4. Repository Cleanup And Verification

- [x] 4.1 Remove or replace Windows-specific launch and interaction paths so the documented runtime flow is Linux-only.
- [x] 4.2 Rewrite repository documentation for Linux + NVIDIA + Docker service usage and run a verification pass for imports, entrypoints, and OpenSpec task completion.

## 5. Operational Hardening

- [x] 5.1 Switch Compose healthcheck from `/healthz` to `/readyz` so container health only passes after the model has loaded.
- [x] 5.2 Add `restart: unless-stopped` to the Compose service so the container recovers automatically from unexpected exits.
- [x] 5.3 Change Dockerfile `CMD` from a direct `uvicorn` invocation to `python3 main.py` so `HOST` and `PORT` environment variables are honoured at runtime.
