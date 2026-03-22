## Why

The current project is organized around an interactive Windows batch workflow that reads `image_list.txt`, prompts on stdin, and writes results back as an offline script. That shape makes Linux deployment, Docker-based GPU scheduling, and service-to-service integration unnecessarily fragile.

We now want this repository to become a Linux-first GPU tagging microservice that can be called by other Docker services through HTTP. This change is needed now so the project can run as a long-lived NVIDIA-backed container instead of a desktop-oriented script.

## What Changes

- **BREAKING** Replace the script-style entry flow with a Linux-only HTTP service built on FastAPI.
- **BREAKING** Remove Windows-specific launch and interaction patterns, including `run.bat`, stdin prompts, and Windows-focused documentation.
- Extract the ONNX inference path into a reusable runtime that loads the model once at process startup and serves multiple requests.
- Add HTTP endpoints for health checks and image tagging using shared filesystem paths.
- Add Linux Docker assets for NVIDIA GPU execution, including a Dockerfile, Compose definition, and service-oriented dependency set.
- Update repository documentation and configuration around Linux deployment, mounted model assets, and service invocation.

## Capabilities

### New Capabilities
- `gpu-tagger-service`: Linux-only HTTP microservice for loading a WD14 ONNX model once and serving tag inference requests over FastAPI.
- `containerized-linux-deployment`: Docker-based deployment flow for running the tagger service with NVIDIA GPU access and mounted model/image volumes.

### Modified Capabilities
- None.

## Impact

- Affected code: top-level entrypoints, inference loading path, packaging, and runtime configuration.
- Affected APIs: introduces `/healthz`, `/readyz`, `/tag`, and `/tag/batch` HTTP endpoints.
- Affected dependencies: adds FastAPI/Uvicorn service dependencies and removes Windows-only or unused packages from the main install path.
- Affected systems: deployment moves to Linux + NVIDIA Container Toolkit + Docker Compose instead of Windows batch execution.
