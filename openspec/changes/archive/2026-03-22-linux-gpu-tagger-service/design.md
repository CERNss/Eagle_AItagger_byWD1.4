## Context

The repository currently centers on a batch-oriented Python script that reads file paths from `image_list.txt`, starts multiple worker processes, performs ONNX inference, and writes tags back into Eagle `metadata.json` files. That flow assumes an interactive desktop environment and contains stdin prompts, Windows launch tooling, and documentation tailored to manual local execution.

The new target shape is a Linux-only microservice that runs inside Docker with NVIDIA GPU access and exposes HTTP endpoints for inference. The existing ONNX preprocessing and tag post-processing code is still useful, but the orchestration model needs to change from short-lived batch execution to a long-lived process that loads the model once and serves requests repeatedly.

The service will be invoked by other containers that share mounted model files and image directories. The first version prioritizes operational simplicity over maximum throughput.

## Goals / Non-Goals

**Goals:**
- Provide a Linux-first FastAPI service with explicit health and readiness endpoints.
- Load the WD14 ONNX model once at startup and reuse the session for all requests.
- Accept image paths from a mounted shared directory and return filtered tag results over HTTP.
- Ship Docker assets for NVIDIA GPU execution with mounted models and images.
- Remove Windows-specific launch paths, interactive prompts, and Windows-focused packaging from the primary runtime flow.

**Non-Goals:**
- Preserving Windows compatibility or batch launcher behavior.
- Supporting per-request process pools or multiple Uvicorn workers in the first release.
- Adding async job queues, distributed scheduling, or multi-GPU orchestration.
- Automatically writing Eagle metadata files inside the HTTP inference path.

## Decisions

### 1. Replace the batch entrypoint with a long-lived FastAPI service

The new primary entrypoint will be an ASGI app served by Uvicorn. This matches the deployment target of a Dockerized microservice and cleanly separates transport concerns from inference logic.

Alternatives considered:
- Keep `main.py` as the primary entry and wrap it from a subprocess-based API layer. Rejected because it preserves startup overhead and desktop workflow assumptions.
- Expose a CLI-only interface and let other services shell out. Rejected because it complicates timeouts, health checks, and container orchestration.

### 2. Extract inference into a reusable runtime object

We will refactor the existing tagger implementation into a runtime class that owns:
- ONNX Runtime session creation
- provider selection with CUDA-first fallback
- tag CSV loading
- image preprocessing and tag post-processing

The runtime will be initialized once during app startup and stored in process memory.

Alternatives considered:
- Rebuild the session on every request. Rejected due to latency and GPU memory churn.
- Reuse the current multiprocessing manager. Rejected for the first release because each worker loads its own model copy and complicates service stability.

### 3. Keep the request contract filesystem-based

The first version of `/tag` and `/tag/batch` will accept image paths rather than uploaded file blobs. This aligns with the current Eagle workflow and avoids transferring large images between services when containers can share a mounted directory.

Alternatives considered:
- Multipart file upload support in the first version. Deferred to keep the initial API surface small and to reduce memory pressure.

### 4. Use Linux-focused packaging and dependency sets

The project will gain a service-specific dependency manifest and Linux Docker assets. Windows-only or unused dependencies will be removed from the main service install path. The legacy `requirements.txt` will be normalized to UTF-8 so Linux tooling can consume it reliably.

Alternatives considered:
- Preserve one cross-platform dependency file with optional extras. Rejected because the user explicitly wants a Linux-only design and the current package list includes Windows-specific baggage.

### 5. Keep metadata mutation out of the service API

The HTTP service will only return inference results. It will not directly update Eagle library JSON files. This keeps the service stateless with respect to client workflows and makes it easier to test, cache, and integrate.

Alternatives considered:
- Add an endpoint that writes `metadata.json` immediately. Rejected because it couples inference with storage side effects and would require more path validation and permission handling.

## Risks / Trade-offs

- [GPU provider mismatch] ONNX Runtime may fall back to CPU if container CUDA libraries do not match the installed wheel. → Mitigation: expose provider state in `/healthz` and document the expected NVIDIA runtime setup.
- [Single-process throughput ceiling] One process keeps operational complexity low but may limit concurrency under heavy load. → Mitigation: benchmark the single-process service first and only add batching or queueing if needed.
- [Filesystem path trust] Accepting image paths requires careful path existence checks and shared mount conventions between services. → Mitigation: validate path existence, fail fast with clear 4xx errors, and document the mount contract.
- [Migration confusion] Existing users may still expect `main.py`, `run.bat`, and offline JSON writing behavior. → Mitigation: update README and entrypoints to make the Linux service the default documented workflow.

## Migration Plan

1. Introduce the service runtime, API schemas, and ASGI app.
2. Add Dockerfile, Compose file, and service dependency manifests.
3. Remove interactive blocking behavior and Windows-only launch artifacts from the primary path.
4. Rewrite README around Linux deployment, mounted models, and HTTP invocation.
5. Verify local startup, health responses, and request handling on CPU-compatible environments; document GPU verification steps for Linux hosts.

Rollback strategy: revert to the previous script-oriented entrypoint from version control if service migration blocks deployment.

## Open Questions

- Should the service reject paths outside a configured image root, or is existence validation sufficient for the first release?
- Do we want to keep any offline CLI utility for library mutation as a separate maintenance tool, or fully retire it in this branch?
