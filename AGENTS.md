# AGENTS.md - Eagle AI Tagger System Spec

## Project Overview

WD14 image tagging service for Eagle-style image libraries. The current repository is a Linux-first FastAPI microservice: it loads a WD14 ONNX model once, accepts image filesystem paths over HTTP, runs ONNX Runtime inference, and returns Chinese/English tag results as JSON.

The service no longer writes Eagle `metadata.json` files directly. Downstream services should call the HTTP API and decide how to persist returned tags.

**Target platform**: Linux + NVIDIA GPU + Docker
**Runtime**: Python 3.10+, FastAPI, ONNX Runtime GPU, CUDA 12.x

---

## Current Architecture

### Entry Points

| File | Role | Notes |
|---|---|---|
| `main.py` | Local service runner | Loads env settings, validates them, starts Uvicorn with one worker |
| `service/app.py` | FastAPI app | Defines lifecycle, middleware, health/readiness, tag APIs |
| `scripts/smoke_test.py` | Operational smoke test | Calls `/healthz` and optionally `/tag` |
| `init.sh` | Asset/bootstrap helper | Downloads model assets and prepares local image directory |
| `build.sh` | Image build/push helper | Builds linux/amd64 Docker image and pushes to registry |

### Core Modules (`service/`)

```text
main.py
  -> service/settings.py       # Environment-driven typed settings
  -> service/app.py            # FastAPI app, request context, HTTP error mapping
      -> service/runtime.py    # ONNX session lifecycle, image path checks, inference, tag post-processing
          -> service/image_utils.py  # PIL/OpenCV WD14 preprocessing
      -> service/schemas.py    # Pydantic request/response models
      -> service/logging.py    # Loguru setup, request id/image path context, stdlib log interception
      -> service/observability.py # OpenTelemetry tracing/metrics setup and metric recording
```

### Request Flow

```text
HTTP client
  -> POST /tag or /tag/batch
  -> FastAPI middleware adds x-request-id and logging context
  -> TaggerRuntime.resolve_image_path()
      -> expands relative paths against IMAGE_ROOT when configured
      -> rejects paths outside IMAGE_ROOT
  -> ImageUtils.preprocess_image()
      -> transparent background fill
      -> square padding
      -> resize to model input size
  -> ONNX Runtime InferenceSession.run()
  -> tag post-processing
      -> skip first 4 rating outputs
      -> threshold/filter/add/exclude tags
      -> choose `right_tag_cn` or `name`
      -> underscore replacement, optional escaping, sorting, top_k
  <- JSON response with provider, image_path, tags, elapsed_ms
```

### Runtime Model

- One Uvicorn worker by default. Do not increase Uvicorn workers casually because each worker loads its own model copy into memory.
- `TaggerRuntime` owns one ONNX Runtime `InferenceSession`.
- Inference calls are guarded by `MAX_CONCURRENT_INFERENCE`; exhausted capacity raises `InferenceBusyError` and maps to HTTP 503 for single-image requests.
- Startup self-check validates static model output label count against the tag CSV before accepting traffic.
- Provider selection is CUDA-first when `CUDAExecutionProvider` is available, otherwise CPU fallback.
- `/tag/batch` processes paths sequentially through the same runtime and returns item-level errors for invalid images.
- Optional NVML sampling records GPU memory usage when CUDA and `pynvml` are available.

---

## HTTP API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/healthz` | Process health, model/tag paths, provider, observability status |
| `GET` | `/readyz` | Returns 200 only after runtime load succeeds |
| `POST` | `/tag` | Tag one image path |
| `POST` | `/tag/batch` | Tag up to `BATCH_LIMIT` image paths |

The API accepts filesystem paths, not file uploads. If `IMAGE_ROOT` is set, requested paths must resolve inside that root.

---

## Configuration

Settings are read from environment variables in `service/settings.py`.

### Core Settings

| Variable | Default | Purpose |
|---|---|---|
| `HOST` | `0.0.0.0` | Bind address for local `python main.py` runs |
| `PORT` | `8000` | HTTP port |
| `MODEL_PATH` | `model/swinv2-v3.onnx` | ONNX model path |
| `TAGS_PATH` | `csv/Tags-cn_2024_ver-1.0.csv` | Tag CSV path |
| `IMAGE_ROOT` | unset | Optional root directory restriction for image paths |
| `DEFAULT_THRESHOLD` | `0.5` | Default score threshold |
| `USE_CHINESE_NAME` | `true` | Use `right_tag_cn` when present |
| `DEFAULT_TOP_K` | `50` | Default max returned tag count |
| `BATCH_LIMIT` | `64` | Maximum `/tag/batch` size |
| `MAX_CONCURRENT_INFERENCE` | `1` | Maximum simultaneous inference calls per process |
| `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` | `30.0` | Seconds to wait for an inference slot before returning 503 |
| `STARTUP_SELF_CHECK` | `true` | Validate model output shape against the tag CSV at startup |
| `REQUIRE_CUDA` | `false` | Fail startup if CUDAExecutionProvider is unavailable |
| `REPLACE_UNDERSCORE` | `true` | Replace `_` with spaces in returned tags |
| `UNDERSCORE_EXCLUDES` | empty | Comma-separated tags that keep underscores |
| `ESCAPE_TAGS` | `false` | Escape backslash and parentheses |
| `ADDITIONAL_TAGS` | empty | Comma-separated tags appended with score `1.0` |
| `EXCLUDE_TAGS` | empty | Comma-separated tags filtered out |
| `SORT_ALPHABETICALLY` | `false` | Sort alphabetically instead of score descending |

### Observability Settings

| Variable | Default | Purpose |
|---|---|---|
| `LOG_LEVEL` | `INFO` | Log level |
| `LOG_FORMAT` | `json` | `json` or `text` |
| `LOG_INCLUDE_TRACE` | `true` | Include trace/span ids in logs when available |
| `LOG_HASH_IMAGE_PATHS` | `true` | Hash image paths in logs |
| `SERVICE_NAME` | `eagle-ai-tagger` | OpenTelemetry service name |
| `SERVICE_VERSION` | `dev` | OpenTelemetry service version |
| `DEPLOYMENT_ENVIRONMENT` | `development` | OpenTelemetry deployment environment |
| `OTEL_ENABLED` | `false` | Enable tracing |
| `OTEL_EXPORTER_ENDPOINT` | unset | OTLP/HTTP trace endpoint |
| `OTEL_EXPORTER_HEADERS` | unset | Comma-separated `key=value` headers |
| `OTEL_TRACE_SAMPLE_RATIO` | `0.1` | Trace sampling ratio |
| `OTEL_METRICS_ENABLED` | `false` | Enable metrics |
| `OTEL_METRICS_EXPORTER_ENDPOINT` | trace endpoint | Optional metrics endpoint override |
| `OTEL_METRIC_EXPORT_INTERVAL` | `60` | Metric export interval in seconds |

---

## Dependencies

### Runtime

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn[standard]` | HTTP service |
| `onnxruntime-gpu` | ONNX model inference with CUDA support |
| `opencv-python-headless`, `pillow`, `numpy` | Image preprocessing |
| `pandas` | Tag CSV loading |
| `loguru` | Structured logging |
| `opentelemetry-*` | Optional tracing and metrics |
| `pynvml` | Optional NVIDIA GPU memory metrics |

### Development

- `requirements-dev.txt` includes runtime dependencies plus `pytest`, `httpx`, and `requests`.
- Current tests focus on settings validation in `tests/test_settings.py`.

---

## Docker And Assets

Default Docker Compose behavior:

- model: `./model/swinv2-v3.onnx` -> `/model/swinv2-v3.onnx`
- tags: `./csv/Tags-cn_2024_ver-1.0.csv` -> `/csv/Tags-cn_2024_ver-1.0.csv`
- images: `${IMAGE_ROOT:-/srv/shared-images}` -> `/data/images`

Inside the container, `MODEL_PATH`, `TAGS_PATH`, and `IMAGE_ROOT` are set to `/model/swinv2-v3.onnx`, `/csv/Tags-cn_2024_ver-1.0.csv`, and `/data/images`.

Docker Compose sets `REQUIRE_CUDA=true` and uses a longer readiness start period so GPU deployment fails fast on missing CUDA while tolerating slow model startup.

Use `init.sh` to download the model and prepare the image root. Use `build.sh` when publishing the image to the configured private registry.

GitHub Actions packaging is tag-driven and gated to `develop`. Branch pushes and PRs run checks only; pushing a tag matching `v*` builds and pushes the container image only when the tagged commit is contained in `origin/develop`.

---

## Testing And Verification

Preferred local checks:

```bash
.venv/bin/python -m pytest -q
python3 scripts/smoke_test.py
python3 scripts/smoke_test.py --image-path /absolute/path/to/image.png
```

Container/GPU checks:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
docker compose up --build
curl http://127.0.0.1:8000/readyz
```

---

## Repository Layout

```text
.
+-- main.py
+-- service/
|   +-- app.py
|   +-- runtime.py
|   +-- schemas.py
|   +-- settings.py
|   +-- image_utils.py
|   +-- logging.py
|   +-- observability.py
+-- scripts/
|   +-- smoke_test.py
+-- tests/
|   +-- conftest.py
|   +-- test_settings.py
+-- csv/
|   +-- Tags-cn_2024_ver-1.0.csv
+-- model/
|   +-- swinv2-v3.onnx  # user-provided/downloaded, not committed
+-- Dockerfile
+-- docker-compose.yaml
+-- init.sh
+-- build.sh
+-- requirements.txt
+-- requirements-dev.txt
+-- README.md
+-- AGENTS.md
+-- openspec/
```

---

## Notes For Future Agents

- Treat `service/` as the source of truth. The old Windows batch workflow is no longer present in the active tree.
- Keep the service single-process unless explicitly changing model lifecycle and GPU memory behavior.
- Do not add direct Eagle `metadata.json` mutation back into this service unless the API contract changes intentionally.
- Prefer tests around settings, path safety, tag post-processing, API error mapping, and observability status when changing behavior.
- Preserve path privacy in logs unless a deployment explicitly disables `LOG_HASH_IMAGE_PATHS`.
