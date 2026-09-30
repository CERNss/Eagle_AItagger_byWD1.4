# Eagle AI Tagger Service

Long-lived FastAPI microservice that tags images for Eagle-style libraries with a WD14 ONNX model (SwinV2-v3). The model is loaded once per process, inference runs on ONNX Runtime with CUDA-first provider selection, and images are referenced by shared filesystem paths over HTTP. Designed to run in Docker on Linux + NVIDIA GPUs; a Windows batch workflow existed in old versions and has been removed on purpose.

Highlights:

- **Path-based API** — `POST /tag` and `POST /tag/batch` take filesystem paths (typically a directory mounted into both this service and the caller); the service never mutates Eagle `metadata.json`.
- **Chinese/English tags** — outputs the `right_tag_cn` column from the tag CSV when available; the first 4 CSV rows (rating tags) are skipped.
- **Self-healing** — input errors map to 4xx; infrastructure failures trigger an in-process session reload, count toward liveness, and a watchdog force-exits the process on hung inference so the orchestrator restarts it.
- **Layered config** — environment variables > `config.yaml` > built-in defaults, with `${VAR}` secret interpolation from `.env`.
- **Observability** — structured JSON logs on stdout by default; optional OpenTelemetry tracing and metrics.

## Requirements

- Linux host with an NVIDIA GPU and a working `nvidia-smi`
- Docker Engine + NVIDIA Container Toolkit
- WD14 ONNX model file, e.g. `swinv2-v3.onnx` (from [SmilingWolf/wd-swinv2-tagger-v3](https://huggingface.co/SmilingWolf/wd-swinv2-tagger-v3))
- Tag CSV with a `name` column: `csv/Tags-cn_2024_ver-1.0.csv`

## Quick Start (Docker Compose)

1. Verify the host GPU runtime first:

   ```bash
   nvidia-smi
   docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
   ```

2. Scaffold local config from the tracked templates (the real files are git-ignored, and Compose refuses to start without `config.yaml` because the bind mount has `create_host_path: false`):

   ```bash
   cp config.example.yaml config.yaml   # canonical, non-secret configuration
   cp .env.example .env                 # secrets / machine-specific overrides
   ```

3. Put assets in place — model at `./model/swinv2-v3.onnx`, tag CSV at `./csv/Tags-cn_2024_ver-1.0.csv`. `./init.sh` automates this: it downloads the model from Hugging Face, scaffolds both config files, creates the shared image directory, and (optionally) pulls the image from a private registry. See `./init.sh --help`.

4. Start the service:

   ```bash
   IMAGE_ROOT=/path/to/shared-images docker compose up --build -d
   ```

   `IMAGE_ROOT` defaults to `/srv/shared-images` and is mounted into the container read-only as `/data/images`. The Compose template sets `REQUIRE_CUDA=true`, so a miswired GPU runtime fails startup instead of silently falling back to CPU.

5. Check it:

   ```bash
   curl http://127.0.0.1:8000/healthz
   python3 scripts/smoke_test.py --image-path /absolute/path/to/image.png
   ```

`./build.sh` builds the `linux/amd64` image locally and pushes it to a registry (override with `IMAGE_NAME`, `IMAGE_TAG`, `REGISTRY`).

## Configuration

Every setting is read through three layers, highest precedence first:

1. **Process environment variables** — including anything in `.env` (auto-loaded) and anything injected by docker-compose.
2. **`config.yaml`** — the canonical, non-secret configuration. Its keys are the variable names below in lowercase. String values may reference secrets via `${VAR}` or `${VAR:-default}`, resolved from the environment: passwords/tokens live in `.env`, `config.yaml` only names them. The file location is `CONFIG_PATH` (default `./config.yaml`); the env file location is `ENV_FILE` (default `./.env`).
3. **Built-in defaults** in `service/settings.py`.

If neither file exists the loader degrades to pure environment variables, so existing deployments keep working unchanged.

### Server, model, and tagging

| Variable | Default | Purpose |
|---|---:|---|
| `HOST` | `0.0.0.0` | Bind address for local `python main.py` runs |
| `PORT` | `8000` | HTTP port |
| `MODEL_PATH` | `model/swinv2-v3.onnx` | ONNX model path |
| `TAGS_PATH` | `csv/Tags-cn_2024_ver-1.0.csv` | Tag CSV path (needs a `name` column; `right_tag_cn` optional) |
| `IMAGE_ROOT` | unset | Optional root directory every image path must resolve inside; relative paths resolve under it |
| `DEFAULT_THRESHOLD` | `0.5` | Default score threshold |
| `USE_CHINESE_NAME` | `true` | Use `right_tag_cn` when available |
| `DEFAULT_TOP_K` | `50` | Default max returned tag count |
| `REPLACE_UNDERSCORE` | `true` | Replace `_` with spaces in returned tags |
| `UNDERSCORE_EXCLUDES` | empty | Comma-separated tags that keep underscores |
| `ESCAPE_TAGS` | `false` | Escape `\`, `(`, `)` in returned tags |
| `ADDITIONAL_TAGS` | empty | Comma-separated tags always appended with score `1.0` |
| `EXCLUDE_TAGS` | empty | Comma-separated tags always filtered out |
| `SORT_ALPHABETICALLY` | `false` | Sort alphabetically instead of by score desc |
| `STARTUP_SELF_CHECK` | `true` | Validate model output shape against the tag CSV during startup |
| `REQUIRE_CUDA` | `false` | Fail startup if CUDAExecutionProvider is unavailable |

### Capacity and resilience

| Variable | Default | Purpose |
|---|---:|---|
| `BATCH_LIMIT` | `64` | Maximum paths per `/tag/batch` request |
| `MAX_CONCURRENT_INFERENCE` | `1` | Maximum simultaneous inference calls per process |
| `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` | `30.0` | Seconds a request waits for an inference slot before returning 503 |
| `LIVENESS_FAILURE_THRESHOLD` | `5` | Consecutive infrastructure failures before `/livez` reports 503 |
| `INFERENCE_HARD_TIMEOUT_SECONDS` | `120.0` | Watchdog ceiling; a hung inference longer than this force-exits the process for an orchestrator restart (`0` disables) |
| `SESSION_AUTO_RELOAD` | `true` | Try to rebuild the ONNX session in-process after a fatal inference error |
| `SESSION_RELOAD_COOLDOWN_SECONDS` | `30.0` | Minimum gap between in-process session reloads |
| `STARTUP_LOAD_RETRIES` | `2` | Retry transient ONNX session creation at startup |
| `STARTUP_LOAD_RETRY_DELAY_SECONDS` | `3.0` | Delay between startup load retries |
| `MAX_IMAGE_PIXELS` | `89478485` | Pillow decode cap (~89.5 MP); oversized images are refused with a 400 instead of risking an OOM. `0` disables the cap. Decode only — the original file is never touched |
| `TIMEOUT_KEEP_ALIVE` | `5` | Uvicorn keep-alive seconds for idle clients |

### Observability

Structured JSON logs go to stdout via Loguru (filesystem paths are SHA-256 hashed by default), so Fluent Bit or any log collector can scrape container logs without extra mounts. OpenTelemetry tracing/metrics are off by default and enabled per deployment:

| Variable | Default | Purpose |
|---|---:|---|
| `LOG_LEVEL` | `INFO` | Loguru log level |
| `LOG_FORMAT` | `json` | `json` (structured) or `text` |
| `LOG_INCLUDE_TRACE` | `true` | Include OTEL trace/span ids in logs when available |
| `LOG_HASH_IMAGE_PATHS` | `true` | Hash filesystem paths instead of logging raw values |
| `SERVICE_NAME` | `eagle-ai-tagger` | OTEL resource `service.name` |
| `SERVICE_VERSION` | `dev` | OTEL resource `service.version` |
| `DEPLOYMENT_ENVIRONMENT` | `development` | OTEL `deployment.environment` |
| `OTEL_ENABLED` | `false` | Enable OTEL tracing (requires `OTEL_EXPORTER_ENDPOINT`) |
| `OTEL_EXPORTER_ENDPOINT` | unset | OTLP/HTTP endpoint, e.g. `http://otel-collector:4318` |
| `OTEL_EXPORTER_HEADERS` | unset | Comma-separated `key=value` headers for OTLP |
| `OTEL_TRACE_SAMPLE_RATIO` | `0.1` | Trace sampling ratio (0–1) |
| `OTEL_METRICS_ENABLED` | `false` | Enable OTEL metrics |
| `OTEL_METRICS_EXPORTER_ENDPOINT` | uses `OTEL_EXPORTER_ENDPOINT` | Override metrics endpoint |
| `OTEL_METRIC_EXPORT_INTERVAL` | `60` | Metric push interval in seconds |

Compose snippet for a collector named `otel-collector`:

```yaml
environment:
  OTEL_ENABLED: "true"
  OTEL_METRICS_ENABLED: "true"
  OTEL_EXPORTER_ENDPOINT: http://otel-collector:4318
  OTEL_TRACE_SAMPLE_RATIO: "0.2"
```

## API

### `GET /healthz`

Process health: model/tag paths, selected provider, and observability flags (logging mode, tracing/metrics booleans) so you can confirm exporter state without shelling into the container.

```bash
curl http://127.0.0.1:8000/healthz
```

### `GET /readyz`

Returns `200` only after the model runtime loaded successfully (one-time load gate), mirroring the observability flags.

```bash
curl http://127.0.0.1:8000/readyz
```

### `GET /livez`

Dynamic liveness: returns `200` while the runtime is loaded, not in a failure storm, and not stuck on a hung inference; otherwise `503` with a reason and the consecutive-failure count. Unlike `/readyz`, `/livez` reflects the live state of the GPU session, so a "running but broken" process actually fails the check. The Docker healthcheck targets this endpoint so `restart: unless-stopped` can recover a zombie or hung container.

```bash
curl http://127.0.0.1:8000/livez
```

### `POST /tag`

Tags a single image path. `threshold`, `use_chinese_name`, and `top_k` are per-request overrides of the defaults.

```bash
curl -X POST http://127.0.0.1:8000/tag \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"/data/images/example.png","threshold":0.5,"use_chinese_name":true,"top_k":50}'
```

Response:

```json
{
  "provider": "CUDAExecutionProvider",
  "image_path": "/data/images/example.png",
  "tags": [
    {"name": "1girl", "score": 0.987654},
    {"name": "少女", "score": 0.912345}
  ],
  "elapsed_ms": 42
}
```

### `POST /tag/batch`

Tags up to `BATCH_LIMIT` paths in one request. The call returns `200` with per-item results; invalid items come back as item-level `success: false` entries instead of aborting the batch.

```bash
curl -X POST http://127.0.0.1:8000/tag/batch \
  -H 'Content-Type: application/json' \
  -d '{"image_paths":["/data/images/a.png","/data/images/b.png"],"threshold":0.5}'
```

Response:

```json
{
  "provider": "CUDAExecutionProvider",
  "results": [
    {
      "image_path": "/data/images/a.png",
      "success": true,
      "tags": [{"name": "1girl", "score": 0.987654}],
      "elapsed_ms": 41,
      "error": null,
      "error_type": null
    },
    {
      "image_path": "/data/images/b.png",
      "success": false,
      "tags": [],
      "elapsed_ms": null,
      "error": "image not found: /data/images/b.png",
      "error_type": "FileNotFoundError"
    }
  ]
}
```

### Error semantics

| Condition | HTTP | Counts against liveness? |
|---|---:|---|
| Missing image / outside `IMAGE_ROOT` / corrupt or oversized image / bad parameters | 404 / 403 / 400 | No |
| Inference slots saturated past `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` | 503 (retryable) | No |
| Infrastructure failure (e.g. dead GPU session) | 503 (retryable) | Yes — triggers session reload and the failure counter |
| Batch over `BATCH_LIMIT` | 400 | No |

## Repository Layout

```text
.
├─ main.py                     # Local Uvicorn runner (single worker)
├─ service/
│  ├─ app.py                   # FastAPI app, endpoints, HTTP error mapping
│  ├─ runtime.py               # ONNX session lifecycle, inference, self-healing, watchdog
│  ├─ image_utils.py           # WD14 preprocessing (decode, downscale, white-pad)
│  ├─ schemas.py               # Pydantic request/response models
│  ├─ settings.py              # Typed settings + validation
│  ├─ config.py                # Layered config loader (env > yaml > defaults)
│  ├─ logging.py               # Loguru setup, request/image-path log context
│  └─ observability.py         # OTEL tracing/metrics setup, metric recorder
├─ scripts/smoke_test.py       # /healthz + optional /tag smoke test
├─ tests/                      # pytest suite (config, settings, resilience, stability…)
├─ init.sh                     # Bootstrap: model download, config scaffold, image pull
├─ build.sh                    # linux/amd64 build + push helper
├─ Dockerfile                  # CUDA 12.9 runtime base, bakes a default config.yaml
├─ docker-compose.yaml         # GPU service wiring, healthcheck, restart policy
├─ config.example.yaml         # Template for config.yaml
├─ .env.example                # Template for .env
├─ csv/Tags-cn_2024_ver-1.0.csv# Tag dictionary (en + zh)
└─ model/                      # Drop swinv2-v3.onnx here
```

## Operational Notes

- **Single process, single session.** The model is loaded once; do not start Uvicorn with multiple workers unless you want multiple model copies in memory.
- **Capacity.** `MAX_CONCURRENT_INFERENCE` (default 1) serializes GPU work to stay stable under bursts; requests queue up to `INFERENCE_ACQUIRE_TIMEOUT_SECONDS`, then get a 503 to retry.
- **Failure handling.** Input problems return 4xx and never affect health. Infrastructure failures return retryable 503, increment the liveness failure counter, and trigger a best-effort in-process session reload (rate-limited by the cooldown). After `LIVENESS_FAILURE_THRESHOLD` consecutive infra failures, `/livez` reports 503 and the orchestrator restarts the container.
- **Hang handling.** A native ONNX call cannot be interrupted from Python, so a watchdog force-exits the process when one inference exceeds `INFERENCE_HARD_TIMEOUT_SECONDS`; `restart: unless-stopped` brings up a fresh process. Set the ceiling comfortably above your slowest real inference.
- **Large images.** Preprocessing downscales in memory before padding and refuses to decode images above `MAX_IMAGE_PIXELS` with a 400. Combined with the Compose memory limit (4 GB), a pathological image OOM-kills only the container, which then restarts. The original file on disk is only ever read.
- **Path safety.** If `IMAGE_ROOT` is set, every requested path must resolve inside it; logs hash image paths by default.
- **GPU sanity.** If `/healthz` reports `CPUExecutionProvider`, the container GPU runtime is not wired correctly. Set `REQUIRE_CUDA=true` when CPU fallback is unacceptable (the Compose template already does).

## Recommended Integration Pattern

Mount the same image directory into both your business service and this tagger, then send shared filesystem paths over HTTP. The service returns inference results only — persisting tags into Eagle `metadata.json` is the caller's job.

```python
import requests

response = requests.post(
    "http://tagger:8000/tag",
    json={
        "image_path": "/data/images/example.png",
        "threshold": 0.5,
        "use_chinese_name": True,
        "top_k": 50,
    },
    timeout=30,
)
response.raise_for_status()
print(response.json())
```

## Development

```bash
python3 -m pip install -r requirements-dev.txt   # runtime + httpx/pytest/requests
python3 -m pytest -q                            # unit/integration suite (no GPU needed)
python3 main.py                                 # run locally (CPU fallback works when REQUIRE_CUDA=false)
python3 scripts/smoke_test.py --image-path /absolute/path/to/image.png
```

CI (`.github/workflows/ci.yml`) runs the suite plus `compileall`, `bash -n` on the shell scripts, `docker compose config`, and OpenSpec spec validation on pushes/PRs to `main`.

## Release

Container publishing is tag-driven and gated to `main`: pushing a Git tag starting with `v` builds the `linux/amd64` image and pushes it to Docker Hub **only when the tagged commit is contained in `origin/main`**. Required repository secrets: `DOCKERHUB_USERNAME`, `DOCKERHUB_TOKEN`. Default image: `cernss/eagle-ai-tagger` (override with the `DOCKERHUB_REPOSITORY` repository variable).

```bash
git switch main
git pull --ff-only origin main
git tag v0.1.0
git push origin v0.1.0     # -> cernss/eagle-ai-tagger:v0.1.0
```

Manual publishing is available via the workflow-dispatch form on the `CI/CD` workflow, restricted to the `main` branch.
