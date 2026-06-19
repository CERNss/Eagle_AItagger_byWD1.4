# Eagle AI Tagger Service

Linux-only WD14 image tagging microservice for Eagle-style libraries. The service loads a WD14 ONNX model once, serves inference over HTTP, and is designed to run in Docker with NVIDIA GPU access.

## What This Repository Is Now

- Long-lived FastAPI service
- ONNX Runtime GPU inference with CUDA-first provider selection
- Filesystem-path based `/tag` and `/tag/batch` APIs
- Docker Compose deployment for Linux + NVIDIA

The previous Windows batch workflow has been removed on purpose. This repository now treats Linux service deployment as the primary and only supported runtime shape.

## Requirements

- Linux host
- NVIDIA GPU with working `nvidia-smi`
- Docker Engine with NVIDIA Container Toolkit installed
- WD14 ONNX model file, for example `swinv2-v3.onnx`
- Tag CSV file: `Tags-cn_2024_ver-1.0.csv`

## Repository Layout

```text
.
├─ main.py                 # Local Uvicorn runner
├─ service/
│  ├─ app.py               # FastAPI app and endpoints
│  ├─ runtime.py           # ONNX runtime and tag post-processing
│  ├─ schemas.py           # Request/response models
│  ├─ settings.py          # Environment-driven configuration
│  └─ image_utils.py       # WD14 image preprocessing
├─ scripts/
│  └─ smoke_test.py        # Basic service smoke test
├─ Dockerfile
├─ docker-compose.yaml
├─ requirements.txt
└─ csv/Tags-cn_2024_ver-1.0.csv
```

## Configuration

The service is configured entirely through environment variables.

| Variable | Default | Purpose |
|---|---:|---|
| `HOST` | `0.0.0.0` | Bind address for local `python main.py` runs |
| `PORT` | `8000` | HTTP port |
| `MODEL_PATH` | `model/swinv2-v3.onnx` | ONNX model path |
| `TAGS_PATH` | `csv/Tags-cn_2024_ver-1.0.csv` | Tag CSV path |
| `IMAGE_ROOT` | unset | Optional root directory restriction for image paths |
| `DEFAULT_THRESHOLD` | `0.5` | Default score threshold |
| `USE_CHINESE_NAME` | `true` | Use `right_tag_cn` when available |
| `DEFAULT_TOP_K` | `50` | Default max returned tag count |
| `BATCH_LIMIT` | `64` | Maximum paths per `/tag/batch` request |
| `MAX_CONCURRENT_INFERENCE` | `1` | Maximum simultaneous inference calls per service process |
| `INFERENCE_ACQUIRE_TIMEOUT_SECONDS` | `30.0` | Seconds a request waits for an inference slot before returning 503 |
| `STARTUP_SELF_CHECK` | `true` | Validate model output shape against the tag CSV during startup |
| `REQUIRE_CUDA` | `false` | Fail startup if CUDAExecutionProvider is unavailable |
| `REPLACE_UNDERSCORE` | `true` | Replace `_` with spaces in returned tags |
| `UNDERSCORE_EXCLUDES` | empty | Comma-separated tags that keep underscores |
| `ESCAPE_TAGS` | `false` | Escape `\`, `(`, `)` in returned tags |
| `ADDITIONAL_TAGS` | empty | Comma-separated tags always appended with score `1.0` |
| `EXCLUDE_TAGS` | empty | Comma-separated tags always filtered out |
| `SORT_ALPHABETICALLY` | `false` | Sort alphabetically instead of by score desc |

### Observability

The service writes structured JSON logs to stdout using Loguru. Fluent Bit or any other collector can scrape the container logs without additional file mounts. OpenTelemetry tracing and metrics are disabled by default but can be turned on per deployment via environment variables.

| Variable | Default | Purpose |
|---|---:|---|
| `LOG_LEVEL` | `INFO` | Loguru log level |
| `LOG_FORMAT` | `json` | `json` (structured) or `text` |
| `LOG_INCLUDE_TRACE` | `true` | Include OTEL trace/span ids in logs when available |
| `LOG_HASH_IMAGE_PATHS` | `true` | Hash filesystem paths instead of logging raw values |
| `SERVICE_NAME` | `eagle-ai-tagger` | OTEL resource `service.name` |
| `SERVICE_VERSION` | `dev` | OTEL resource `service.version` |
| `DEPLOYMENT_ENVIRONMENT` | `development` | OTEL `deployment.environment` |
| `OTEL_ENABLED` | `false` | Enable OTEL tracing |
| `OTEL_EXPORTER_ENDPOINT` | unset | OTLP/HTTP endpoint (e.g. `http://otel-collector:4318`) |
| `OTEL_EXPORTER_HEADERS` | unset | Comma-separated `key=value` headers for OTLP |
| `OTEL_TRACE_SAMPLE_RATIO` | `0.1` | Trace sampling ratio (0–1) |
| `OTEL_METRICS_ENABLED` | `false` | Enable OTEL metrics |
| `OTEL_METRICS_EXPORTER_ENDPOINT` | uses `OTEL_EXPORTER_ENDPOINT` | Override metrics endpoint |
| `OTEL_METRIC_EXPORT_INTERVAL` | `60` | Metric push interval in seconds |

Example Compose snippet wiring tracing/metrics to a collector named `otel-collector`:

```yaml
environment:
  LOG_LEVEL: INFO
  OTEL_ENABLED: "true"
  OTEL_METRICS_ENABLED: "true"
  OTEL_EXPORTER_ENDPOINT: http://otel-collector:4318
  OTEL_TRACE_SAMPLE_RATIO: "0.2"
```

When tracing/metrics remain disabled (the defaults), only structured stdout logs are produced, so Fluent Bit can still ship them without additional configuration.

## Local Run

1. Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

2. Put your model file in `./model`, for example:

```bash
mkdir -p model
cp /path/to/swinv2-v3.onnx model/swinv2-v3.onnx
```

3. Start the service:

```bash
python3 main.py
```

4. Optional smoke test:

```bash
python3 scripts/smoke_test.py
python3 scripts/smoke_test.py --image-path /absolute/path/to/image.png
```

## Docker Run

The provided Compose file expects:

- model file at `./model/swinv2-v3.onnx`
- tag CSV at `./csv/Tags-cn_2024_ver-1.0.csv`
- images shared from `${IMAGE_ROOT:-/srv/shared-images}` on the host

Compose mounts those directories into the container as:

- `/model/swinv2-v3.onnx`
- `/csv/Tags-cn_2024_ver-1.0.csv`
- `/data/images`

Use a custom host image directory by setting `IMAGE_ROOT` for Compose:

```bash
IMAGE_ROOT=/path/to/shared-images docker compose up --build
```

Start the service:

```bash
docker compose up --build
```

The Compose template sets `REQUIRE_CUDA=true`, so GPU deployment fails startup instead of silently falling back to CPU. For CPU-only local experimentation, run `python3 main.py` directly or override the Compose environment intentionally.

Before doing that, verify the host first:

```bash
nvidia-smi
docker run --rm --gpus all nvidia/cuda:12.9.0-base-ubuntu22.04 nvidia-smi
```

## API

### `GET /healthz`

Returns process health, provider information, model/tag paths, and observability status (logging mode plus tracing/metrics booleans). The observability fields let you confirm whether OTEL exporters are active without shelling into the container.

Example:

```bash
curl http://127.0.0.1:8000/healthz
```

### `GET /readyz`

Returns `200` only after the model runtime has loaded successfully. The response mirrors the observability flags so readiness probes can assert tracing/metrics state as part of deployment checks.

Example:

```bash
curl http://127.0.0.1:8000/readyz
```

### `POST /tag`

Tags a single image path.

Request:

```json
{
  "image_path": "/data/images/example.png",
  "threshold": 0.5,
  "use_chinese_name": true,
  "top_k": 50
}
```

Example:

```bash
curl -X POST http://127.0.0.1:8000/tag \
  -H 'Content-Type: application/json' \
  -d '{"image_path":"/data/images/example.png","threshold":0.5,"use_chinese_name":true,"top_k":50}'
```

### `POST /tag/batch`

Tags multiple image paths in one request. Invalid items are returned as item-level errors instead of aborting the whole batch.

Request:

```json
{
  "image_paths": [
    "/data/images/example-1.png",
    "/data/images/example-2.png"
  ],
  "threshold": 0.5,
  "use_chinese_name": true,
  "top_k": 50
}
```

## Operational Notes

- The first version is intentionally single-process and single-session. Do not start Uvicorn with multiple workers unless you are ready for multiple model copies in memory.
- Inference is protected by `MAX_CONCURRENT_INFERENCE`; the default serializes GPU work to keep the service stable under bursts.
- If the inference limit is saturated longer than `INFERENCE_ACQUIRE_TIMEOUT_SECONDS`, `/tag` returns HTTP 503 so clients can retry.
- `STARTUP_SELF_CHECK=true` fails startup when the model output label count does not match the tag CSV.
- The service does not mutate Eagle `metadata.json` files. It only returns inference results.
- If `IMAGE_ROOT` is set, every requested image path must resolve inside that directory.
- If `/healthz` reports `CPUExecutionProvider`, your container GPU runtime is not wired correctly. Set `REQUIRE_CUDA=true` when CPU fallback is unacceptable.

## Recommended Integration Pattern

Mount the same image directory into both your business service and this tagger service, then send shared filesystem paths over HTTP.

Example Python client:

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
