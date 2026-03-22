## 1. Dependencies & Configuration

- [x] 1.1 Add `loguru` and required OpenTelemetry packages to `requirements.txt` and `requirements-dev.txt`.
- [x] 1.2 Extend `service/settings.py` with observability settings (log level/format, OTEL endpoints, sample ratios, metric intervals, feature flags) plus validation logic.
- [x] 1.3 Provide migration notes in README/Docker/compose docs describing new env vars.

## 2. Structured Logging Implementation

- [x] 2.1 Create `service/logging.py` (or similar) to configure Loguru JSON stdout sink, stdlib bridge, and sensitive data redaction helpers.
- [x] 2.2 Add FastAPI middleware/contextvars so per-request logs include request_id, trace_id/span_id (when available), and hashed image metadata.
- [x] 2.3 Replace existing print/log statements in runtime/service modules with Loguru calls using structured fields.

## 3. OpenTelemetry Tracing

- [x] 3.1 Implement `service/observability.py` bootstrap that initializes TracerProvider with resource attributes, sampling, OTLP exporter, and graceful disablement.
- [x] 3.2 Instrument FastAPI app via `FastAPIInstrumentor` and wrap runtime stages (`load`, `predict`, preprocess/inference/postprocess) with manual spans carrying provider/batch attributes.
- [x] 3.3 Ensure exporter retry/backoff is non-blocking and logs warnings when the collector is unreachable.

## 4. OpenTelemetry Metrics

- [x] 4.1 Configure MeterProvider + periodic OTLP reader honoring env-configured interval; define counters/histograms for throughput, latency, error ratios, and batch sizes.
- [x] 4.2 Integrate optional GPU metrics (NVML) that publish memory/device gauges when CUDA provider is active and degrade gracefully when unavailable.
- [x] 4.3 Wire metric recording into request handlers and runtime so data flows alongside tracing.

## 5. Surfacing Status, Tests, and Docs

- [x] 5.1 Update `/healthz` and `/readyz` responses to expose observability status fields (logging mode, tracing_enabled, metrics_enabled).
- [x] 5.2 Add/extend automated tests (unit or integration) covering settings validation, log serialization, and toggling tracing/metrics.
- [x] 5.3 Document Fluent Bit + OTEL wiring in README/compose files, including sample env blocks and guidance for disabling/enabling features.
