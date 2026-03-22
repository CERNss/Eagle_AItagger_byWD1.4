## Context

The Eagle AI Tagger now runs primarily as a FastAPI service (see `service/app.py`) backed by `TaggerRuntime` and ONNXRuntime GPU sessions. Logging is limited to implicit stdout noise from FastAPI/Uvicorn plus a handful of `print` statements deep in the runtime, so SREs cannot correlate failed requests with GPU state, nor can they stream structured events into Fluent Bit. Tracing and metrics are entirely absent, meaning we cannot measure request latency per model provider, batching behavior, or tag throughput when the service is deployed inside Docker/Compose alongside Eagle libraries. We need to inject observability without changing tagging semantics, keep the container footprint light, and allow ops teams to toggle exporters/endpoints via environment variables.

## Goals / Non-Goals

**Goals:**
- Provide structured JSON logs via Loguru on stdout so Fluent Bit can scrape, enrich, and ship them without custom sidecars.
- Add OpenTelemetry tracing spans that follow each request from FastAPI entry through runtime load, preprocess, inference, and postprocess steps, exporting via OTLP.
- Publish OTEL metrics (counters/gauges/histograms) covering request volume, latency, GPU provider usage, error ratios, and batch throughput with configurable intervals.
- Introduce configuration surfaces for observability (log level, OTEL endpoints, resource metadata, feature flags) with safe defaults when disabled.
- Keep inference hot path overhead below ~3% CPU/GPU by batching spans/metrics and allowing sampling configuration.

**Non-Goals:**
- Building or maintaining the downstream Fluent Bit / OTEL Collector infrastructure; we only emit stdout and OTLP traffic.
- Rewriting Eagle Desktop integrations or metadata schemas; tagging semantics stay identical.
- Providing persistent log storage or alerting rules; consumers must configure their monitoring stacks.
- Supporting legacy Windows batch runners (run.bat) in this change; focus remains on the Python service path.

## Decisions

### Structured logging with Loguru JSON sink
- Replace ad-hoc prints with a single Loguru configuration executed at process start (e.g., in `main.py` before `uvicorn.run`). We will `logger.remove()` the default sink and add a stdout sink that formats records as JSON containing timestamp, level, message, `request_id`, `trace_id`, `span_id`, `image_hash`, `provider`, and elapsed milliseconds. Loguru's `serialize=True` handles JSON encoding.
- A helper `setup_logging(settings: Settings)` will live in a new module (e.g., `service/logging.py`) and also bridge Python's stdlib logging via `logger.add` so dependencies like FastAPI/Uvicorn go through the same pipeline. Fluent Bit can then ingest stdout without additional file mounts.
- Request-specific context will use `contextvars` plus FastAPI middleware so any log emitted within a request automatically carries correlation fields. Image paths will be SHA-256 hashed (or truncated) before logging to avoid leaking library paths while still allowing join with Eagle metadata.
- Alternatives considered: sticking with stdlib `logging` (more verbose setup) or using structlog (heavier dependency). Loguru offers the lightest integration effort and built-in JSON output.

### OTEL tracing instrumentation strategy
- Use `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, and `FastAPIInstrumentor` to auto-create spans for HTTP endpoints. We will initialize a `TracerProvider` with resource attributes derived from settings (service.name, service.version, deployment.environment) in `service/observability.py` at startup.
- Custom spans will wrap runtime stages: `runtime.load`, `runtime.predict`, `image.preprocess`, `onnx.inference`, and `tags.postprocess`. Each span records attributes like `model.provider`, `batch.size`, `image.count`, and GPU/cpu device info from `onnxruntime`. Exceptions bubble up with status code set per OTEL spec.
- Sampling defaults to `ParentBased(TraceIdRatioBased)` with a configurable ratio env (default 0.1) to limit overhead. When disabled, the OTEL SDK will use a `NoOpTracerProvider`, effectively removing tracing costs.
- Alternatives: instrumentation via `opentelemetry-instrumentation-uvicorn` or asynchronous exporters. We prefer manual bootstrap for tighter control and compatibility with Compose, while still allowing future auto-instrumentation if needed.

### Metrics collection plan
- Initialize a `MeterProvider` with OTLP metric exporter (same endpoint as traces by default) and a periodic `PushMetricReader`. Metrics to emit:
  - Counter `tagger.request.count` with attributes `endpoint`, `status`, `provider`.
  - Histogram `tagger.request.latency.ms` measuring total request time.
  - Gauge/UpDownCounter for `tagger.gpu.load` (if `pynvml` present) to monitor provider pressure.
  - Counter `tagger.batch.size` capturing items per batch inference.
- Runtime hooks will measure elapsed time in milliseconds (already available) and record to the histogram; metrics instrumentation sits alongside tracing to reuse the same context. Collection interval and exporter endpoint become env-configurable.
- Alternative considered: Prometheus client + /metrics endpoint. Rejected because Fluent Bit/OTEL Collector flow already established and Prometheus scraping would add another port.

### Configuration surface and feature flags
- Extend `service/settings.py` with an `ObservabilitySettings` dataclass nested within `Settings`, capturing:
  - `log_level`, `log_format` (json/plain), `log_include_trace` booleans.
  - `otel_enabled`, `otel_exporter_endpoint`, `otel_headers`, `otel_trace_sample_ratio`, `otel_metrics_enabled`.
  - `fluentbit_annotations` optional map for future use.
- Environment variables default to disable tracing/metrics while keeping structured logging on (since JSON stdout is benign). When OTEL is disabled, we skip initializing providers/exporters to avoid needless threads.
- Provide validation to ensure endpoints use http(s) and sample ratios lie between 0 and 1. Compose/Docker docs will document how to wire Fluent Bit (already scraping stdout) plus how to point OTEL exporter at a collector service name.

### Dependency and bootstrapping changes
- Add `loguru`, `opentelemetry-sdk`, `opentelemetry-exporter-otlp-proto-http`, `opentelemetry-instrumentation-fastapi`, and `opentelemetry-instrumentation-logging` to `requirements*.txt`. Keep versions aligned with Python 3.10+ compatibility used in the service container.
- Update the Dockerfile entrypoint to invoke a new launcher (e.g., `python -m service.launch`) that calls `setup_logging()` and `setup_observability()` before running uvicorn. Compose file gains optional OTEL endpoint env variables plus documentation for the Fluent Bit sidecar.

## Risks / Trade-offs

- [Increased dependency footprint] → Mitigation: pin versions, update Docker base image layers, and ensure optional imports (OTEL) degrade gracefully when packages missing.
- [Runtime overhead from tracing/metrics] → Mitigation: default low sample ratio, allow disabling entirely, and benchmark inference throughput before/after (target <3% overhead).
- [Potential PII leakage in logs] → Mitigation: hash sensitive paths, redact tokens, and gate debug-level logging behind envs.
- [Exporter outages blocking requests] → Mitigation: use asynchronous OTLP exporters with retry/backoff and drop data on failure rather than blocking request threads.
- [Misconfiguration leading to empty logs] → Mitigation: add startup validation warnings and health endpoints that report observability status fields.

## Migration Plan

1. Introduce logging/observability bootstrap modules and wire them in `main.py`/`service/app.py`, defaulting to structured logging without OTEL enabled.
2. Add new dependencies and rebuild Docker image; ensure unit/functional tests run without OTEL endpoints configured.
3. Enable OTEL in staging by setting exporter env vars and verifying traces/metrics reach the collector; use synthetic load to validate Fluent Bit parsing.
4. Roll out to production clusters with gradual increases to trace sample ratio; monitor overhead metrics and adjust thresholds.
5. Update README/compose docs with instructions for enabling observability and verifying via `readyz` enhancements.

## Open Questions

- Do we need to surface trace/span IDs back to clients (e.g., via response headers) for easier support diagnostics?
- Should we expose a `/metrics` plaintext endpoint for Prometheus as a fallback if OTEL collectors are unavailable?
- Are there compliance requirements dictating how long logs can retain hashed image identifiers, influencing log level defaults?
