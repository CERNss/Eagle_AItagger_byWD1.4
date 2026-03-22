## Why

Running the tagger as a FastAPI microservice inside Linux containers leaves us blind when requests fail or GPU inference degrades. Logs are currently ad-hoc `print` statements surfaced only in the container stdout, and there is no tracing or metrics pipeline to show per-request latency, GPU provider usage, or tagging success rates. As we scale deployments, SRE teams need centralized observability to satisfy on-call response targets and correlate regressions with input workloads and GPU health.

## What Changes

- Adopt `loguru` as the unified application logger, configure JSON output to stdout, and ensure every FastAPI request plus runtime events emit structured fields (request id, image path hash, provider, elapsed ms) so Fluent Bit can scrape and forward them.
- Thread OpenTelemetry instrumentation through the service (FastAPI app, runtime inference, external file I/O) to export traces to an OTLP endpoint, letting us follow a request from HTTP entry to ONNX inference and tag post-processing.
- Emit service-level OTEL metrics (request counts, latency histograms, error ratios, GPU provider usage, batch throughput) with configurable collection intervals, so Grafana dashboards can track SLA drift.
- Extend settings/env configuration to surface OTEL exporter endpoints, resource attributes, sampling configuration, and Fluent Bit toggles, while keeping backwards-compatible defaults when observability is disabled.
- Update Docker/compose runtime docs to describe the Fluent Bit sidecar wiring and any new dependencies (`loguru`, `opentelemetry-sdk`, exporters), ensuring the requirements/install steps remain reproducible.

## Capabilities

### New Capabilities
- `structured-logging`: Standardizes service logging via loguru with JSON payloads, correlation fields, and stdout routing compatible with Fluent Bit collectors.
- `otel-tracing`: Provides OTEL trace spans across FastAPI endpoints and the tagging runtime (load, preprocess, inference, postprocess) with configurable OTLP exporters and sampling.
- `otel-metrics`: Publishes OTEL metrics (counters, histograms, gauges) summarizing inference throughput, latency, GPU provider usage, and error rates, ready for Prometheus/Grafana via Fluent Bit or OTEL Collector pipelines.
- `observability-config`: Adds validated settings/env vars controlling log levels, OTEL resource metadata, exporter endpoints, and feature flags so ops teams can enable/disable observability per deployment.

### Modified Capabilities
- _None._ Existing functionality remains, but runtime modules will gain instrumentation hooks without changing tagging semantics.

## Impact

- `service/app.py`, `service/runtime.py`, `service/image_utils.py`: add loguru instrumentation, trace spans, and timing events around request handling and inference.
- `service/settings.py`: introduce new env-backed configuration for log levels, OTEL exporters, resource attributes, and Fluent Bit toggles.
- `service/schemas.py`, `service/runtime.py`: surface trace/span identifiers or diagnostic info in responses when helpful (optional fields).
- `main.py`, `Dockerfile`, `compose.yaml`: ensure the process initializes loguru/OTEL early, ships structured stdout, and documents any sidecar/collector dependencies.
- `requirements.txt`, `requirements-dev.txt`: add `loguru`, `opentelemetry-sdk`, OTLP exporters, and testing utilities.
- `README.md` (and ops docs): describe how to enable observability, required env vars, and Fluent Bit pipeline expectations.
