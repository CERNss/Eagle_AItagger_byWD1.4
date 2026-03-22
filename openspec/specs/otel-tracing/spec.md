## Purpose

Define distributed tracing requirements for HTTP requests and runtime inference stages in the Eagle AI Tagger service.

## Requirements

### Requirement: HTTP span coverage
FastAPI endpoints SHALL be instrumented with OpenTelemetry so every inbound request produces a root span labeled with the route, HTTP method, response status, and requester metadata.

#### Scenario: Request hits /tag endpoint
- **WHEN** a client submits a POST /tag request
- **THEN** an OTEL span named `HTTP POST /tag` SHALL be recorded with attributes for status code, content length, and client IP.

### Requirement: Runtime stage spans
The inference pipeline SHALL create nested spans for `runtime.load`, `runtime.predict`, `image.preprocess`, `onnx.inference`, and `tags.postprocess`, capturing timing and provider/device attributes for each stage.

#### Scenario: Predict pipeline executes
- **WHEN** the runtime processes an image batch
- **THEN** each stage SHALL emit its own child span with attributes for model provider, batch size, and GPU device, enabling flame-graph style trace views.

### Requirement: Configurable sampling
Trace sampling SHALL default to a configurable ratio (ParentBased + TraceIdRatioBased) exposed via environment variable and accept values from 0.0 to 1.0 inclusive. Setting the ratio to 0 SHALL disable tracing entirely without disrupting logging/metrics.

#### Scenario: Ops disables tracing
- **WHEN** `OTEL_TRACE_SAMPLE_RATIO=0`
- **THEN** no spans SHALL be exported and the runtime SHALL operate without allocating OTEL tracer providers.

### Requirement: Export resiliency
Trace export SHALL use OTLP/HTTP with retries and non-blocking behavior; exporter outages SHALL drop spans after retry budget rather than blocking request threads.

#### Scenario: Collector unreachable
- **WHEN** the configured OTLP endpoint is unreachable for more than the retry timeout
- **THEN** spans SHALL be dropped and a warning log emitted, but the FastAPI response SHALL still complete successfully.
