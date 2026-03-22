## Purpose

Define metrics requirements for service throughput, latency, errors, and GPU behavior in the Eagle AI Tagger service.

## Requirements

### Requirement: Request metrics
The service SHALL emit OTEL metric instruments that track total tagged requests, HTTP status outcome, and batch sizes to quantify throughput.

#### Scenario: Batch tagging request completes
- **WHEN** `/tag/batch` processes N images
- **THEN** the counter `tagger.request.count` SHALL increase by 1 with attributes `{endpoint:"/tag/batch",status:"success",provider:<runtime provider>}` and a histogram sample SHALL record total elapsed milliseconds along with batch size N.

### Requirement: Error ratio tracking
Metric instruments SHALL capture failure counts by error class so alerting systems can detect spikes in inference exceptions or filesystem errors.

#### Scenario: Runtime raises FileNotFoundError
- **WHEN** a tagging request fails because an image path is missing
- **THEN** the counter `tagger.request.errors` SHALL increment with attributes identifying `error.class="FileNotFoundError"` and the affected endpoint.

### Requirement: GPU utilization metrics
When CUDA/GPU providers are active, the runtime SHALL emit gauges for GPU memory usage and provider name sourced from `pynvml` or ONNXRuntime telemetry; when GPU data is unavailable, metrics SHALL gracefully fallback without emitting invalid values.

#### Scenario: CUDAExecutionProvider active
- **WHEN** the runtime completes a request using the CUDA provider
- **THEN** gauge `tagger.gpu.memory.bytes` SHALL publish the memory used, and gauge `tagger.gpu.provider` SHALL reflect the provider name; if NVML is not present, the gauges SHALL be omitted with no errors.

### Requirement: Configurable export interval
Metric export SHALL use an OTLP periodic reader with an interval configurable via environment variable and defaulting to 60 seconds, ensuring that lowering the interval does not block request threads.

#### Scenario: Ops sets METRIC_PUSH_INTERVAL=15
- **WHEN** the service starts with interval 15 seconds
- **THEN** metric batches SHALL flush approximately every 15 seconds without increasing request latency beyond normal variance.
