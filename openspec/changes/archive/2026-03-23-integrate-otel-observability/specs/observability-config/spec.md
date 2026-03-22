## ADDED Requirements

### Requirement: Environment-driven settings
Observability SHALL be controlled exclusively via environment variables surfaced through `service.settings.Settings`, covering log level/format, OTEL exporter endpoint, OTEL headers, trace sample ratio, metric push interval, and feature toggles for logging/tracing/metrics.

#### Scenario: Service starts with custom env
- **WHEN** HOST, LOG_LEVEL, OTEL_EXPORTER_ENDPOINT, OTEL_TRACE_SAMPLE_RATIO, and METRIC_PUSH_INTERVAL are provided
- **THEN** the resolved Settings object SHALL reflect these values and pass them into logging/tracing bootstrap without requiring code changes.

### Requirement: Safe defaults
When observability-related variables are omitted, the runtime SHALL default to JSON logging enabled, tracing disabled, metrics disabled, OTLP endpoints unset, and sampling ratio 0.1, ensuring existing deployments keep working.

#### Scenario: No env overrides
- **WHEN** the service starts with no new env variables defined
- **THEN** it SHALL produce structured stdout logs but skip initializing OTEL exporters, preventing startup regressions.

### Requirement: Validation and errors
Settings parsing SHALL validate that numeric values fall within allowed ranges (e.g., 0 ≤ sample ratio ≤ 1) and that OTLP endpoints use http/https; invalid values SHALL abort startup with a clear error message before any inference occurs.

#### Scenario: Invalid sample ratio provided
- **WHEN** OTEL_TRACE_SAMPLE_RATIO=2 is supplied
- **THEN** Settings.validate() SHALL raise ValueError explaining the allowed range and prevent the service from listening on the network.

### Requirement: Status reporting
Health/ready endpoints SHALL reflect observability enablement state (e.g., fields for `logging_mode`, `tracing_enabled`, `metrics_enabled`) so operators can verify configuration without shell access.

#### Scenario: Operator checks /readyz
- **WHEN** `/readyz` is called after startup
- **THEN** the JSON response SHALL include booleans or strings describing whether tracing and metrics exporters are active.
