## ADDED Requirements

### Requirement: Structured log serialization
The service SHALL emit every application log via Loguru using JSON serialization to stdout, including at minimum `timestamp`, `level`, `message`, `service.name`, `request_id` (when available), and deployment metadata fields so Fluent Bit can parse consistently.

#### Scenario: Log entry is emitted
- **WHEN** any component in the process writes a log at INFO or higher
- **THEN** the stdout line SHALL be valid JSON with the required fields present and no leading/trailing decoration.

### Requirement: Context propagation
The system SHALL attach correlation identifiers (request_id, trace_id, span_id) plus hashed image path metadata to every log generated during a FastAPI request lifecycle.

#### Scenario: Request-scoped log message
- **WHEN** a log is emitted while processing an HTTP request
- **THEN** the log SHALL include the correlation identifiers so downstream collectors can join logs with traces.

### Requirement: Stdlib logging bridge
The service SHALL route Python stdlib logging records (including FastAPI/Uvicorn internals) through Loguru so that all logs share the same format and sinks.

#### Scenario: Dependency emits stdlib log
- **WHEN** Uvicorn logs a request via `logging.getLogger("uvicorn")`
- **THEN** the resulting stdout entry SHALL appear in the same JSON format as Loguru-managed logs.

### Requirement: Sensitive value redaction
The logging subsystem SHALL hash or redact filesystem paths and authentication headers before serialization to prevent leaking Eagle library structure or OTEL credentials.

#### Scenario: Runtime logs image path
- **WHEN** the runtime logs the source path of an image being tagged
- **THEN** the path SHALL be converted to a deterministic hash or truncated form with no direct filesystem path in the JSON payload.
