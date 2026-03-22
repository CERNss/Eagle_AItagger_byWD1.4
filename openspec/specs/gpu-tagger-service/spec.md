# gpu-tagger-service Specification

## Purpose
TBD - created by archiving change linux-gpu-tagger-service. Update Purpose after archive.
## Requirements
### Requirement: Service exposes runtime health state
The system SHALL expose HTTP health endpoints that report whether the tagging service process is running and whether the model runtime has been loaded successfully.

#### Scenario: Liveness check succeeds
- **WHEN** a client sends `GET /healthz`
- **THEN** the service returns HTTP 200
- **THEN** the response includes the active runtime provider and model-loaded state

#### Scenario: Readiness depends on model availability
- **WHEN** a client sends `GET /readyz` after successful model initialization
- **THEN** the service returns HTTP 200
- **THEN** the response indicates the service is ready to process tagging requests

#### Scenario: Readiness returns 503 before model is loaded
- **WHEN** a client sends `GET /readyz` before model initialization completes
- **THEN** the service returns HTTP 503
- **THEN** the response explains the runtime is not yet ready

### Requirement: Service tags a single image path
The system SHALL accept a request containing an image path and return tag inference results for that image without mutating Eagle metadata files.

#### Scenario: Tagging a valid image path
- **WHEN** a client sends `POST /tag` with an existing image path and optional threshold overrides
- **THEN** the service returns HTTP 200
- **THEN** the response includes the resolved provider, image path, ordered tag list, and elapsed processing time

#### Scenario: Missing image path is rejected
- **WHEN** a client sends `POST /tag` with a path that does not exist
- **THEN** the service returns an HTTP 404-style client error
- **THEN** the response explains that the image was not found

#### Scenario: Image path outside IMAGE_ROOT is rejected
- **WHEN** `IMAGE_ROOT` is configured and a client sends `POST /tag` with a path that resolves outside that root
- **THEN** the service returns HTTP 403
- **THEN** the response explains the path is outside the permitted root

### Requirement: Service supports batch tagging requests
The system SHALL accept a batch of image paths and return per-image tagging results in a single response. The batch size is bounded by the `BATCH_LIMIT` setting and enforced at runtime.

#### Scenario: Batch request returns individual results
- **WHEN** a client sends `POST /tag/batch` with multiple image paths
- **THEN** the service returns HTTP 200
- **THEN** the response includes one result item per requested path with success or error information

#### Scenario: One invalid path does not cancel the batch
- **WHEN** a batch request includes both valid and invalid image paths
- **THEN** the service continues processing the valid paths
- **THEN** the invalid path is reported as an item-level error in the batch response

#### Scenario: Batch exceeding BATCH_LIMIT is rejected
- **WHEN** a client sends `POST /tag/batch` with more paths than the configured `BATCH_LIMIT`
- **THEN** the service returns HTTP 400
- **THEN** the response states the batch size and configured limit

### Requirement: Runtime reuses a single model session
The system SHALL initialize the ONNX Runtime session during service startup and reuse that session across requests in the same process.

#### Scenario: Startup loads model once
- **WHEN** the service process starts successfully
- **THEN** the model session and tag dictionary are loaded before the service is marked ready
- **THEN** subsequent tagging requests use the existing in-memory runtime

### Requirement: Tag post-processing is configurable per deployment
The system SHALL support environment-driven post-processing of raw inference scores to control output format, filtering, and sorting without requiring code changes.

#### Scenario: Underscore replacement is applied by default
- **WHEN** `REPLACE_UNDERSCORE` is true (the default)
- **THEN** tag names returned in responses use spaces instead of underscores
- **THEN** tags listed in `UNDERSCORE_EXCLUDES` retain their original underscores

#### Scenario: Tags can be escaped for prompt use
- **WHEN** `ESCAPE_TAGS` is true
- **THEN** backslashes and parentheses in returned tag names are escaped
- **THEN** the escaped form is suitable for direct use in prompt strings

#### Scenario: Additional tags are always appended
- **WHEN** `ADDITIONAL_TAGS` is set to a comma-separated list
- **THEN** each listed tag appears in every response with score 1.0
- **THEN** tags already present from inference are not duplicated

#### Scenario: Excluded tags are always removed
- **WHEN** `EXCLUDE_TAGS` is set to a comma-separated list
- **THEN** the listed tags are absent from every response regardless of their inference score

#### Scenario: Alphabetical sort can be enabled
- **WHEN** `SORT_ALPHABETICALLY` is true
- **THEN** returned tags are sorted by name ascending instead of by score descending

