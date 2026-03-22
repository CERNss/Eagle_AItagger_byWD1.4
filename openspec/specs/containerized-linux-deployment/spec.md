# containerized-linux-deployment Specification

## Purpose
TBD - created by archiving change linux-gpu-tagger-service. Update Purpose after archive.
## Requirements
### Requirement: Service ships Linux container assets
The system SHALL provide Docker assets for building and running the tagger service on Linux.

#### Scenario: Image can be built for the service
- **WHEN** an operator builds the repository container image
- **THEN** the build uses a Linux-oriented runtime layout suitable for serving the FastAPI application

#### Scenario: Container entrypoint honours environment-driven configuration
- **WHEN** the container starts
- **THEN** the entrypoint delegates to `main.py` so that `HOST` and `PORT` environment variables are respected

#### Scenario: Service documents Linux startup
- **WHEN** an operator follows the repository documentation
- **THEN** the documented workflow describes Linux deployment, model mounting, and service startup rather than Windows batch execution

### Requirement: Deployment supports NVIDIA GPU access
The system SHALL provide a container deployment example that is compatible with Docker and NVIDIA GPU reservation.

#### Scenario: Compose config declares GPU access
- **WHEN** an operator reviews the example Compose configuration
- **THEN** the service declares NVIDIA GPU access and mounted model/image volumes

#### Scenario: Runtime configuration is environment-driven
- **WHEN** an operator sets model paths, tag paths, and threshold defaults through environment variables
- **THEN** the containerized service uses those settings without requiring Windows-specific configuration files

#### Scenario: Container restarts automatically on failure
- **WHEN** the service process exits unexpectedly
- **THEN** Docker restarts the container unless it was stopped intentionally

#### Scenario: Observability flushes even when startup fails
- **WHEN** the service fails to load the model at startup
- **THEN** any buffered OTel spans and metrics are flushed before the process exits
- **THEN** the container exits with a non-zero code so Docker restarts it

#### Scenario: Compose healthcheck targets the readiness endpoint
- **WHEN** Docker evaluates container health
- **THEN** the healthcheck calls `GET /readyz` so that health only passes after the model has loaded successfully

### Requirement: Linux package manifests exclude Windows-only runtime needs
The system SHALL publish service dependency manifests that avoid Windows-only packages in the primary Linux runtime path. Dev-only dependencies SHALL be isolated to a separate manifest.

#### Scenario: Service dependencies avoid Windows-only packages
- **WHEN** an operator installs the service runtime dependencies on Linux
- **THEN** the primary dependency set does not require Windows-only packages such as console helpers

#### Scenario: Unused packages are excluded from the service manifest
- **WHEN** an operator installs from `requirements.txt`
- **THEN** packages not imported by the service code (such as `packaging`, `psutil`) are absent

#### Scenario: Script-only dependencies live in the dev manifest
- **WHEN** an operator installs from `requirements-dev.txt`
- **THEN** packages used only by scripts and test tooling (such as `requests`, `pytest`, `httpx`) are included

#### Scenario: Dependency manifest is UTF-8 compatible
- **WHEN** Linux tooling reads the dependency manifest
- **THEN** the file is encoded in a format compatible with standard `pip install -r` usage

