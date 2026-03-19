from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from fastapi import FastAPI
from loguru import logger
from opentelemetry import metrics as otel_metrics
from opentelemetry import trace as otel_trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.metrics import CallbackOptions, Observation
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased
from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter

from .logging import setup_logging
from .settings import ObservabilitySettings, Settings


@dataclass(frozen=True)
class ObservabilityStatus:
    logging_mode: str
    tracing_enabled: bool
    metrics_enabled: bool
    trace_sample_ratio: float
    metric_interval_seconds: int
    exporter_endpoint: str | None


class MetricsRecorder:
    def __init__(self) -> None:
        self.enabled = False
        self.request_counter = None
        self.request_latency = None
        self.error_counter = None
        self.images_processed_counter = None
        self._gpu_memory_state: dict[str, int] = {}

    def configure(self, meter_provider: MeterProvider, settings: ObservabilitySettings) -> None:
        meter = meter_provider.get_meter("eagle.ai.tagger")
        self.request_counter = meter.create_counter(
            "tagger.request.count",
            unit="1",
            description="Number of HTTP requests processed by the tagger service.",
        )
        self.request_latency = meter.create_histogram(
            "tagger.request.latency.ms",
            unit="ms",
            description="End-to-end latency for tagger HTTP requests.",
        )
        self.error_counter = meter.create_counter(
            "tagger.request.errors",
            unit="1",
            description="Count of failed tagger requests grouped by error class.",
        )
        self.images_processed_counter = meter.create_counter(
            "tagger.images.processed",
            unit="items",
            description="Total number of images handled by batch requests.",
        )

        def _gpu_callback(_: CallbackOptions) -> Iterable[Observation]:
            for provider, value in self._gpu_memory_state.items():
                yield Observation(value=value, attributes={"provider": provider})

        meter.create_observable_gauge(
            "tagger.gpu.memory.bytes",
            callbacks=[_gpu_callback],
            unit="By",
            description="GPU memory usage sampled from NVML when available.",
        )
        self.enabled = True

    def record_request(
        self,
        endpoint: str,
        status: str,
        provider: str,
        elapsed_ms: int,
        batch_size: int | None = None,
    ) -> None:
        if not self.enabled or self.request_counter is None or self.request_latency is None:
            return
        attributes = {"endpoint": endpoint, "status": status, "provider": provider}
        self.request_counter.add(1, attributes)
        self.request_latency.record(float(elapsed_ms), attributes)
        if batch_size is not None and self.images_processed_counter is not None:
            self.images_processed_counter.add(batch_size, attributes)

    def record_error(self, endpoint: str, provider: str, error_class: str) -> None:
        if not self.enabled or self.error_counter is None:
            return
        self.error_counter.add(1, {"endpoint": endpoint, "error.class": error_class, "provider": provider})

    def record_gpu(self, provider: str, memory_bytes: int | None) -> None:
        if not self.enabled or memory_bytes is None:
            return
        self._gpu_memory_state[provider] = memory_bytes


_metrics_recorder = MetricsRecorder()
_observability_status = ObservabilityStatus(
    logging_mode="json",
    tracing_enabled=False,
    metrics_enabled=False,
    trace_sample_ratio=0.0,
    metric_interval_seconds=60,
    exporter_endpoint=None,
)
_tracer_provider: TracerProvider | None = None
_meter_provider: MeterProvider | None = None


def setup_observability(app: FastAPI, settings: Settings) -> ObservabilityStatus:
    global _observability_status
    setup_logging(settings)
    observability = settings.observability
    tracing_enabled = _configure_tracing(app, observability)
    metrics_enabled = _configure_metrics(observability)
    _observability_status = ObservabilityStatus(
        logging_mode=observability.logging.log_format,
        tracing_enabled=tracing_enabled,
        metrics_enabled=metrics_enabled,
        trace_sample_ratio=observability.tracing.sample_ratio,
        metric_interval_seconds=observability.metrics.export_interval_seconds,
        exporter_endpoint=observability.tracing.exporter_endpoint or observability.metrics.exporter_endpoint,
    )
    return _observability_status


def shutdown_observability() -> None:
    """Flush and shut down OTel providers, then drain the Loguru async queue."""
    global _tracer_provider, _meter_provider
    if _tracer_provider is not None:
        try:
            _tracer_provider.shutdown()
            logger.info("tracing.shutdown.ok")
        except Exception as exc:
            logger.warning("tracing.shutdown.failed", error=str(exc))
        finally:
            _tracer_provider = None
    if _meter_provider is not None:
        try:
            _meter_provider.shutdown()
            logger.info("metrics.shutdown.ok")
        except Exception as exc:
            logger.warning("metrics.shutdown.failed", error=str(exc))
        finally:
            _meter_provider = None
    logger.complete()


def _configure_tracing(app: FastAPI, observability: ObservabilitySettings) -> bool:
    global _tracer_provider
    if not observability.tracing.enabled:
        return False
    resource = Resource.create(
        {
            "service.name": observability.service_name,
            "service.version": observability.service_version,
            "deployment.environment": observability.deployment_environment,
        }
    )
    sampler = ParentBased(TraceIdRatioBased(observability.tracing.sample_ratio))
    tracer_provider = TracerProvider(resource=resource, sampler=sampler)
    exporter = OTLPSpanExporter(
        endpoint=observability.tracing.exporter_endpoint,
        headers=dict(observability.tracing.headers),
    )
    processor = BatchSpanProcessor(exporter, schedule_delay_millis=500)
    tracer_provider.add_span_processor(processor)
    otel_trace.set_tracer_provider(tracer_provider)
    FastAPIInstrumentor.instrument_app(app, tracer_provider=tracer_provider)
    _tracer_provider = tracer_provider
    logger.info(
        "tracing.enabled",
        exporter_endpoint=observability.tracing.exporter_endpoint,
        sample_ratio=observability.tracing.sample_ratio,
    )
    return True


def _configure_metrics(observability: ObservabilitySettings) -> bool:
    global _metrics_recorder, _meter_provider
    if not observability.metrics.enabled:
        return False
    resource = Resource.create(
        {
            "service.name": observability.service_name,
            "service.version": observability.service_version,
            "deployment.environment": observability.deployment_environment,
        }
    )
    exporter = OTLPMetricExporter(
        endpoint=observability.metrics.exporter_endpoint,
        headers=dict(observability.metrics.headers),
    )
    reader = PeriodicExportingMetricReader(
        exporter,
        export_interval_millis=observability.metrics.export_interval_seconds * 1000,
    )
    provider = MeterProvider(metric_readers=[reader], resource=resource)
    otel_metrics.set_meter_provider(provider)
    _meter_provider = provider
    _metrics_recorder = MetricsRecorder()
    _metrics_recorder.configure(provider, observability)
    logger.info(
        "metrics.enabled",
        exporter_endpoint=observability.metrics.exporter_endpoint,
        interval_seconds=observability.metrics.export_interval_seconds,
    )
    return True


def metrics_recorder() -> MetricsRecorder:
    return _metrics_recorder


def observability_status() -> ObservabilityStatus:
    return _observability_status
