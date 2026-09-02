"""OpenTelemetry tracing configuration for ControlPlane.ai.

Feature-flagged via CONTROLPLANE_OTLP_ENDPOINT. If unset, uses no-op tracing.
"""
import logging
from typing import Optional

logger = logging.getLogger("controlplane.tracing")
_tracer = None


def configure_tracing(service_name: str = "controlplane-api", otlp_endpoint: str = ""):
    """Configure OpenTelemetry TracerProvider.

    If otlp_endpoint is non-empty, sets up OTLP HTTP/gRPC exporter and FastAPI instrumentation.
    Otherwise initializes standard Tracer with no-op exporter.
    """
    global _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name or "controlplane-api"})
        provider = TracerProvider(resource=resource)

        if otlp_endpoint and otlp_endpoint.strip():
            try:
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

                exporter = OTLPSpanExporter(endpoint=otlp_endpoint.strip())
                provider.add_span_processor(BatchSpanProcessor(exporter))
                logger.info("OpenTelemetry OTLP exporter configured for endpoint: %s", otlp_endpoint)
            except Exception as exp_err:
                logger.warning("Could not initialize OTLP exporter (%s); continuing with no-op exporter.", exp_err)

        trace.set_tracer_provider(provider)
        _tracer = trace.get_tracer(service_name or "controlplane-api")
        return _tracer
    except Exception as exc:
        logger.warning("OpenTelemetry setup skipped or failed: %s", exc)
        return None


def get_tracer(name: str = "controlplane"):
    """Get active tracer instance or standard tracer."""
    global _tracer
    if _tracer is not None:
        return _tracer
    try:
        from opentelemetry import trace
        return trace.get_tracer(name)
    except Exception:
        return None


def get_current_trace_id() -> str:
    """Return the active span's trace_id as a 32-character hex string, or '-' if none."""
    try:
        from opentelemetry import trace
        span = trace.get_current_span()
        if span and span.get_span_context().is_valid:
            return format(span.get_span_context().trace_id, "032x")
    except Exception:
        pass
    return "-"
