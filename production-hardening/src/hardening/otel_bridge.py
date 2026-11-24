"""Optional: mirror built-in spans into the OpenTelemetry SDK.

Install the extra with `uv sync --extra otel`, then:

    from hardening.otel_bridge import install_otel
    install_otel(tracer)      # exporter reads OTEL_EXPORTER_OTLP_TRACES_ENDPOINT and _HEADERS

Spans then flow to any OTLP backend, for example a local Phoenix server or a Langfuse project.
"""
from __future__ import annotations

from typing import Any

from .tracing import Span, Tracer


class OtelBridge:
    def __init__(self, provider: Any = None):
        from opentelemetry import trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        if provider is None:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            provider = TracerProvider(resource=Resource.create({"service.name": "issue-agent"}))
            provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
        self._trace = trace
        self.provider = provider
        self.otel = provider.get_tracer("hardening")
        self._live: dict[str, Any] = {}

    def on_start(self, span: Span) -> None:
        parent = self._live.get(span.parent_id or "")
        ctx = self._trace.set_span_in_context(parent) if parent is not None else None
        self._live[span.span_id] = self.otel.start_span(span.name, context=ctx, start_time=span.start_ns)

    def on_end(self, span: Span) -> None:
        o = self._live.pop(span.span_id)
        for k, v in span.attributes.items():
            if v is not None:
                o.set_attribute(k, v if isinstance(v, (bool, int, float, str)) else str(v))
        if span.status == "error":
            o.set_status(self._trace.Status(self._trace.StatusCode.ERROR))
        o.end(end_time=span.end_ns)


def install_otel(tracer: Tracer, provider: Any = None) -> OtelBridge:
    bridge = OtelBridge(provider)
    tracer.listeners.append(bridge)
    return bridge
