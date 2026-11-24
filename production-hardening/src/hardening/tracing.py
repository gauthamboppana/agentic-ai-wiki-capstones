"""A small built-in tracer with OTLP-style JSON export.

Every model call and every tool call runs inside `tracer.span(...)`. Spans nest through a
stack, so a run produces one tree: agent.run > model.call / tool.call > guardrail checks.
Attribute names follow the OpenTelemetry GenAI conventions where one exists
(gen_ai.request.model, gen_ai.usage.input_tokens, gen_ai.tool.name), so the exported file
reads naturally in any OTLP viewer. Listeners let the optional OpenTelemetry bridge mirror
spans live (see otel_bridge.py).
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator


@dataclass
class Span:
    name: str
    trace_id: str
    span_id: str
    parent_id: str | None
    start_ns: int
    end_ns: int = 0
    attributes: dict[str, Any] = field(default_factory=dict)
    status: str = "ok"

    def set(self, **attrs: Any) -> None:
        self.attributes.update(attrs)


class Tracer:
    def __init__(self, service: str = "issue-agent", seed: str = "run"):
        self.service = service
        self.seed = seed
        self.spans: list[Span] = []
        self.listeners: list[Any] = []   # objects with on_start(span) and on_end(span)
        self._stack: list[Span] = []
        self._n = 0

    def _new_id(self, nbytes: int) -> str:
        self._n += 1
        return hashlib.sha256(f"{self.seed}:{self._n}".encode()).hexdigest()[: nbytes * 2]

    @contextmanager
    def span(self, name: str, **attrs: Any) -> Iterator[Span]:
        parent = self._stack[-1] if self._stack else None
        s = Span(name, parent.trace_id if parent else self._new_id(16), self._new_id(8),
                 parent.span_id if parent else None, time.time_ns(), attributes=dict(attrs))
        for listener in self.listeners:
            listener.on_start(s)
        self._stack.append(s)
        try:
            yield s
        except Exception as exc:
            s.status = "error"
            s.attributes["error.message"] = str(exc)
            raise
        finally:
            s.end_ns = time.time_ns()
            self._stack.pop()
            self.spans.append(s)
            for listener in self.listeners:
                listener.on_end(s)

    def find(self, name: str, where: Callable[[Span], bool] = lambda s: True) -> list[Span]:
        return [s for s in self.spans if s.name == name and where(s)]

    # -- export ------------------------------------------------------------------------
    def export_otlp(self) -> dict[str, Any]:
        """The OTLP/JSON shape that OTLP HTTP collectors accept at /v1/traces."""
        def value(v: Any) -> dict[str, Any]:
            if isinstance(v, bool):
                return {"boolValue": v}
            if isinstance(v, int):
                return {"intValue": str(v)}
            if isinstance(v, float):
                return {"doubleValue": v}
            return {"stringValue": str(v)}

        def attrs(d: dict[str, Any]) -> list[dict[str, Any]]:
            return [{"key": k, "value": value(v)} for k, v in sorted(d.items()) if v is not None]

        spans = [{
            "traceId": s.trace_id, "spanId": s.span_id, "parentSpanId": s.parent_id or "",
            "name": s.name, "kind": 1 if s.parent_id else 2,
            "startTimeUnixNano": str(s.start_ns), "endTimeUnixNano": str(s.end_ns),
            "attributes": attrs(s.attributes),
            "status": {"code": 2 if s.status == "error" else 1},
        } for s in sorted(self.spans, key=lambda s: s.start_ns)]
        return {"resourceSpans": [{
            "resource": {"attributes": attrs({"service.name": self.service})},
            "scopeSpans": [{"scope": {"name": "hardening.tracing", "version": "0.1.0"}, "spans": spans}],
        }]}

    def write_otlp(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.export_otlp(), indent=1))
        return path


def post_otlp_json(path: str | Path, endpoint: str, headers: dict[str, str] | None = None) -> int:
    """Send an exported file to an OTLP HTTP endpoint that accepts JSON. Not used in tests."""
    import urllib.request

    req = urllib.request.Request(endpoint, Path(path).read_bytes(),
                                 {"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=30) as r:  # noqa: S310 (endpoint comes from local config)
        return r.status
