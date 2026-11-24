"""Small-then-large model cascade with a confidence check."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .model import CATEGORIES, Model, ModelRequest, ModelResponse


def validity_confidence(request: ModelRequest, resp: ModelResponse) -> float:
    """Fallback when a model reports no confidence: is the answer well formed?"""
    if request.task == "read":
        data = resp.data or {}
        if data.get("kind") == "issue" and data.get("category") not in CATEGORIES + ["general"]:
            return 0.0
        return 0.8
    names = {t["name"] for t in request.tools}
    if any(tc.name not in names for tc in resp.tool_calls):
        return 0.0
    return 0.8 if (resp.tool_calls or resp.text) else 0.0


@dataclass
class Cascade:
    small: Model
    large: Model
    threshold: float = 0.7

    def run(self, request: ModelRequest, call: Callable[[Model, ModelRequest], ModelResponse]) -> ModelResponse:
        """`call` invokes one model (inside its own span). Escalate when the small answer is unsure."""
        first = call(self.small, request)
        conf = first.confidence if first.confidence is not None else validity_confidence(request, first)
        if conf >= self.threshold:
            return first
        return call(self.large, request)
