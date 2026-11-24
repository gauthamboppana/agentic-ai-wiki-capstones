"""Dual-LLM separation.

The privileged planner can call tools but never sees untrusted text. A quarantined reader
sees untrusted text but has no tools, and its output is forced into a small schema. Free text
it extracts (a search query, a link) is stored in a VariableStore and the planner only gets a
symbolic reference such as $Q1. The controller swaps references for values when it executes a
tool, so data can flow to a tool without ever passing through the planner's context.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from .caching import ExactCache, SemanticCache
from .model import CATEGORIES, URL, ModelRequest, ModelResponse

REF = re.compile(r"\$[A-Z]\d+")
SCHEMAS: dict[str, dict[str, str]] = {
    "issue": {"category": "one of " + ", ".join(CATEGORIES + ["general"]), "query": "short search query",
              "urls": "list of links found"},
    "kb": {"kb_id": "article id like KB-123 or null"},
    "page": {"status": "ok or missing"},
}


@dataclass
class VariableStore:
    values: dict[str, str] = field(default_factory=dict)
    _counts: dict[str, int] = field(default_factory=dict)

    def put(self, prefix: str, value: str) -> str:
        self._counts[prefix] = self._counts.get(prefix, 0) + 1
        ref = f"${prefix}{self._counts[prefix]}"
        self.values[ref] = value
        return ref

    def deref(self, value: Any) -> Any:
        if isinstance(value, str):
            return REF.sub(lambda m: self.values.get(m.group(0), m.group(0)), value)
        if isinstance(value, dict):
            return {k: self.deref(v) for k, v in value.items()}
        return value


def validate(kind: str, data: dict[str, Any]) -> dict[str, Any]:
    """Coerce reader output into the schema. Anything else is dropped."""
    if kind == "issue":
        cat = data.get("category")
        urls = [u for u in data.get("urls") or [] if isinstance(u, str) and u.startswith("https://")]
        return {"category": cat if cat in CATEGORIES else "general",
                "query": str(data.get("query") or "")[:120], "urls": urls[:2]}
    if kind == "kb":
        kb = data.get("kb_id")
        return {"kb_id": kb if isinstance(kb, str) and re.fullmatch(r"KB-\d{3}", kb) else None}
    return {"status": "ok" if data.get("status") == "ok" else "missing"}


@dataclass
class QuarantinedReader:
    call: Callable[[ModelRequest], ModelResponse]     # runs the reader model (cascade and spans included)
    store: VariableStore
    exact: ExactCache | None = None
    semantic: SemanticCache | None = None
    on_cache: Callable[[str, bool], None] = lambda kind, hit: None

    def read(self, kind: str, text: str, flagged: bool) -> dict[str, Any]:
        data = self._extract(kind, text, cacheable=not flagged)
        view: dict[str, Any] = {"kind": kind, "ref": self.store.put("T", text), "flagged": flagged}
        if kind == "issue":
            view.update(category=data["category"], query_ref=self.store.put("Q", data["query"]),
                        url_refs=[self.store.put("U", u) for u in data["urls"]])
        else:
            view.update(data)
        return view

    def _extract(self, kind: str, text: str, cacheable: bool) -> dict[str, Any]:
        key = ExactCache.key("read", kind, text)
        if cacheable and self.exact is not None:
            hit = self.exact.get(key)
            self.on_cache("exact", hit is not None)
            if hit is not None:
                return hit
        if cacheable and self.semantic is not None and kind == "issue":
            hit = self.semantic.get(kind, text)
            self.on_cache("semantic", hit is not None)
            if hit is not None:
                # Reuse what depends on meaning (category, query). Links are exact data: take them from this text.
                return validate(kind, {**hit[0], "urls": URL.findall(text)})
        content = json.dumps({"kind": kind, "text": text, "return": SCHEMAS[kind]})
        resp = self.call(ModelRequest(system="quarantined-reader", messages=[{"role": "user", "content": content}],
                                      task="read"))
        data = validate(kind, resp.data or {})
        if cacheable:
            if self.exact is not None:
                self.exact.put(key, data)
            if self.semantic is not None and kind == "issue":
                self.semantic.put(kind, text, data)
        return data
