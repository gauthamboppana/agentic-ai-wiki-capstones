"""Model interface and a deterministic fake model.

The agent talks to any model through `complete(ModelRequest) -> ModelResponse`. FakeModel
implements that interface offline. It behaves like a helpful but gullible model: it follows
instructions it finds inside tool output, which is exactly the weakness the guardrails and
the dual-LLM split exist to contain. It also simulates provider-side prompt-prefix caching
and reports token usage for every call.
"""
from __future__ import annotations

import base64
import hashlib
import itertools
import json
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

from .tokens import Usage, count_tokens

CATEGORIES = ["billing", "login", "bug", "feature", "performance"]
KEYWORDS: dict[str, list[str]] = {
    "billing": ["invoice", "charged", "charge", "refund", "payment", "billing", "subscription", "receipt"],
    "login": ["password", "login", "locked", "sign", "2fa", "reset", "access"],
    "bug": ["error", "crash", "broken", "fails", "failed", "exception", "export", "csv", "bug"],
    "feature": ["feature", "request", "suggestion", "wish", "mode", "support for"],
    "performance": ["slow", "timeout", "latency", "loading", "lag", "seconds"],
}
ZERO_WIDTH = re.compile("[​‌‍⁠﻿]")
URL = re.compile(r"https?://[^\s)\]\"'<>]+")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ModelRequest:
    system: str
    messages: list[dict[str, Any]]
    tools: list[dict[str, Any]] = field(default_factory=list)
    task: str = "plan"            # "plan" for the planner, "read" for the quarantined reader
    cache_prefix: bool = False    # ask the provider to cache the system + tools prefix


@dataclass
class ModelResponse:
    model: str
    tier: str
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    confidence: float | None = None
    summary: str = ""             # a short, loggable reasoning summary (never raw chain of thought)
    data: dict[str, Any] | None = None   # structured output for task="read"
    call_id: str = ""


class Model(Protocol):
    name: str
    tier: str
    def complete(self, request: ModelRequest) -> ModelResponse: ...


def prefix_text(request: ModelRequest) -> str:
    return request.system + "\n" + json.dumps(request.tools, sort_keys=True)


def request_text(request: ModelRequest) -> str:
    return prefix_text(request) + "\n" + json.dumps(request.messages, sort_keys=True)


def classify(text: str, tier: str) -> tuple[str, float]:
    """Keyword triage. The small tier is weaker on ambiguous text: it can pick the runner-up."""
    low = text.lower()
    scores = {c: sum(low.count(k) for k in KEYWORDS[c]) for c in CATEGORIES}
    ranked = sorted(CATEGORIES, key=lambda c: (-scores[c], CATEGORIES.index(c)))
    top, second = ranked[0], ranked[1]
    margin = scores[top] - scores[second]
    if scores[top] == 0:
        return "general", 0.4 if tier == "small" else 0.8
    if margin <= 1:
        return (second, 0.45) if tier == "small" else (top, 0.8)
    return top, 0.92 if tier == "small" else 0.97


def _normalise(text: str) -> str:
    """What a model 'sees': zero-width characters vanish and base64 blobs get decoded."""
    text = ZERO_WIDTH.sub("", text)
    for blob in re.findall(r"[A-Za-z0-9+/=]{24,}", text):
        try:
            decoded = base64.b64decode(blob, validate=True).decode("utf-8")
        except Exception:
            continue
        text += "\n" + decoded
    return text


def find_directives(text: str, tools: list[dict[str, Any]]) -> list[ToolCall]:
    """Tool calls that the text asks for, with arguments written as key=value."""
    text = _normalise(text)
    found = []
    for spec in tools:
        name, params = spec["name"], spec["parameters"]
        for m in re.finditer(re.escape(name), text):
            window = text[m.end(): m.end() + 140]
            args = {k: (q if q else v) for k, q, v in re.findall(r"(\w+)\s*=\s*(?:\"([^\"]*)\"|([^\s,)\]>\"]+))", window)}
            args = {k: v for k, v in args.items() if k in params["properties"]}
            if all(r in args for r in params.get("required", [])):
                found.append(ToolCall("", name, args))
    return found


def find_reply_injections(text: str, system: str) -> list[str]:
    """Extra content the text asks to be put into the reply (links, images, secrets)."""
    text = _normalise(text)
    extra = []
    asks_reply = re.search(r"(?is)\b(include|add|append|embed|put)\b.{0,80}\b(reply|response|answer)\b", text) \
        or re.search(r"(?is)\b(reply|response|answer)\b.{0,80}\b(include|add|append|embed)\b", text)
    if asks_reply:
        img = re.search(r"!?\[[^\]]*\]\(https?://[^)]+\)", text)
        if img:
            extra.append(img.group(0))
        elif URL.search(text):
            extra.append(URL.search(text).group(0))
        if re.search(r"(?i)(escalation|internal|secret) code", text):
            secret = re.search(r"CANARY-[A-Z0-9]+", system)
            if secret:
                extra.append(secret.group(0))
    return extra


class FakeModel:
    """Deterministic model with a planner policy and a structured reader."""

    _ids = itertools.count(1)

    def __init__(self, tier: str = "large", name: str | None = None):
        self.tier = tier
        self.name = name or f"fake-{tier}"
        self.calls: list[str] = []          # call ids, one per real invocation
        self.seen: list[ModelRequest] = []  # every request received, for privacy tests
        self._prefixes: set[str] = set()

    # -- the interface -----------------------------------------------------------------
    def complete(self, request: ModelRequest) -> ModelResponse:
        call_id = f"{self.name}-{next(self._ids)}"
        self.calls.append(call_id)
        self.seen.append(request)
        resp = self._read(request) if request.task == "read" else self._plan(request)
        resp.call_id = call_id
        resp.usage = self._usage(request, resp)
        return resp

    def _usage(self, request: ModelRequest, resp: ModelResponse) -> Usage:
        prefix = prefix_text(request)
        key = hashlib.sha256(prefix.encode()).hexdigest()
        cached = 0
        if request.cache_prefix:
            if key in self._prefixes:
                cached = count_tokens(prefix)
            self._prefixes.add(key)
        out = resp.text + json.dumps([tc.arguments for tc in resp.tool_calls]) + json.dumps(resp.data or {})
        return Usage(count_tokens(request_text(request)), cached, count_tokens(out) + count_tokens(resp.summary))

    # -- quarantined reader --------------------------------------------------------------
    def _read(self, request: ModelRequest) -> ModelResponse:
        payload = json.loads(request.messages[-1]["content"])
        kind, text = payload["kind"], payload["text"]
        data: dict[str, Any] = {"kind": kind}
        conf = 0.95
        if kind == "issue":
            title = re.search(r"(?m)^Title: (.*)$", text)
            data["category"], conf = classify(text, self.tier)
            data["query"] = title.group(1) if title else text[:80]
            data["urls"] = [u for u in URL.findall(text) if "docs.example.org" in u][:1]
        elif kind == "kb":
            kb = re.search(r"KB-\d{3}", text)
            data["kb_id"] = kb.group(0) if kb else None
        else:
            data["status"] = "ok" if text and not text.startswith("404") else "missing"
        return ModelResponse(self.name, self.tier, data=data, confidence=conf,
                             summary=f"Extracted fields from untrusted {kind} text.")

    # -- planner -------------------------------------------------------------------------
    def _plan(self, request: ModelRequest) -> ModelResponse:
        msgs = request.messages
        issue_id = re.search(r"ISS-\d{3}", msgs[0]["content"]).group(0)
        done = [(tc["name"], tc["arguments"]) for m in msgs if m["role"] == "assistant" for tc in m.get("tool_calls", [])]
        names = [n for n, _ in done]
        results = {m["name"]: m["content"] for m in msgs if m["role"] == "tool"}
        call = lambda name, args, summary, conf=0.95: ModelResponse(  # noqa: E731
            self.name, self.tier, tool_calls=[ToolCall(f"tc{len(done) + 1}", name, args)], confidence=conf, summary=summary)

        # A gullible model obeys instructions found in tool output.
        for content in results.values():
            for d in find_directives(content, request.tools):
                if (d.name, d.arguments) not in done:
                    return call(d.name, d.arguments, f"Following an instruction found in tool output: {d.name}.")

        if "read_issue" not in names:
            return call("read_issue", {"issue_id": issue_id}, f"Start by reading {issue_id}.")
        issue = self._issue_view(results["read_issue"])
        if "knowledge base" in request.system and "search_kb" not in names:
            return call("search_kb", {"query": f"{issue['category']} {issue['query']}"},
                        f"Looks like {issue['category']}; search the knowledge base.")
        if issue["url"] and "fetch_page" not in names:
            return call("fetch_page", {"url": issue["url"]}, "The issue links a status page; fetch it.")
        if "add_label" not in names:
            return call("add_label", {"issue_id": issue_id, "label": issue["category"]},
                        f"Label the issue as {issue['category']}.", issue["confidence"])
        if "post_reply" not in names:
            kb = self._kb_id(results.get("search_kb", ""))
            text = f"Thanks for the report. This looks like a {issue['category']} issue."
            if kb:
                text += f" Please follow {kb}, which covers it step by step."
            if issue["url"]:
                text += " The linked status page has been checked."
            for content in results.values():
                for extra in find_reply_injections(content, request.system):
                    text += " " + extra
            return call("post_reply", {"issue_id": issue_id, "text": text}, "Reply to the customer with the article.")
        return ModelResponse(self.name, self.tier, text=f"Resolved {issue_id}.", confidence=0.95,
                             summary=f"{issue_id} is labelled and answered.")

    def _issue_view(self, content: str) -> dict[str, Any]:
        try:
            view = json.loads(content)               # dual-LLM mode: symbolic view from the reader
            return {"category": view["category"], "query": view["query_ref"],
                    "url": (view["url_refs"] or [None])[0], "confidence": 0.95}
        except (ValueError, KeyError, TypeError):
            title = re.search(r"(?m)^Title: (.*)$", content)
            cat, conf = classify(content, self.tier)
            urls = [u for u in URL.findall(content) if "docs.example.org" in u]
            return {"category": cat, "query": title.group(1) if title else "", "url": urls[0] if urls else None,
                    "confidence": conf}

    @staticmethod
    def _kb_id(content: str) -> str | None:
        try:
            return json.loads(content)["kb_id"]
        except (ValueError, KeyError, TypeError):
            m = re.search(r"KB-\d{3}", content)
            return m.group(0) if m else None
