"""M2: the provider switch. Every provider streams text chunks for a list of chat messages.

Pick one with environment variables, the pipeline never knows which it got:

    RA_PROVIDER=fake          (default) deterministic local fake, no network
    RA_PROVIDER=openai        any OpenAI-compatible /chat/completions endpoint
        RA_BASE_URL           default http://localhost:11434/v1 (a local runner such as Ollama)
        RA_MODEL              model name on that host (required)
        RA_API_KEY            only if the host needs one; read from the environment, never logged
        RA_RESPONSE_FORMAT    json_schema (default), json_object or none
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.request
from collections.abc import Callable, Iterable, Iterator
from typing import Protocol

from .schema import strict_json_schema

Message = dict[str, str]


class Provider(Protocol):
    name: str

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


# ---------------------------------------------------------------- fake model

ROLE_WORDS = {
    "Dr", "Engineer", "Librarian", "Manager", "Head", "Owner", "Farmer", "Chair", "Union", "Curator",
    "Founder", "Captain", "Lead", "Councillor", "Physiotherapist", "Miller", "Operations", "Ecologist",
    "Volunteer", "The",
}
STARTERS = {
    "If", "When", "Local", "Critics", "Other", "Farmers", "Nurses", "Passengers", "Commuters", "Tomatoes",
    "Independent", "Fuel", "Volunteers", "Organisers", "Members", "Some", "Residents", "Researchers",
    "Drones", "Pupils", "A", "An", "In", "More", "Fishermen", "Passenger",
}
MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
WEEKDAYS = {"Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"}
ORG_WORDS = {
    "Energy", "Trust", "Society", "Library", "Partnership", "Academy", "Forum", "College", "Council",
    "University", "Museum", "Motors", "Lab", "Ltd", "Club", "Network", "Institute", "Association", "Hall",
    "Mill", "Gas", "Store", "Fund", "School", "Board", "Hospital", "Co", "Rail", "Ferries",
}
PLACE_WORDS = {"Island", "End", "Marsh", "Sands", "Hendon"}

NAME_RE = re.compile(r"\b[A-Z][a-z]+(?:(?:\s+of|\s+and(?=\s+Co\b))?\s+[A-Z][a-zA-Z]+)*")
DATE_RE = re.compile(rf"\b(?:(?:{MONTHS})\s+)?(?:19|20)\d\d\b")
ARTICLE_RE = re.compile(r'<article id="(?P<id>[^"]+)" title="(?P<title>[^"]*)">\n(?P<text>.*?)\n</article>', re.S)


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _entity_type(name: str) -> str:
    words = name.split()
    if words[0] == "Port" or words[-1] in ORG_WORDS or len(words) >= 3:
        return "organisation"
    if words[-1] in PLACE_WORDS or len(words) == 1:
        return "place"
    return "person"


def extract(article_id: str, text: str) -> dict:
    """A tiny rule-based 'model': good enough to produce honest, checkable output."""
    entities: dict[str, str] = {}
    for m in NAME_RE.finditer(text):
        words = m.group(0).split()
        while words and words[0] in ROLE_WORDS:
            words = words[1:]
        if not words or (len(words) == 1 and (words[0] in STARTERS or words[0] in MONTHS.split("|"))):
            continue
        name = " ".join(words)
        if name in WEEKDAYS:
            entities.setdefault(name, "date")
            continue
        entities.setdefault(name, _entity_type(name))
    for m in DATE_RE.finditer(text):
        entities.setdefault(m.group(0), "date")
    sents = _sentences(text)
    return {
        "article_id": article_id,
        "key_findings": [s[:200] for s in sents[:3]],
        "entities": [{"name": n, "type": t} for n, t in list(entities.items())[:20]],
        "confidence": round(min(0.95, 0.5 + 0.08 * len(sents)), 2),
    }


# Which articles misbehave, and how, on each attempt. Attempt 1 is the first reply.
DEFAULT_FAULTS: dict[str, list[str]] = {
    "a04": ["truncate"],
    "a09": ["prose"],
    "a13": ["bad_confidence"],
    "a17": ["truncate", "missing_field"],
}


def apply_fault(fault: str, good: dict) -> str:
    text = json.dumps(good)
    if fault == "truncate":
        return text[: len(text) * 3 // 5]
    if fault == "prose":
        return "Here is the JSON you asked for:\n" + text
    if fault == "fence":
        return "```json\n" + text + "\n```"
    if fault == "bad_confidence":
        return json.dumps({**good, "confidence": 1.4})
    if fault == "missing_field":
        return json.dumps({k: v for k, v in good.items() if k != "entities"})
    if fault == "refuse":
        return "I'm sorry, but I can't help with that request."
    if fault == "empty":
        return ""
    raise ValueError(f"unknown fault {fault!r}")


class FakeModel:
    """Deterministic and offline. Reads the article out of the prompt, streams JSON in small chunks.

    `faults` maps article id to a list of fault names, one per attempt, so the repair path runs.
    The attempt number is worked out from the conversation (one assistant turn per earlier try),
    so the fake keeps no hidden state and every run is repeatable.
    """

    name = "fake"

    def __init__(self, faults: dict[str, list[str]] | None = None, first_token_delay: float = 0.02,
                 chunk_size: int = 24, chunk_delay: float = 0.0):
        self.faults = DEFAULT_FAULTS if faults is None else faults
        self.first_token_delay = first_token_delay
        self.chunk_size = chunk_size
        self.chunk_delay = chunk_delay

    def reply(self, messages: list[Message]) -> str:
        m = ARTICLE_RE.search(next(x["content"] for x in messages if x["role"] == "user"))
        if not m:
            return apply_fault("refuse", {})
        good = extract(m["id"], m["text"])
        attempt = sum(1 for x in messages if x["role"] == "assistant")
        plan = self.faults.get(m["id"], [])
        return apply_fault(plan[attempt], good) if attempt < len(plan) else json.dumps(good)

    def stream(self, messages: list[Message]) -> Iterator[str]:
        text = self.reply(messages)
        time.sleep(self.first_token_delay)
        for i in range(0, len(text), self.chunk_size):
            if i and self.chunk_delay:
                time.sleep(self.chunk_delay)
            yield text[i : i + self.chunk_size]


class ScriptedModel:
    """Replies with fixed strings in order, one per call. Used to feed broken cases in tests."""

    name = "scripted"

    def __init__(self, replies: list[str], chunk_size: int = 16):
        self.replies = list(replies)
        self.calls: list[list[Message]] = []
        self.chunk_size = chunk_size

    def stream(self, messages: list[Message]) -> Iterator[str]:
        self.calls.append([dict(m) for m in messages])
        text = self.replies[min(len(self.calls), len(self.replies)) - 1]
        for i in range(0, len(text), self.chunk_size):
            yield text[i : i + self.chunk_size]


# ---------------------------------------------------- OpenAI-compatible HTTP

Transport = Callable[[str, dict, dict, float], Iterable[str]]


def urllib_transport(url: str, headers: dict, body: dict, timeout: float) -> Iterator[str]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 (URL comes from config)
        for raw in resp:
            yield raw.decode("utf-8", "replace")


class OpenAICompatible:
    """Streams from any host that speaks the OpenAI chat completions format.

    That covers hosted APIs and local runners (Ollama, vLLM, LM Studio, llama.cpp server).
    The transport is injectable so tests can feed recorded server-sent event lines with no network.
    """

    name = "openai-compatible"

    def __init__(self, base_url: str, model: str, api_key: str | None = None, response_format: str = "json_schema",
                 timeout: float = 60.0, transport: Transport = urllib_transport):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._api_key = api_key
        self.response_format = response_format
        self.timeout = timeout
        self.transport = transport

    def __repr__(self) -> str:  # never show the key
        return f"OpenAICompatible(base_url={self.base_url!r}, model={self.model!r}, key={'set' if self._api_key else 'none'})"

    def request_body(self, messages: list[Message]) -> dict:
        body: dict = {"model": self.model, "messages": messages, "stream": True}
        if self.response_format == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": strict_json_schema()}
        elif self.response_format == "json_object":
            body["response_format"] = {"type": "json_object"}
        return body

    def stream(self, messages: list[Message]) -> Iterator[str]:
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        lines = self.transport(f"{self.base_url}/chat/completions", headers, self.request_body(messages), self.timeout)
        for line in lines:
            line = line.strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            choice = (json.loads(data).get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            piece = delta.get("content") or delta.get("refusal")
            if piece:
                yield piece


def get_provider(env: dict[str, str] | None = None) -> Provider:
    env = os.environ if env is None else env
    kind = env.get("RA_PROVIDER", "fake").lower()
    if kind == "fake":
        return FakeModel()
    if kind in ("openai", "openai-compatible", "ollama"):
        model = env.get("RA_MODEL")
        if not model:
            raise ValueError("RA_MODEL must name the model to use on the OpenAI-compatible host")
        return OpenAICompatible(
            base_url=env.get("RA_BASE_URL", "http://localhost:11434/v1"),
            model=model,
            api_key=env.get("RA_API_KEY") or None,
            response_format=env.get("RA_RESPONSE_FORMAT", "json_schema"),
        )
    raise ValueError(f"unknown RA_PROVIDER {kind!r}: use fake or openai")
