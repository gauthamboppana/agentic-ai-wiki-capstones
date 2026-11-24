"""Chat models behind one method, `complete(system, user) -> str`.

The default is `FakeModel`: deterministic, offline and free, so tests repeat exactly. It reads the same
prompt a real model would get and answers extractively, quoting the best sentence and its citation.
`OpenAICompatibleModel` talks to any OpenAI-compatible endpoint (a hosted API, or a local open-weights
model served by Ollama, vLLM or LM Studio). It is switched on by environment variables only.
"""

from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Protocol

from .text import sentences, terms

NOT_FOUND = "Not in the documents."

SYSTEM_PROMPT = (
    "You answer questions about technical manuals using only the context blocks provided.\n"
    "Rules:\n"
    "1. Every sentence must end with the citation of the block it comes from, copied exactly, e.g. [h200 p.4].\n"
    "2. Do not add facts, numbers or advice that are not in the cited block.\n"
    f"3. If the context does not answer the question, reply with exactly: {NOT_FOUND}\n"
    "4. Keep it to one to three sentences."
)

BLOCK_RE = re.compile(r"^### (\[[a-z0-9-]+ p\.\d+\])([^\n]*)\n(.*?)(?=^### |\Z)", re.M | re.S)


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


def build_prompt(question: str, contexts) -> str:
    blocks = "\n\n".join(f"### {c.cite} {c.section}\n{c.text}" for c in contexts)
    return f"Question: {question}\n\nContext:\n\n{blocks}\n"


class FakeModel:
    """Extractive stand-in for a model: quotes the sentence that covers the most question terms."""

    def __init__(self, min_coverage: float = 0.34):
        self.min_coverage = min_coverage
        self.calls = 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        question = user.split("\n", 1)[0].removeprefix("Question:").strip()
        q = set(terms(question))
        if not q:
            return NOT_FOUND
        candidates = []  # (coverage, block order, cite, sentence)
        for order, (cite, section, body) in enumerate(BLOCK_RE.findall(user)):
            heading = set(terms(section))  # the heading tells the reader what a table row is about
            for s in sentences(body):
                if s.startswith("Table "):
                    continue  # a caption names the table, it does not answer anything
                coverage = len(q & (set(terms(s)) | heading)) / len(q)
                candidates.append((coverage, -order, cite, s))
        if not candidates:
            return NOT_FOUND
        candidates.sort(reverse=True)
        best = candidates[0]
        if best[0] < self.min_coverage:
            return NOT_FOUND
        picked = [best]
        matched = q & set(terms(best[3]))
        for c in candidates[1:]:  # one supporting sentence from another page, on the same point
            if c[2] != best[2] and c[0] >= 0.6 * best[0] and len(matched & set(terms(c[3]))) >= 2:
                picked.append(c)
                break
        return " ".join(f"{s.rstrip('.')}. {cite}" for _, _, cite, s in picked)


class ScriptedModel:
    """Replays canned replies in order. Useful for testing retry and judge parsing paths."""

    def __init__(self, replies: list[str]):
        self.replies, self.calls = list(replies), 0

    def complete(self, system: str, user: str) -> str:
        self.calls += 1
        return self.replies.pop(0) if self.replies else NOT_FOUND


class OpenAICompatibleModel:
    """POST {base_url}/chat/completions. The key comes from the environment and is never printed."""

    def __init__(self, base_url: str, model: str, api_key: str | None = None, timeout: float = 60):
        self.base_url, self.model, self._key, self.timeout = base_url.rstrip("/"), model, api_key, timeout

    def __repr__(self) -> str:
        return f"OpenAICompatibleModel(base_url={self.base_url!r}, model={self.model!r})"

    def complete(self, system: str, user: str) -> str:
        body = {"model": self.model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        headers = {"Content-Type": "application/json"}
        if self._key:
            headers["Authorization"] = f"Bearer {self._key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.load(resp)
        return data["choices"][0]["message"]["content"].strip()


def get_model() -> ChatModel:
    """PDFRAG_PROVIDER=openai-compatible plus PDFRAG_BASE_URL and PDFRAG_MODEL (and PDFRAG_API_KEY if needed)."""
    provider = os.environ.get("PDFRAG_PROVIDER", "fake")
    if provider == "fake":
        return FakeModel()
    if provider == "openai-compatible":
        return OpenAICompatibleModel(
            base_url=os.environ.get("PDFRAG_BASE_URL", "http://localhost:11434/v1"),
            model=os.environ["PDFRAG_MODEL"],  # whatever name your server lists
            api_key=os.environ.get("PDFRAG_API_KEY"),
        )
    raise ValueError(f"unknown PDFRAG_PROVIDER {provider!r}")
