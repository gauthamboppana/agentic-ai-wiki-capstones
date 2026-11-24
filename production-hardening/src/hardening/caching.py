"""Exact and semantic response caches, with a deterministic local embedding.

Prompt-prefix caching happens at the provider: the agent keeps its system prompt and tool
list byte-identical and first in every request, asks for caching, and the provider reports
`cached_input_tokens`, which are billed at the cached rate (see tokens.py). The two caches
here sit in front of the model and skip the call altogether.
"""
from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any

DIMS = 256


def embed(text: str) -> list[float]:
    """Hashed bag of words and word pairs, L2-normalised. Stable across runs and machines."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    feats = words + [a + "_" + b for a, b in zip(words, words[1:])]
    vec = [0.0] * DIMS
    for f in feats:
        h = int.from_bytes(hashlib.blake2b(f.encode(), digest_size=4).digest(), "big")
        vec[h % DIMS] += 1.0 if (h >> 31) & 1 else -1.0
    norm = math.sqrt(sum(v * v for v in vec)) or 1.0
    return [v / norm for v in vec]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0


class ExactCache:
    def __init__(self) -> None:
        self._store: dict[str, Any] = {}
        self.stats = CacheStats()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()

    def get(self, key: str) -> Any | None:
        if key in self._store:
            self.stats.hits += 1
            return self._store[key]
        self.stats.misses += 1
        return None

    def put(self, key: str, value: Any) -> None:
        self._store[key] = value


@dataclass
class SemanticCache:
    """Returns a stored value when a new input is close enough to an old one, within a namespace."""
    threshold: float = 0.80
    entries: list[tuple[str, list[float], Any]] = field(default_factory=list)
    stats: CacheStats = field(default_factory=CacheStats)

    def get(self, namespace: str, text: str) -> tuple[Any, float] | None:
        q = embed(text)
        best, score = None, -1.0
        for ns, vec, value in self.entries:
            if ns == namespace:
                s = cosine(q, vec)
                if s > score:
                    best, score = value, s
        if best is not None and score >= self.threshold:
            self.stats.hits += 1
            return best, score
        self.stats.misses += 1
        return None

    def put(self, namespace: str, text: str, value: Any) -> None:
        self.entries.append((namespace, embed(text), value))
