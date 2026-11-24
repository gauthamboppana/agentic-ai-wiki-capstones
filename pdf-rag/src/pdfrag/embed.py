"""Dense embeddings behind one small interface.

`HashingEmbedder` is deterministic, offline and free: it hashes character n-grams of each word into a
fixed-size vector. It catches near matches that exact keywords miss ("restart" and "restarts", "pre-charge"
and "precharge"). Swap in a real model with `SentenceTransformerEmbedder` (`uv sync --extra models`).
"""

from __future__ import annotations

import hashlib
import math
from typing import Protocol

from .text import STOPWORDS, tokens


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> list[list[float]]: ...


def normalise(v: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def cosine(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))  # vectors are already unit length


class HashingEmbedder:
    def __init__(self, dim: int = 1024, n_min: int = 3, n_max: int = 5):
        self.dim, self.n_min, self.n_max = dim, n_min, n_max

    def _bucket(self, gram: str) -> tuple[int, float]:
        h = int.from_bytes(hashlib.blake2b(gram.encode(), digest_size=8).digest(), "big")
        return h % self.dim, 1.0 if (h >> 63) & 1 else -1.0  # signed hashing reduces collision bias

    def embed_one(self, text: str) -> list[float]:
        v = [0.0] * self.dim
        for word in tokens(text):
            if word in STOPWORDS:
                continue
            padded = f"<{word}>"
            for n in range(self.n_min, self.n_max + 1):
                for i in range(len(padded) - n + 1):
                    idx, sign = self._bucket(padded[i : i + n])
                    v[idx] += sign
        return normalise(v)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(t) for t in texts]


class SentenceTransformerEmbedder:
    """A real embedding model, loaded locally. Needs `uv sync --extra models` and a one-off download."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [list(map(float, v)) for v in self.model.encode(texts, normalize_embeddings=True)]
