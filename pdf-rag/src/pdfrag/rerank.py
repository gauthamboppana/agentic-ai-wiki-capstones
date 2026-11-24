"""Rerankers score (question, chunk) pairs directly, after fusion has narrowed the field.

`LexicalReranker` is a deterministic stand-in for a cross-encoder: the share of the question's content
terms (weighted by rarity) that the chunk contains, plus a small bonus for matching adjacent pairs.
Its score is in [0, 1.1], which also makes it a usable "is the answer here at all" signal.
`CrossEncoderReranker` is the real thing (`uv sync --extra models`); recalibrate the abstain threshold for it.
"""

from __future__ import annotations

import math
from typing import Protocol

from .chunking import Chunk
from .text import terms


class Reranker(Protocol):
    def rerank(self, query: str, chunks: list[Chunk], top_n: int = 5) -> list[tuple[Chunk, float]]: ...


class LexicalReranker:
    def __init__(self, idf: dict[str, float], unseen_idf: float, ignore: set[str] | None = None):
        self.idf, self.unseen_idf, self.ignore = idf, unseen_idf, ignore or set()

    def score(self, query: str, chunk: Chunk) -> float:
        q = [t for t in dict.fromkeys(terms(query)) if t not in self.ignore]
        if not q:
            return 0.0
        body = terms(chunk.section + "\n" + chunk.text)  # not the title: naming the product is not evidence
        have = set(body)
        weight = {t: self.idf.get(t, self.unseen_idf) for t in q}
        coverage = sum(weight[t] for t in q if t in have) / sum(weight.values())
        pairs = set(zip(body, body[1:]))
        q_pairs = list(zip(q, q[1:]))
        bonus = 0.1 * sum(p in pairs for p in q_pairs) / len(q_pairs) if q_pairs else 0.0
        return coverage + bonus

    def rerank(self, query, chunks, top_n=5):
        scored = [(c, self.score(query, c)) for c in chunks]
        return sorted(scored, key=lambda cs: -cs[1])[:top_n]  # stable: ties keep fusion order


class CrossEncoderReranker:  # pragma: no cover - needs the optional extra and a model download
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_n=5):
        logits = self.model.predict([(query, c.search_text) for c in chunks])
        scored = [(c, 1 / (1 + math.exp(-float(s)))) for c, s in zip(chunks, logits)]
        return sorted(scored, key=lambda cs: -cs[1])[:top_n]
