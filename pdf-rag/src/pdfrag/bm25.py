"""BM25 keyword search over chunk text. Exact terms matter in manuals: fault codes, part numbers, addresses."""

from __future__ import annotations

import math
from collections import Counter

from .chunking import Chunk
from .text import terms


class BM25:
    def __init__(self, chunks: list[Chunk], k1: float = 1.4, b: float = 0.75):
        self.chunks, self.k1, self.b = chunks, k1, b
        self.docs = [Counter(terms(c.search_text)) for c in chunks]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(len(self.lengths), 1)
        df = Counter(t for d in self.docs for t in d)
        n = len(chunks)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.max_idf = math.log(1 + (n + 0.5) / 0.5)  # what an unseen term would score

    def score(self, query_terms: list[str], i: int) -> float:
        doc, length, s = self.docs[i], self.lengths[i], 0.0
        for t in query_terms:
            f = doc.get(t, 0)
            if f:
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg))
        return s

    def search(self, query: str, k: int = 10, keep=lambda c: True) -> list[tuple[Chunk, float]]:
        q = terms(query)
        scored = [(c, self.score(q, i)) for i, c in enumerate(self.chunks) if keep(c)]
        scored = [cs for cs in scored if cs[1] > 0]
        return sorted(scored, key=lambda cs: (-cs[1], cs[0].chunk_id))[:k]
