"""Reciprocal rank fusion: merge ranked lists by rank, so BM25 scores and cosine scores never need rescaling."""

from __future__ import annotations

from .chunking import Chunk


def rrf(rankings: list[list[Chunk]], k: int = 60) -> list[tuple[Chunk, float]]:
    scores: dict[str, float] = {}
    by_id: dict[str, Chunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            by_id[chunk.chunk_id] = chunk
    return sorted(((by_id[i], s) for i, s in scores.items()), key=lambda cs: (-cs[1], cs[0].chunk_id))
