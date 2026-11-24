"""A vector store interface with metadata filters, and two local implementations.

`where` is a dict of metadata equality tests, for example {"doc_id": "h200"} or {"kind": "table"}.
A list value means "any of": {"page": [3, 4]}. Qdrant, pgvector or Chroma can implement the same three
methods; see `QdrantStore` and the README for the mapping.
"""

from __future__ import annotations

import json
import sqlite3
from array import array
from typing import Protocol

from .chunking import Chunk
from .embed import cosine

FILTER_FIELDS = ("doc_id", "page", "kind", "section")


def matches(chunk: Chunk, where: dict | None) -> bool:
    if not where:
        return True
    for key, want in where.items():
        have = chunk.meta[key]
        if isinstance(want, (list, tuple, set)):
            if have not in want:
                return False
        elif have != want:
            return False
    return True


class VectorStore(Protocol):
    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(self, vector: list[float], k: int = 10, where: dict | None = None) -> list[tuple[Chunk, float]]: ...

    def chunks(self, where: dict | None = None) -> list[Chunk]: ...


class InMemoryStore:
    def __init__(self) -> None:
        self._rows: list[tuple[Chunk, list[float]]] = []

    def add(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        self._rows.extend(zip(chunks, vectors))

    def search(self, vector, k=10, where=None):
        scored = [(c, cosine(vector, v)) for c, v in self._rows if matches(c, where)]
        return sorted(scored, key=lambda cs: (-cs[1], cs[0].chunk_id))[:k]

    def chunks(self, where=None):
        return [c for c, _ in self._rows if matches(c, where)]


class SQLiteStore:
    """Chunks, metadata and vectors in one SQLite file (or ":memory:"). Filters run in SQL, scoring in Python."""

    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS chunks (chunk_id TEXT PRIMARY KEY, doc_id TEXT, doc_title TEXT,"
            " page INTEGER, section TEXT, kind TEXT, text TEXT, vector BLOB)"
        )
        self.db.execute("CREATE INDEX IF NOT EXISTS by_doc_page ON chunks (doc_id, page)")

    def add(self, chunks, vectors):
        self.db.executemany(
            "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(c.chunk_id, c.doc_id, c.doc_title, c.page, c.section, c.kind, c.text, array("d", v).tobytes()) for c, v in zip(chunks, vectors)],
        )
        self.db.commit()

    def _select(self, where: dict | None) -> list[tuple]:
        clauses, params = [], []
        for key, want in (where or {}).items():
            if key not in FILTER_FIELDS:
                raise ValueError(f"cannot filter on {key!r}")
            if isinstance(want, (list, tuple, set)):
                want = list(want)
                clauses.append(f"{key} IN ({', '.join('?' * len(want))})")
                params += want
            else:
                clauses.append(f"{key} = ?")
                params.append(want)
        sql = "SELECT chunk_id, doc_id, doc_title, page, section, kind, text, vector FROM chunks"
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        return self.db.execute(sql + " ORDER BY rowid", params).fetchall()

    def search(self, vector, k=10, where=None):
        scored = []
        for *fields, blob in self._select(where):
            scored.append((Chunk(*fields), cosine(vector, array("d", blob).tolist())))
        return sorted(scored, key=lambda cs: (-cs[1], cs[0].chunk_id))[:k]

    def chunks(self, where=None):
        return [Chunk(*fields) for *fields, _ in self._select(where)]


class QdrantStore:  # pragma: no cover - needs the optional extra and a running Qdrant (or its local mode)
    """The same interface on Qdrant. `uv sync --extra qdrant`, then QdrantStore(location=":memory:")."""

    def __init__(self, dim: int, collection: str = "chunks", **client_args) -> None:
        from qdrant_client import QdrantClient, models

        self.models, self.collection = models, collection
        self.client = QdrantClient(**client_args)
        self.client.recreate_collection(collection, vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE))

    def _filter(self, where):
        m = self.models
        if not where:
            return None
        must = [
            m.FieldCondition(key=k, match=m.MatchAny(any=list(v)) if isinstance(v, (list, tuple, set)) else m.MatchValue(value=v))
            for k, v in where.items()
        ]
        return m.Filter(must=must)

    def add(self, chunks, vectors):
        points = [self.models.PointStruct(id=i, vector=v, payload={**c.__dict__}) for i, (c, v) in enumerate(zip(chunks, vectors))]
        self.client.upsert(self.collection, points)

    def search(self, vector, k=10, where=None):
        hits = self.client.search(self.collection, query_vector=vector, limit=k, query_filter=self._filter(where))
        return [(Chunk(**h.payload), h.score) for h in hits]

    def chunks(self, where=None):
        points, _ = self.client.scroll(self.collection, scroll_filter=self._filter(where), limit=10_000)
        return [Chunk(**p.payload) for p in points]


def dump(chunks: list[Chunk]) -> str:
    return json.dumps([c.__dict__ for c in chunks], indent=1)
