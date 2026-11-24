"""Episodic memory: one row per past fix, retrieved by word overlap with the new issue."""

from __future__ import annotations

import json
import re
import sqlite3
import time
from pathlib import Path

STOP = {"the", "and", "for", "not", "are", "with", "that", "this", "but", "should", "like", "looks", "into", "from"}


def words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 2 and w not in STOP}


class EpisodicMemory:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS episodes (issue_id TEXT, text TEXT, files TEXT, lesson TEXT, ts REAL)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def add(self, issue_id: str, text: str, files: list[str], lesson: str) -> None:
        with self._db() as db:
            db.execute("INSERT INTO episodes VALUES (?, ?, ?, ?, ?)", (issue_id, text, json.dumps(files), lesson, time.time()))

    def search(self, query: str, k: int = 2, min_overlap: int = 2) -> list[dict]:
        """The k past fixes sharing the most words with the query (at least `min_overlap`)."""
        q = words(query)
        with self._db() as db:
            rows = db.execute("SELECT issue_id, text, files, lesson FROM episodes").fetchall()
        scored = []
        for issue_id, text, files, lesson in rows:
            overlap = len(q & words(text + " " + lesson))
            if overlap >= min_overlap:
                scored.append((overlap, {"issue_id": issue_id, "files": json.loads(files), "lesson": lesson,
                                         "overlap": overlap}))
        scored.sort(key=lambda s: -s[0])
        return [m for _, m in scored[:k]]

    def count(self) -> int:
        with self._db() as db:
            return db.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
