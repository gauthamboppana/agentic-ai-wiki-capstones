"""SQLite checkpoints: one row per node run, so any run can be resumed from its last row."""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Checkpoint:
    run_id: str
    step: int
    node: str
    next_node: str
    state: dict
    status: str


class SqliteCheckpointer:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS checkpoints (run_id TEXT, step INTEGER, node TEXT, next_node TEXT,"
                " state TEXT, status TEXT, ts REAL, PRIMARY KEY (run_id, step, status))"
            )

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def save(self, run_id: str, step: int, node: str, next_node: str, state: dict, status: str) -> None:
        with self._db() as db:
            db.execute(
                "INSERT OR REPLACE INTO checkpoints VALUES (?, ?, ?, ?, ?, ?, ?)",
                (run_id, step, node, next_node, json.dumps(state), status, time.time()),
            )

    def latest(self, run_id: str) -> Checkpoint | None:
        with self._db() as db:
            row = db.execute(
                "SELECT run_id, step, node, next_node, state, status FROM checkpoints WHERE run_id = ?"
                " ORDER BY rowid DESC LIMIT 1",
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return Checkpoint(row[0], row[1], row[2], row[3], json.loads(row[4]), row[5])

    def history(self, run_id: str) -> list[tuple[int, str, str, str]]:
        with self._db() as db:
            return db.execute(
                "SELECT step, node, next_node, status FROM checkpoints WHERE run_id = ? ORDER BY rowid",
                (run_id,),
            ).fetchall()

    def runs(self) -> list[tuple[str, str, str]]:
        """(run_id, issue_id, last status) for every run, oldest first."""
        with self._db() as db:
            rows = db.execute("SELECT run_id, state, status FROM checkpoints ORDER BY rowid").fetchall()
        last: dict[str, tuple[str, str]] = {}
        for run_id, state, status in rows:
            last[run_id] = (json.loads(state).get("issue_id", "?"), status)
        return [(r, i, s) for r, (i, s) in last.items()]
