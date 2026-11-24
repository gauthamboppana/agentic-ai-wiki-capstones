"""Append-only, hash-chained audit log.

Each entry stores the hash of the entry before it, and its own hash covers its content and
that link. Editing, deleting or reordering any line breaks the chain from that point on.
Cutting lines off the end leaves a valid shorter chain, so keep the latest head hash
somewhere the agent cannot write (a ticket, a second store) and pass it to verify().

The agent logs short reasoning summaries, tool mutations, blocked actions and guardrail
flags. It never logs raw chain of thought or secrets.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

GENESIS = "0" * 64


def _digest(entry: dict[str, Any]) -> str:
    body = {k: v for k, v in entry.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass
class Verdict:
    ok: bool
    entries: int
    head: str
    bad_seq: int | None = None
    reason: str = ""


class AuditLog:
    """Write-only API: append() and read access. There is no update or delete."""

    def __init__(self, path: str | Path | None = None, clock: Callable[[], float] = time.time):
        self.path = Path(path) if path else None
        self.clock = clock
        self.entries: list[dict[str, Any]] = []
        if self.path and self.path.exists():
            self.entries = read_entries(self.path)

    @property
    def head(self) -> str:
        return self.entries[-1]["hash"] if self.entries else GENESIS

    def append(self, kind: str, **data: Any) -> dict[str, Any]:
        entry = {"seq": len(self.entries), "ts": round(self.clock(), 3), "kind": kind,
                 "data": data, "prev": self.head}
        entry["hash"] = _digest(entry)
        self.entries.append(entry)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:   # append mode only
                f.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    def verify(self, expected_head: str | None = None) -> Verdict:
        return verify_entries(self.entries, expected_head)


def read_entries(path: str | Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def verify_entries(entries: list[dict[str, Any]], expected_head: str | None = None) -> Verdict:
    prev = GENESIS
    for i, e in enumerate(entries):
        if e.get("seq") != i:
            return Verdict(False, len(entries), prev, i, "sequence gap or reorder")
        if e.get("prev") != prev:
            return Verdict(False, len(entries), prev, i, "broken link to previous entry")
        if _digest(e) != e.get("hash"):
            return Verdict(False, len(entries), prev, i, "content does not match its hash")
        prev = e["hash"]
    if expected_head is not None and prev != expected_head:
        return Verdict(False, len(entries), prev, len(entries), "head does not match the anchored hash (truncated?)")
    return Verdict(True, len(entries), prev)


def verify_file(path: str | Path, expected_head: str | None = None) -> Verdict:
    try:
        entries = read_entries(path)
    except ValueError:
        return Verdict(False, 0, GENESIS, None, "a line is not valid JSON")
    return verify_entries(entries, expected_head)
