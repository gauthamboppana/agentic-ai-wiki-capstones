"""The approval gate: diffs, decisions, policies and the ledger that every write is audited against."""

from __future__ import annotations

import difflib
import hashlib
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def unified_diff(path: str, before: str, after: str) -> str:
    return "".join(difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}"))


@dataclass
class Decision:
    action: str  # "approve", "edit" or "reject"
    by: str = "policy"
    reason: str = ""
    edited: dict[str, str] = field(default_factory=dict)  # path -> replacement content, for "edit"

    def to_dict(self) -> dict:
        return {"action": self.action, "by": self.by, "reason": self.reason, "edited": self.edited}

    @classmethod
    def from_dict(cls, d: dict) -> "Decision":
        return cls(d["action"], d.get("by", "policy"), d.get("reason", ""), d.get("edited", {}))


class ApprovalPolicy(Protocol):
    def decide(self, proposal: dict) -> Decision | None:
        """Return a decision, or None to pause the run until a human decides."""


class AutoApprove:
    """For tests and demos: approves everything, and says so in the ledger."""

    def decide(self, proposal: dict) -> Decision:
        return Decision("approve", by="auto-approve policy")


class RejectAll:
    def decide(self, proposal: dict) -> Decision:
        return Decision("reject", by="reject-all policy", reason="not allowed in this run")


class Scripted:
    """Returns pre-set decisions in order, then approves. Useful for testing edit and reject."""

    def __init__(self, decisions: list[Decision]):
        self.decisions = list(decisions)

    def decide(self, proposal: dict) -> Decision:
        return self.decisions.pop(0) if self.decisions else Decision("approve", by="scripted policy")


class Defer:
    """Never decides: the run stops with `awaiting_approval` and is resumed with a decision later."""

    def decide(self, proposal: dict) -> None:
        return None


class CliPrompt:
    """Shows the diff to a human and waits for approve, edit or reject."""

    def __init__(self, ask: Callable[[str], str] = input, show: Callable[[str], None] = print):
        self.ask, self.show = ask, show

    def decide(self, proposal: dict) -> Decision:
        self.show(f"\nProposed change for {proposal['issue_id']} (attempt {proposal['attempt']}): {proposal['summary']}")
        self.show(proposal["diff"])
        while True:
            answer = self.ask("[a]pprove, [e]dit or [r]eject? ").strip().lower()
            if answer in ("a", "approve"):
                return Decision("approve", by="human (cli)")
            if answer in ("r", "reject"):
                return Decision("reject", by="human (cli)", reason=self.ask("Reason: ").strip())
            if answer in ("e", "edit"):
                edited = {}
                for path, content in proposal["after"].items():
                    self.show(f"Type the new content for {path}, then a line with a single '.' (empty keeps it):")
                    lines = []
                    while (line := self.ask("")) != ".":
                        lines.append(line)
                    edited[path] = "\n".join(lines) + "\n" if lines else content
                return Decision("edit", by="human (cli)", edited=edited)


class Ledger:
    """Approval records and the write log, in the same SQLite file as the checkpoints."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._db() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS approvals (id TEXT PRIMARY KEY, run_id TEXT, issue_id TEXT, path TEXT,"
                " action TEXT, decided_by TEXT, reason TEXT, sha_before TEXT, sha_after TEXT, diff TEXT,"
                " used INTEGER DEFAULT 0, ts REAL)"
            )
            db.execute(
                "CREATE TABLE IF NOT EXISTS writes (id INTEGER PRIMARY KEY AUTOINCREMENT, approval_id TEXT,"
                " path TEXT, sha256 TEXT, ts REAL)"
            )

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path, timeout=10)

    def record(self, run_id: str, issue_id: str, path: str, decision: Decision, before: str, after: str) -> str:
        approval_id = uuid.uuid4().hex[:12]
        with self._db() as db:
            db.execute(
                "INSERT INTO approvals (id, run_id, issue_id, path, action, decided_by, reason, sha_before,"
                " sha_after, diff, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (approval_id, run_id, issue_id, path, decision.action, decision.by, decision.reason,
                 sha256(before), sha256(after), unified_diff(path, before, after), time.time()),
            )
        return approval_id

    def authorise_write(self, approval_id: str, path: str, content: str) -> str | None:
        """Called by the filesystem server before it writes. Returns an error message, or None if allowed."""
        with self._db() as db:
            row = db.execute("SELECT path, action, sha_after, used FROM approvals WHERE id = ?",
                             (approval_id,)).fetchone()
            if row is None:
                return f"no approval record {approval_id!r}"
            if row[1] not in ("approve", "edit"):
                return f"approval {approval_id} was a {row[1]}, not an approval"
            if row[0] != path:
                return f"approval {approval_id} is for {row[0]}, not {path}"
            if row[2] != sha256(content):
                return "content differs from what was approved"
            if row[3]:
                return f"approval {approval_id} was already used"
            db.execute("UPDATE approvals SET used = 1 WHERE id = ?", (approval_id,))
            db.execute("INSERT INTO writes (approval_id, path, sha256, ts) VALUES (?, ?, ?, ?)",
                       (approval_id, path, sha256(content), time.time()))
        return None

    def approvals(self) -> list[dict]:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT * FROM approvals ORDER BY ts")]

    def writes(self) -> list[dict]:
        with self._db() as db:
            db.row_factory = sqlite3.Row
            return [dict(r) for r in db.execute("SELECT * FROM writes ORDER BY id")]


def audit(ledger: Ledger, repo: Path, seed: Path) -> dict:
    """Check every write against an approval, and every changed file against the last approved write.

    The second check catches writes that bypassed the tool server altogether.
    """
    approvals = {a["id"]: a for a in ledger.approvals()}
    problems = []
    last_written: dict[str, str] = {}
    for w in ledger.writes():
        a = approvals.get(w["approval_id"])
        if a is None or a["action"] not in ("approve", "edit") or a["path"] != w["path"] or a["sha_after"] != w["sha256"]:
            problems.append(f"write to {w['path']} has no matching approval")
        last_written[w["path"]] = w["sha256"]
    for f in sorted(p for p in repo.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        rel = f.relative_to(repo).as_posix()
        now = sha256(f.read_text(encoding="utf-8"))
        original = seed / rel
        if original.exists() and sha256(original.read_text(encoding="utf-8")) == now:
            continue
        if last_written.get(rel) != now:
            problems.append(f"{rel} changed without an approved write")
    rejected = [a for a in approvals.values() if a["action"] == "reject"]
    return {"writes": len(ledger.writes()), "approvals": len(approvals), "rejected": len(rejected),
            "unauthorised": len(problems), "problems": problems}
