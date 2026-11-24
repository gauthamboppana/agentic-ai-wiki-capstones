"""A minimal MCP-compatible tool server over stdio, written with the standard library only.

It speaks JSON-RPC 2.0, one message per line, and implements the three methods an agent needs:
`initialize`, `tools/list` and `tools/call`. Two servers share this file:

    python -m reviewer.mcp_server workspace --root REPO --db STATE.db   # files and tests
    python -m reviewer.mcp_server tracker --db STATE.db                 # the issue tracker

The official MCP Python SDK (`mcp` package, FastMCP) can replace this file: see the README.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sqlite3
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from reviewer.approval import Ledger

PROTOCOL_VERSION = "2025-06-18"
TEST_LINE = re.compile(r"^(\w+) \(([\w.]+)\)\s*\.\.\. (ok|FAIL|ERROR|skipped.*|expected failure)$")


class Tool:
    def __init__(self, name: str, description: str, schema: dict, fn: Callable[..., Any]):
        self.name, self.description, self.schema, self.fn = name, description, schema, fn

    def spec(self) -> dict:
        return {"name": self.name, "description": self.description, "inputSchema": self.schema}


def obj(props: dict, required: list[str] | None = None) -> dict:
    return {"type": "object", "properties": props, "required": required or []}


STR = {"type": "string"}


# ----- workspace server: files and tests -----

class Workspace:
    def __init__(self, root: Path, ledger: Ledger):
        self.root = root.resolve()
        self.ledger = ledger

    def _path(self, rel: str) -> Path:
        p = (self.root / rel).resolve()
        if p != self.root and self.root not in p.parents:
            raise ValueError(f"path {rel!r} is outside the repository")
        return p

    def fs_list(self) -> list[str]:
        return sorted(p.relative_to(self.root).as_posix() for p in self.root.rglob("*")
                      if p.is_file() and "__pycache__" not in p.parts)

    def fs_read(self, path: str) -> str:
        return self._path(path).read_text(encoding="utf-8")

    def fs_write(self, path: str, content: str, approval_id: str) -> str:
        problem = self.ledger.authorise_write(approval_id, path, content)
        if problem:
            raise PermissionError(f"write refused: {problem}")
        self._path(path).write_text(content, encoding="utf-8")
        return f"wrote {len(content)} characters to {path}"

    def tests_run(self, tests: list[str] | None = None) -> dict:
        cmd = [sys.executable, "-m", "unittest"] + (["-v", *tests] if tests else ["discover", "-v", "-s", "tests", "-t", "."])
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        try:
            out = subprocess.run(cmd, cwd=self.root, env=env, capture_output=True, text=True, timeout=60)
        except subprocess.TimeoutExpired:
            return {"passed": [], "failed": tests or [], "ok": False, "output": "timed out"}
        passed, failed = [], []
        for line in out.stderr.splitlines():
            m = TEST_LINE.match(line.strip())
            if m:  # Python 3.11 prints (module.Class), 3.12+ prints (module.Class.test)
                test_id = m.group(2) if m.group(2).endswith("." + m.group(1)) else f"{m.group(2)}.{m.group(1)}"
                (passed if m.group(3) == "ok" else failed).append(test_id)
        if out.returncode != 0 and not failed:
            failed = tests or ["<suite>"]
        return {"passed": passed, "failed": failed, "ok": out.returncode == 0, "output": out.stderr[-1500:]}

    def tools(self) -> list[Tool]:
        return [
            Tool("fs_list", "List every file in the repository.", obj({}), self.fs_list),
            Tool("fs_read", "Read a text file from the repository.", obj({"path": STR}, ["path"]), self.fs_read),
            Tool("fs_write", "Write a file. Needs the id of an approval record whose content hash matches.",
                 obj({"path": STR, "content": STR, "approval_id": STR}, ["path", "content", "approval_id"]),
                 self.fs_write),
            Tool("tests_run", "Run unit tests by id, or the whole suite if none are given.",
                 obj({"tests": {"type": "array", "items": STR}}), self.tests_run),
        ]


# ----- tracker server: the issue tracker -----

class Tracker:
    def __init__(self, db_path: str | Path):
        self.db_path = str(db_path)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS issues (id TEXT PRIMARY KEY, data TEXT, status TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS comments (issue_id TEXT, text TEXT)")

    def _db(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, timeout=10)

    def seed(self, issues: list[dict]) -> None:
        with self._db() as db:
            for issue in issues:
                db.execute("INSERT OR IGNORE INTO issues VALUES (?, ?, 'open')", (issue["id"], json.dumps(issue)))

    def issues_list(self, status: str = "open") -> list[dict]:
        with self._db() as db:
            rows = db.execute("SELECT data FROM issues WHERE status = ? OR ? = 'all' ORDER BY id", (status, status))
            return [{"id": d["id"], "title": d["title"]} for d in (json.loads(r[0]) for r in rows)]

    def issues_get(self, id: str) -> dict:
        with self._db() as db:
            row = db.execute("SELECT data, status FROM issues WHERE id = ?", (id,)).fetchone()
            if row is None:
                raise KeyError(f"no issue {id}")
            comments = [r[0] for r in db.execute("SELECT text FROM comments WHERE issue_id = ?", (id,))]
        return {**json.loads(row[0]), "status": row[1], "comments": comments}

    def issues_comment(self, id: str, text: str) -> str:
        with self._db() as db:
            db.execute("INSERT INTO comments VALUES (?, ?)", (id, text))
        return "ok"

    def issues_close(self, id: str, resolution: str) -> str:
        with self._db() as db:
            db.execute("UPDATE issues SET status = 'closed' WHERE id = ?", (id,))
            db.execute("INSERT INTO comments VALUES (?, ?)", (id, f"closed: {resolution}"))
        return "closed"

    def tools(self) -> list[Tool]:
        return [
            Tool("issues_list", "List issues by status: open, closed or all.", obj({"status": STR}), self.issues_list),
            Tool("issues_get", "Get one issue with its comments.", obj({"id": STR}, ["id"]), self.issues_get),
            Tool("issues_comment", "Add a comment to an issue.", obj({"id": STR, "text": STR}, ["id", "text"]),
                 self.issues_comment),
            Tool("issues_close", "Close an issue with a short resolution note.",
                 obj({"id": STR, "resolution": STR}, ["id", "resolution"]), self.issues_close),
        ]


# ----- JSON-RPC over stdio -----

def handle(message: dict, tools: dict[str, Tool], name: str) -> dict | None:
    method, msg_id = message.get("method"), message.get("id")
    if msg_id is None:  # a notification, such as notifications/initialized: no reply
        return None
    if method == "initialize":
        result: Any = {"protocolVersion": PROTOCOL_VERSION, "capabilities": {"tools": {}},
                       "serverInfo": {"name": name, "version": "0.1.0"}}
    elif method == "tools/list":
        result = {"tools": [t.spec() for t in tools.values()]}
    elif method == "tools/call":
        params = message.get("params", {})
        tool = tools.get(params.get("name"))
        if tool is None:
            return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32602, "message": f"unknown tool {params.get('name')}"}}
        try:
            value = tool.fn(**params.get("arguments", {}))
            result = {"content": [{"type": "text", "text": json.dumps(value)}], "structuredContent": {"result": value},
                      "isError": False}
        except Exception as exc:  # tool errors go back to the model as results, not protocol errors
            result = {"content": [{"type": "text", "text": f"{type(exc).__name__}: {exc}"}], "isError": True}
    elif method == "ping":
        result = {}
    else:
        return {"jsonrpc": "2.0", "id": msg_id, "error": {"code": -32601, "message": f"method not found: {method}"}}
    return {"jsonrpc": "2.0", "id": msg_id, "result": result}


def serve(tools: list[Tool], name: str, stdin=sys.stdin, stdout=sys.stdout) -> None:
    table = {t.name: t for t in tools}
    for line in stdin:
        if not line.strip():
            continue
        try:
            reply = handle(json.loads(line), table, name)
        except json.JSONDecodeError:
            reply = {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "parse error"}}
        if reply is not None:
            stdout.write(json.dumps(reply) + "\n")
            stdout.flush()


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("server", choices=["workspace", "tracker"])
    ap.add_argument("--root", type=Path)
    ap.add_argument("--db", type=Path, required=True)
    args = ap.parse_args(argv)
    if args.server == "workspace":
        serve(Workspace(args.root, Ledger(args.db)).tools(), "reviewer-workspace")
    else:
        serve(Tracker(args.db).tools(), "reviewer-tracker")


if __name__ == "__main__":
    main()
