"""A minimal MCP client: starts a server as a subprocess and talks JSON-RPC to it over stdio."""

from __future__ import annotations

import json
import subprocess
import sys
from itertools import count
from pathlib import Path


class ToolError(RuntimeError):
    pass


class McpClient:
    def __init__(self, command: list[str]):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self._ids = count(1)
        self.server_info = self.request("initialize", {
            "protocolVersion": "2025-06-18", "capabilities": {},
            "clientInfo": {"name": "reviewer", "version": "0.1.0"}})["serverInfo"]
        self.notify("notifications/initialized")
        self.tools = {t["name"]: t for t in self.request("tools/list")["tools"]}

    def notify(self, method: str, params: dict | None = None) -> None:
        self._send({"jsonrpc": "2.0", "method": method, "params": params or {}})

    def request(self, method: str, params: dict | None = None) -> dict:
        msg_id = next(self._ids)
        self._send({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}})
        line = self.proc.stdout.readline()
        if not line:
            raise ToolError(f"server exited during {method}")
        reply = json.loads(line)
        if "error" in reply:
            raise ToolError(reply["error"]["message"])
        return reply["result"]

    def call(self, name: str, **arguments):
        result = self.request("tools/call", {"name": name, "arguments": arguments})
        if result.get("isError"):
            raise ToolError(result["content"][0]["text"])
        if "structuredContent" in result:
            return result["structuredContent"]["result"]
        return json.loads(result["content"][0]["text"])

    def _send(self, message: dict) -> None:
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()

    def close(self) -> None:
        if self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)


class ToolBox:
    """Routes a tool name to whichever connected server offers it."""

    def __init__(self, clients: list[McpClient]):
        self.clients = clients
        self.route = {name: c for c in clients for name in c.tools}

    @classmethod
    def start(cls, repo: Path, db: Path) -> "ToolBox":
        py = sys.executable
        return cls([
            McpClient([py, "-m", "reviewer.mcp_server", "workspace", "--root", str(repo), "--db", str(db)]),
            McpClient([py, "-m", "reviewer.mcp_server", "tracker", "--db", str(db)]),
        ])

    def call(self, name: str, **arguments):
        if name not in self.route:
            raise ToolError(f"no server offers {name}")
        return self.route[name].call(name, **arguments)

    def names(self) -> list[str]:
        return sorted(self.route)

    def close(self) -> None:
        for c in self.clients:
            c.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
