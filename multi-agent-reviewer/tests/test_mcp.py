"""The MCP servers answer initialize, tools/list and tools/call over stdio."""

import json
import subprocess
import sys

from reviewer.app import init_workdir


def rpc(proc, msg_id, method, params=None):
    proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": msg_id, "method": method, "params": params or {}}) + "\n")
    proc.stdin.flush()
    reply = json.loads(proc.stdout.readline())
    assert reply["id"] == msg_id
    return reply


def start(*args):
    return subprocess.Popen([sys.executable, "-m", "reviewer.mcp_server", *args],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)


def test_workspace_server_over_stdio(tmp_path):
    work = init_workdir(tmp_path)
    proc = start("workspace", "--root", str(work / "repo"), "--db", str(work / "state.db"))
    try:
        init = rpc(proc, 1, "initialize", {"protocolVersion": "2025-06-18", "capabilities": {}})
        assert init["result"]["serverInfo"]["name"] == "reviewer-workspace"
        tools = {t["name"]: t for t in rpc(proc, 2, "tools/list")["result"]["tools"]}
        assert set(tools) == {"fs_list", "fs_read", "fs_write", "tests_run"}
        assert tools["fs_write"]["inputSchema"]["required"] == ["path", "content", "approval_id"]
        call = rpc(proc, 3, "tools/call", {"name": "fs_read", "arguments": {"path": "shop/dates.py"}})["result"]
        assert not call["isError"] and "weekday()" in call["structuredContent"]["result"]
        escape = rpc(proc, 4, "tools/call", {"name": "fs_read", "arguments": {"path": "../state.db"}})["result"]
        assert escape["isError"] and "outside the repository" in escape["content"][0]["text"]
        assert rpc(proc, 5, "nope")["error"]["code"] == -32601
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)


def test_tracker_server_over_stdio(tmp_path):
    work = init_workdir(tmp_path)
    proc = start("tracker", "--db", str(work / "state.db"))
    try:
        rpc(proc, 1, "initialize")
        names = [t["name"] for t in rpc(proc, 2, "tools/list")["result"]["tools"]]
        assert "issues_get" in names
        issues = rpc(proc, 3, "tools/call", {"name": "issues_list", "arguments": {}})["result"]
        assert len(json.loads(issues["content"][0]["text"])) == 5
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)
