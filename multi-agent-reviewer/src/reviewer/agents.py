"""The four agents (planner, investigator, fixer, critic), the approval gate, and the graph that joins them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from reviewer.approval import ApprovalPolicy, Decision, Ledger, unified_diff
from reviewer.graph import END, Graph, Interrupt, StopReason
from reviewer.memory import EpisodicMemory
from reviewer.mcp_client import ToolBox


@dataclass
class Ctx:
    """Everything a node needs that is not state: live connections, never checkpointed."""
    tools: ToolBox
    model: Any
    policy: ApprovalPolicy
    memory: EpisodicMemory
    ledger: Ledger
    run_id: str


def planner(state: dict, ctx: Ctx) -> dict:
    issue = ctx.tools.call("issues_get", id=state["issue_id"])
    memories = ctx.memory.search(issue["title"] + " " + issue["body"])
    plan = ctx.model.decide("planner", {"issue": issue, "memories": memories})
    return {"issue": issue, "memories": memories, "plan": plan, "attempt": 0, "feedback": ""}


def investigator(state: dict, ctx: Ctx) -> dict:
    updates: dict = {}
    if "baseline_passed" not in state:
        updates["baseline_passed"] = ctx.tools.call("tests_run")["passed"]
    files = {p: ctx.tools.call("fs_read", path=p) for p in state["plan"]["files"]}
    result = ctx.tools.call("tests_run", tests=state["issue"]["tests"])
    if state["attempt"] == 0 and result["ok"]:
        return {**updates, "files": files, "stop_reason": StopReason.ALREADY_FIXED}
    finding = ctx.model.decide("investigator", {"issue": state["issue"], "files": files, "failing": result["output"]})
    return {**updates, "files": files, "finding": finding["finding"], "failing": result["failed"]}


def fixer(state: dict, ctx: Ctx) -> dict:
    reply = ctx.model.decide("fixer", {"issue": state["issue"], "files": state["files"], "finding": state["finding"],
                                       "attempt": state["attempt"], "feedback": state["feedback"]})
    if not reply.get("edits"):
        return {"proposal": None, "stop_reason": StopReason.GAVE_UP}
    after = dict(state["files"])
    for e in reply["edits"]:
        if e["path"] not in after or e["find"] not in after[e["path"]]:
            return {"proposal": None, "attempt": state["attempt"] + 1,
                    "feedback": f"edit to {e['path']} did not apply: text not found"}
        after[e["path"]] = after[e["path"]].replace(e["find"], e["replace"], 1)
    changed = {p: c for p, c in after.items() if c != state["files"][p]}
    diff = "".join(unified_diff(p, state["files"][p], c) for p, c in changed.items())
    return {"proposal": {"issue_id": state["issue_id"], "attempt": state["attempt"], "summary": reply["summary"],
                         "before": {p: state["files"][p] for p in changed}, "after": changed, "diff": diff}}


def gate(state: dict, ctx: Ctx) -> dict:
    """The interrupt before any write. Nothing reaches the filesystem without a decision recorded here."""
    proposal = state["proposal"]
    pending = state.get("pending_decision")
    decision = Decision.from_dict(pending) if pending else ctx.policy.decide(proposal)
    if decision is None:
        raise Interrupt("waiting for a reviewer to approve, edit or reject", {"diff": proposal["diff"]})
    write_set, approvals = {}, []
    for path, after in proposal["after"].items():
        content = decision.edited.get(path, after) if decision.action == "edit" else after
        approval_id = ctx.ledger.record(ctx.run_id, state["issue_id"], path, decision, proposal["before"][path], content)
        approvals.append({"id": approval_id, "path": path})
        write_set[path] = content
    updates = {"decision": decision.action, "pending_decision": None, "approvals": approvals}
    if decision.action == "reject":
        ctx.tools.call("issues_comment", id=state["issue_id"], text=f"fix rejected: {decision.reason}")
        return {**updates, "write_set": {}, "stop_reason": StopReason.REJECTED}
    return {**updates, "write_set": write_set}


def writer(state: dict, ctx: Ctx) -> dict:
    for a in state["approvals"]:
        content = state["write_set"][a["path"]]
        if ctx.tools.call("fs_read", path=a["path"]) != content:  # skip writes already done before a crash
            ctx.tools.call("fs_write", path=a["path"], content=content, approval_id=a["id"])
    return {"written": sorted(set(state.get("written", [])) | set(state["write_set"]))}


def critic(state: dict, ctx: Ctx) -> dict:
    target = ctx.tools.call("tests_run", tests=state["issue"]["tests"])
    suite = ctx.tools.call("tests_run")
    regressions = sorted(set(state["baseline_passed"]) - set(suite["passed"]))
    verdict = ctx.model.decide("critic", {"issue": state["issue"], "failed": target["failed"],
                                          "regressions": regressions, "diff": state["proposal"]["diff"]})
    passed = verdict["verdict"] == "pass" and target["ok"] and not regressions  # code has the final word
    if passed:
        return {"verdict": "pass", "lesson": verdict.get("lesson", ""), "suite_passed": len(suite["passed"])}
    return {"verdict": "retry", "attempt": state["attempt"] + 1, "feedback": verdict.get("feedback", "tests still fail")}


def finish(state: dict, ctx: Ctx) -> dict:
    issue = state["issue"]
    ctx.memory.add(issue["id"], issue["title"] + " " + issue["body"], state["written"], state["lesson"])
    ctx.tools.call("issues_close", id=issue["id"], resolution=state["lesson"])
    return {"stop_reason": StopReason.RESOLVED}


def build_graph(max_steps: int = 20) -> Graph:
    g = Graph(entry="planner", max_steps=max_steps)
    for name, fn in [("planner", planner), ("investigator", investigator), ("fixer", fixer), ("gate", gate),
                     ("writer", writer), ("critic", critic), ("finish", finish)]:
        g.add_node(name, fn)
    g.add_edge("planner", "investigator")
    g.add_conditional_edges("investigator", lambda s: END if s.get("stop_reason") else "fixer")
    g.add_conditional_edges("fixer", lambda s: "gate" if s["proposal"] else (END if s.get("stop_reason") else "investigator"))
    g.add_conditional_edges("gate", lambda s: END if s["decision"] == "reject" else "writer")
    g.add_edge("writer", "critic")
    g.add_conditional_edges("critic", lambda s: "finish" if s["verdict"] == "pass" else "investigator")
    g.add_edge("finish", END)
    return g
