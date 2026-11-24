"""A small state graph engine: nodes, conditional edges, checkpoints, interrupts and a step guard.

Each node is a function `fn(state, ctx) -> dict` that returns updates to merge into the state.
After every node the engine works out the next node, saves a checkpoint, and moves on.
Every run ends with an explicit stop reason, never silently.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Callable

END = "__end__"

NodeFn = Callable[[dict, Any], dict]
Router = Callable[[dict], str]


class StopReason:
    RESOLVED = "resolved"
    REJECTED = "rejected_by_reviewer"
    AWAITING_APPROVAL = "awaiting_approval"
    MAX_ITERATIONS = "max_iterations"
    GAVE_UP = "gave_up"
    ALREADY_FIXED = "already_fixed"
    COMPLETED = "completed"


class Interrupt(Exception):
    """Raised by a node to pause the run. The node runs again on resume."""

    def __init__(self, reason: str, payload: dict | None = None):
        super().__init__(reason)
        self.reason = reason
        self.payload = payload or {}


class SimulatedCrash(RuntimeError):
    """Raised after a checkpoint to imitate the process being killed mid-run."""


@dataclass
class RunResult:
    run_id: str
    stop_reason: str
    state: dict
    steps: int
    trace: list[str] = field(default_factory=list)
    interrupt: dict | None = None


class Graph:
    def __init__(self, entry: str, max_steps: int = 20):
        self.entry = entry
        self.max_steps = max_steps
        self.nodes: dict[str, NodeFn] = {}
        self.edges: dict[str, str | Router] = {}

    def add_node(self, name: str, fn: NodeFn) -> None:
        self.nodes[name] = fn

    def add_edge(self, src: str, dst: str) -> None:
        self.edges[src] = dst

    def add_conditional_edges(self, src: str, router: Router) -> None:
        self.edges[src] = router

    def next_node(self, src: str, state: dict) -> str:
        edge = self.edges.get(src, END)
        dst = edge(state) if callable(edge) else edge
        if dst != END and dst not in self.nodes:
            raise ValueError(f"edge from {src} leads to unknown node {dst}")
        return dst

    def run(self, state: dict, ctx: Any, *, run_id: str, checkpointer, start: str | None = None,
            step: int = 0, crash_after: str | None = None, hard_crash: bool = False) -> RunResult:
        node = start or self.entry
        trace: list[str] = []
        while node != END:
            if step >= self.max_steps:
                state["stop_reason"] = StopReason.MAX_ITERATIONS
                state["stop_detail"] = f"stopped after {step} node runs without finishing (limit {self.max_steps})"
                checkpointer.save(run_id, step, node, node, state, status="stopped")
                return RunResult(run_id, StopReason.MAX_ITERATIONS, state, step, trace)
            try:
                updates = self.nodes[node](state, ctx) or {}
            except Interrupt as stop:
                state["stop_reason"] = StopReason.AWAITING_APPROVAL
                state["stop_detail"] = stop.reason
                checkpointer.save(run_id, step, node, node, state, status="interrupted")
                return RunResult(run_id, StopReason.AWAITING_APPROVAL, state, step, trace, stop.payload)
            state.update(updates)
            step += 1
            trace.append(node)
            state.setdefault("history", []).append(node)
            nxt = self.next_node(node, state)
            status = "finished" if nxt == END else "running"
            checkpointer.save(run_id, step, node, nxt, state, status=status)
            if crash_after == node:
                if hard_crash:
                    os._exit(137)  # like kill -9: no cleanup, only the checkpoint survives
                raise SimulatedCrash(f"simulated crash after {node}")
            node = nxt
        reason = state.get("stop_reason") or StopReason.COMPLETED
        state["stop_reason"] = reason
        return RunResult(run_id, reason, state, step, trace)
