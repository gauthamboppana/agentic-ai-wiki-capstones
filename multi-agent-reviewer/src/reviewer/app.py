"""Wires the graph to a work directory: a fresh copy of the seeded repo plus one SQLite state file."""

from __future__ import annotations

import json
import shutil
import uuid
from pathlib import Path

from reviewer.agents import Ctx, build_graph
from reviewer.approval import ApprovalPolicy, AutoApprove, Decision, Ledger, audit
from reviewer.checkpoint import SqliteCheckpointer
from reviewer.graph import RunResult
from reviewer.mcp_client import ToolBox
from reviewer.mcp_server import Tracker
from reviewer.memory import EpisodicMemory
from reviewer.model import default_model

SEED = Path(__file__).parent / "seed"
SEED_REPO = SEED / "repo"


def seed_issues() -> list[dict]:
    return json.loads((SEED / "issues.json").read_text(encoding="utf-8"))


def init_workdir(workdir: Path) -> Path:
    """Copy the seeded repo into workdir/repo and load the seeded issues. Safe to call twice."""
    workdir = Path(workdir)
    repo = workdir / "repo"
    if not repo.exists():
        shutil.copytree(SEED_REPO, repo, ignore=shutil.ignore_patterns("__pycache__"))
    Tracker(workdir / "state.db").seed(seed_issues())
    return workdir


class Reviewer:
    def __init__(self, workdir: Path, model=None, policy: ApprovalPolicy | None = None, max_steps: int = 20):
        self.workdir = init_workdir(Path(workdir))
        self.repo, self.db = self.workdir / "repo", self.workdir / "state.db"
        self.model = model or default_model()
        self.policy = policy or AutoApprove()
        self.graph = build_graph(max_steps)
        self.checkpoints = SqliteCheckpointer(self.db)
        self.ledger = Ledger(self.db)
        self.memory = EpisodicMemory(self.db)

    def _go(self, run_id: str, state: dict, **kw) -> RunResult:
        with ToolBox.start(self.repo, self.db) as tools:
            ctx = Ctx(tools, self.model, self.policy, self.memory, self.ledger, run_id)
            return self.graph.run(state, ctx, run_id=run_id, checkpointer=self.checkpoints, **kw)

    def run(self, issue_id: str, crash_after: str | None = None, hard_crash: bool = False) -> RunResult:
        run_id = f"{issue_id}-{uuid.uuid4().hex[:6]}"
        return self._go(run_id, {"issue_id": issue_id}, crash_after=crash_after, hard_crash=hard_crash)

    def resume(self, run_id: str, decision: Decision | None = None, crash_after: str | None = None) -> RunResult:
        """Continue a run from its last checkpoint. A decision answers a run paused at the approval gate."""
        cp = self.checkpoints.latest(run_id)
        if cp is None:
            raise KeyError(f"no checkpoints for run {run_id}")
        state = cp.state
        if cp.status in ("finished", "stopped"):
            return RunResult(run_id, state.get("stop_reason", "completed"), state, cp.step)
        if decision is not None:
            state["pending_decision"] = decision.to_dict()
        state.pop("stop_reason", None)
        state.pop("stop_detail", None)
        return self._go(run_id, state, start=cp.next_node, step=cp.step, crash_after=crash_after)

    def audit(self) -> dict:
        return audit(self.ledger, self.repo, SEED_REPO)

    def suite(self) -> dict:
        with ToolBox.start(self.repo, self.db) as tools:
            return tools.call("tests_run")
