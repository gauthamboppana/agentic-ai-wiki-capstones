"""Target 3: a run killed mid-way resumes from its last checkpoint and finishes."""

import subprocess
import sys

import pytest

from reviewer.graph import SimulatedCrash


def test_killed_process_resumes_from_last_checkpoint(make_reviewer, tmp_path):
    work = tmp_path / "killed"
    killed = subprocess.run([sys.executable, "-m", "reviewer.cli", "run", "ISSUE-4", "--workdir", str(work),
                             "--approve", "auto", "--crash-after", "writer"],
                            capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=120)
    assert killed.returncode == 137  # os._exit, like kill -9: no clean-up ran

    rv = make_reviewer("killed")
    run_id, issue_id, status = rv.checkpoints.runs()[-1]
    assert (issue_id, status) == ("ISSUE-4", "running")
    last = rv.checkpoints.latest(run_id)
    assert (last.node, last.next_node) == ("writer", "critic")

    result = rv.resume(run_id)
    assert result.stop_reason == "resolved"
    assert result.trace == ["critic", "finish"]  # carried on, did not start again
    assert len(rv.ledger.writes()) == 2  # the two writes from before the kill, none repeated
    assert rv.audit()["unauthorised"] == 0


@pytest.mark.parametrize("node", ["planner", "fixer", "gate"])
def test_crash_after_any_node_resumes_and_finishes(make_reviewer, node):
    rv = make_reviewer()
    with pytest.raises(SimulatedCrash):
        rv.run("ISSUE-1", crash_after=node)
    run_id = rv.checkpoints.runs()[-1][0]
    result = rv.resume(run_id)
    assert result.stop_reason == "resolved"
    assert result.state["history"].count(node) == 1  # the crashed node did not run twice
    assert rv.audit()["unauthorised"] == 0


def test_resuming_a_finished_run_does_nothing(make_reviewer):
    rv = make_reviewer()
    done = rv.run("ISSUE-2")
    again = rv.resume(done.run_id)
    assert again.stop_reason == "resolved" and again.trace == []
