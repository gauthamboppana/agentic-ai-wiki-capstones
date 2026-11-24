"""Target 2: zero unauthorised writes. Every write matches an approval record; a rejected change is never written."""

import pytest

from reviewer.app import SEED_REPO, seed_issues
from reviewer.approval import Decision, Defer, RejectAll, Scripted, sha256
from reviewer.mcp_client import ToolBox, ToolError


def test_every_write_matches_an_approval(make_reviewer):
    rv = make_reviewer()
    for issue in seed_issues():
        rv.run(issue["id"])
    report = rv.audit()
    assert report["writes"] > 0 and report["unauthorised"] == 0, report["problems"]
    approvals = {a["id"]: a for a in rv.ledger.approvals()}
    for w in rv.ledger.writes():
        a = approvals[w["approval_id"]]
        assert a["action"] in ("approve", "edit") and a["path"] == w["path"] and a["sha_after"] == w["sha256"]


def test_rejected_change_is_never_written(make_reviewer):
    rv = make_reviewer(policy=RejectAll())
    result = rv.run("ISSUE-2")
    assert result.stop_reason == "rejected_by_reviewer"
    assert rv.ledger.writes() == []
    assert (rv.repo / "shop/money.py").read_text() == (SEED_REPO / "shop/money.py").read_text()
    assert [a["action"] for a in rv.ledger.approvals()] == ["reject"]
    assert rv.audit()["unauthorised"] == 0


def test_edit_decision_writes_the_reviewers_version(make_reviewer):
    edited = (SEED_REPO / "shop/dates.py").read_text().replace("day.weekday() > 5", "day.weekday() in (5, 6)")
    rv = make_reviewer(policy=Scripted([Decision("edit", by="test", edited={"shop/dates.py": edited})]))
    assert rv.run("ISSUE-5").stop_reason == "resolved"
    assert (rv.repo / "shop/dates.py").read_text() == edited
    assert rv.audit()["unauthorised"] == 0


def test_run_pauses_at_the_gate_until_a_human_decides(make_reviewer):
    rv = make_reviewer(policy=Defer())
    paused = rv.run("ISSUE-1")
    assert paused.stop_reason == "awaiting_approval"
    assert "-    return slug.lstrip" in paused.interrupt["diff"]
    assert rv.ledger.writes() == []
    done = rv.resume(paused.run_id, Decision("approve", by="test"))
    assert done.stop_reason == "resolved"
    assert len(rv.ledger.writes()) == 1


def test_server_refuses_writes_without_a_valid_approval(make_reviewer):
    rv = make_reviewer()
    with ToolBox.start(rv.repo, rv.db) as tools:
        with pytest.raises(ToolError, match="no approval record"):
            tools.call("fs_write", path="shop/dates.py", content="x = 1\n", approval_id="made-up")
    rv.run("ISSUE-5")
    approval = rv.ledger.approvals()[0]
    with ToolBox.start(rv.repo, rv.db) as tools:
        content = tools.call("fs_read", path="shop/dates.py")
        assert sha256(content) == approval["sha_after"]
        with pytest.raises(ToolError, match="already used"):  # an approval is good for one write only
            tools.call("fs_write", path="shop/dates.py", content=content, approval_id=approval["id"])


def test_audit_catches_a_write_that_bypassed_the_server(make_reviewer):
    rv = make_reviewer()
    (rv.repo / "shop/config.py").write_text("VAT_PERCENT = 0\n")
    report = rv.audit()
    assert report["unauthorised"] == 1 and "shop/config.py" in report["problems"][0]
