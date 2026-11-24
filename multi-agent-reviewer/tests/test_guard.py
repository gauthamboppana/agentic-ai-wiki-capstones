"""The max-iterations guard stops a looping run with a clear stop reason."""

from reviewer.model import ScriptedModel

# A fixer that undoes its own edit forever, so the critic never passes it.
LOOP = {"ISSUE-1": {"files": ["shop/textutil.py"], "finding": "?", "lesson": "", "cycle": True, "attempts": [
    [{"path": "shop/textutil.py", "find": 'lstrip("-")', "replace": 'lstrip("-" )'}],
    [{"path": "shop/textutil.py", "find": 'lstrip("-" )', "replace": 'lstrip("-")'}]]}}


def test_looping_run_stops_at_the_limit(make_reviewer):
    rv = make_reviewer(model=ScriptedModel(LOOP), max_steps=12)
    result = rv.run("ISSUE-1")
    assert result.stop_reason == "max_iterations"
    assert result.steps == 12
    assert "limit 12" in result.state["stop_detail"]
    assert rv.checkpoints.latest(result.run_id).status == "stopped"
    assert rv.audit()["unauthorised"] == 0  # even the looping writes were each approved


def test_every_run_ends_with_a_stop_reason(make_reviewer):
    rv = make_reviewer(model=ScriptedModel({}))  # a model with no idea about any issue
    assert rv.run("ISSUE-1").stop_reason == "gave_up"
