"""Target 1: at least five seeded issues fixed end to end, with the repo's tests passing after each fix."""

from reviewer.app import seed_issues


def test_all_seeded_issues_resolved_and_tests_pass_after_each(make_reviewer):
    rv = make_reviewer()
    issues = seed_issues()
    assert len(issues) >= 5
    failing_before = len(rv.suite()["failed"])
    assert failing_before == len(issues)  # one failing test per seeded bug
    for n, issue in enumerate(issues, start=1):
        result = rv.run(issue["id"])
        assert result.stop_reason == "resolved", result.state.get("stop_detail")
        suite = rv.suite()
        assert set(issue["tests"]) <= set(suite["passed"])  # the issue's own tests now pass
        assert len(suite["failed"]) == failing_before - n  # and nothing else broke
    assert rv.suite()["ok"]


def test_multi_file_issues_touch_more_than_one_file(make_reviewer):
    rv = make_reviewer()
    for issue in seed_issues():
        result = rv.run(issue["id"])
        if "multi-file" in issue["labels"]:
            assert len(result.state["written"]) >= 2, issue["id"]


def test_planner_recalls_a_similar_past_fix(make_reviewer):
    rv = make_reviewer()
    for issue_id in ("ISSUE-3", "ISSUE-4"):
        result = rv.run(issue_id)
    assert "ISSUE-3" in result.state["plan"]["recalled"]
    assert rv.memory.count() == 2


def test_critic_sends_a_partial_fix_back_for_another_attempt(make_reviewer):
    result = make_reviewer().run("ISSUE-3")
    assert result.stop_reason == "resolved"
    assert result.state["history"].count("critic") == 2
