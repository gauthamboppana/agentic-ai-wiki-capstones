"""Acceptance 3: no crash on truncated or refused output. Three broken cases: retry recovers, or a clear error."""

import pytest

from research_assistant.cli import broken_cases
from research_assistant.pipeline import ResearchError, analyse
from research_assistant.providers import ScriptedModel
from research_assistant.sse import parse_sse, sse_stream

TRUNCATED = '{"article_id": "a01", "key_findings": ["The coastal town of Pen'
REFUSED = "I'm sorry, but I can't help with that request."


def test_case_1_truncated_then_recovers(article, good_reply):
    result = analyse(article, ScriptedModel([TRUNCATED, good_reply]))
    assert result.retries == 1
    assert result.attempts[0].error_kind == "truncated"


def test_case_2_refused_then_recovers(article, good_reply):
    result = analyse(article, ScriptedModel([REFUSED, REFUSED, good_reply]))
    assert result.retries == 2
    assert [a.error_kind for a in result.attempts] == ["refused", "refused", None]


def test_case_3_exhausted_retries_raise_a_clear_error(article):
    with pytest.raises(ResearchError) as info:
        analyse(article, ScriptedModel([TRUNCATED, REFUSED, TRUNCATED]))
    message = str(info.value)
    assert "'a01'" in message and "3 attempts" in message and "2 repairs" in message
    assert "truncated" in message
    assert [a.error_kind for a in info.value.attempts] == ["truncated", "refused", "truncated"]


def test_stream_ends_with_error_frame_not_exception(article):
    events = parse_sse("".join(sse_stream(article, ScriptedModel([REFUSED] * 3))))
    assert events[-1][0] == "error"
    assert [d["will_retry"] for e, d in events if e == "invalid"] == [True, True, False]


def test_demo_reports_all_three_cases():
    lines = broken_cases()
    assert len(lines) == 3
    assert "recovered after 1" in lines[0] and "recovered after 1" in lines[1]
    assert "clear error" in lines[2]
