"""M4: strict validation, up to two repair retries, then a clear error."""

import pytest

from research_assistant.pipeline import MAX_REPAIRS, ResearchError, analyse, check
from research_assistant.providers import ScriptedModel


@pytest.mark.parametrize("text,kind", [
    ("", "empty"),
    ('{"article_id": "a01", "key_fin', "truncated"),
    ("I'm sorry, but I can't help with that.", "refused"),
    ('```json\n{"article_id": "a01"}\n```', "not_json"),
    ('{"article_id": "a01"}', "schema"),
])
def test_check_classifies_problems(text, kind):
    findings, got, message = check(text, "a01")
    assert findings is None and got == kind and message


def test_no_silent_fixes(good_reply):
    findings, kind, _ = check("Sure! " + good_reply, "a01")
    assert findings is None and kind == "not_json"


def test_wrong_article_id_is_rejected(good_reply):
    assert check(good_reply, "a02")[1] == "schema"


def test_repair_prompt_carries_the_error(article, good_reply):
    model = ScriptedModel(['{"article_id": "a01"}', good_reply])
    result = analyse(article, model)
    assert result.retries == 1
    repair = model.calls[1]
    assert repair[-2] == {"role": "assistant", "content": '{"article_id": "a01"}'}
    assert "schema" in repair[-1]["content"] and "key_findings" in repair[-1]["content"]


def test_two_repairs_is_the_limit(article):
    model = ScriptedModel(["nope"] * 10)
    with pytest.raises(ResearchError) as info:
        analyse(article, model)
    assert len(model.calls) == 1 + MAX_REPAIRS
    assert len(info.value.attempts) == 3


def test_provider_exception_is_an_attempt_not_a_crash(article, good_reply):
    class Flaky:
        name = "flaky"
        calls = 0

        def stream(self, messages):
            Flaky.calls += 1
            if Flaky.calls == 1:
                raise TimeoutError("read timed out")
            yield good_reply

    result = analyse(article, Flaky())
    assert result.retries == 1 and result.attempts[0].error_kind == "provider_error"


def test_unsure_below_threshold(article, good_reply):
    result = analyse(article, ScriptedModel([good_reply]), unsure_below=0.99)
    assert result.status == "unsure"
