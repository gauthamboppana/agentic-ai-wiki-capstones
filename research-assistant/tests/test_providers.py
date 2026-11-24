"""M2: the provider switch. Fake by default; the OpenAI-compatible adapter is tested with recorded SSE lines."""

import json

import pytest

from research_assistant.data import article_by_id
from research_assistant.pipeline import analyse
from research_assistant.providers import FakeModel, OpenAICompatible, get_provider


def test_default_is_fake():
    assert isinstance(get_provider({}), FakeModel)


def test_env_switches_to_openai_compatible_without_leaking_key():
    p = get_provider({"RA_PROVIDER": "openai", "RA_MODEL": "local-model", "RA_API_KEY": "not-a-real-key"})
    assert isinstance(p, OpenAICompatible)
    assert p.base_url == "http://localhost:11434/v1"
    assert "not-a-real-key" not in repr(p)


def test_openai_needs_a_model_name():
    with pytest.raises(ValueError, match="RA_MODEL"):
        get_provider({"RA_PROVIDER": "openai"})


def recorded_stream(text: str) -> list[str]:
    """What a host sends back: one `data:` line per delta, then [DONE]."""
    lines = [": keep-alive\n"]
    for i in range(0, len(text), 10):
        lines.append("data: " + json.dumps({"choices": [{"delta": {"content": text[i:i + 10]}}]}) + "\n")
        lines.append("\n")
    return [*lines, "data: [DONE]\n"]


def test_openai_adapter_runs_the_same_code_path(good_reply):
    seen = {}

    def transport(url, headers, body, timeout):
        seen.update(url=url, headers=headers, body=body)
        return recorded_stream(good_reply)

    p = OpenAICompatible("http://localhost:11434/v1/", "local-model", api_key="not-a-real-key", transport=transport)
    result = analyse(article_by_id("a01"), p)
    assert result.findings.article_id == "a01"
    assert seen["url"] == "http://localhost:11434/v1/chat/completions"
    assert seen["body"]["stream"] is True
    assert seen["body"]["response_format"]["json_schema"]["strict"] is True
    assert seen["headers"]["Authorization"].startswith("Bearer ")


def test_no_key_means_no_auth_header(good_reply):
    seen = {}

    def transport(url, headers, body, timeout):
        seen.update(headers=headers)
        return recorded_stream(good_reply)

    analyse(article_by_id("a01"), OpenAICompatible("http://localhost:11434/v1", "m", transport=transport))
    assert "Authorization" not in seen["headers"]
