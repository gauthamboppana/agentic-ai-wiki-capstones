import pytest

from pdfrag.models import NOT_FOUND, FakeModel, OpenAICompatibleModel, ScriptedModel, get_model
from pdfrag.pipeline import Pipeline


def test_default_model_is_the_offline_fake(monkeypatch):
    monkeypatch.delenv("PDFRAG_PROVIDER", raising=False)
    assert isinstance(get_model(), FakeModel)


def test_openai_compatible_adapter_never_shows_the_key(monkeypatch):
    monkeypatch.setenv("PDFRAG_PROVIDER", "openai-compatible")
    monkeypatch.setenv("PDFRAG_MODEL", "local-model")
    monkeypatch.setenv("PDFRAG_API_KEY", "test-key-not-real")
    model = get_model()
    assert isinstance(model, OpenAICompatibleModel)
    assert "test-key-not-real" not in repr(model)


def test_unknown_provider_fails_clearly(monkeypatch):
    monkeypatch.setenv("PDFRAG_PROVIDER", "nope")
    with pytest.raises(ValueError):
        get_model()


def test_uncited_answer_gets_one_repair_then_a_refusal(chunks):
    q = "What is the factory IP address of the LX-40?"
    fixed = Pipeline(chunks, model=ScriptedModel(["It is 192.168.10.40.", "It is 192.168.10.40. [lx40 p.2]"])).answer(q)
    assert fixed.citations == [("lx40", 2)]
    refused = Pipeline(chunks, model=ScriptedModel(["It is 192.168.10.40.", "Still no citation."])).answer(q)
    assert refused.text == NOT_FOUND and "without a citation" in refused.reason
