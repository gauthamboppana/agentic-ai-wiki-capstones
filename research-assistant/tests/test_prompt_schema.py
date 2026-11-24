"""M1: the prompt template fills every variable and the schema is strict."""

import pytest
from pydantic import ValidationError

from research_assistant.prompt import TEMPLATE, build_prompt
from research_assistant.schema import Findings, strict_json_schema


def test_prompt_has_article_rules_and_schema():
    p = build_prompt("x1", "A title", "Some text about Penmarrow.")
    assert '<article id="x1" title="A title">' in p
    assert "Some text about Penmarrow." in p
    assert "- Return one JSON object" in p
    assert '"confidence"' in p


def test_missing_variable_is_an_error():
    with pytest.raises(KeyError):
        TEMPLATE.substitute(article_id="x1")


def test_strict_schema_requires_everything_and_forbids_extras():
    schema = strict_json_schema()
    assert schema["strict"] is True
    objects = [schema["schema"], *schema["schema"]["$defs"].values()]
    for obj in objects:
        assert obj["additionalProperties"] is False
        assert set(obj["required"]) == set(obj["properties"])
    assert "title" not in schema["schema"]


def test_schema_rejects_bad_values():
    good = {"article_id": "a", "key_findings": ["x"], "entities": [{"name": "Ada", "type": "person"}], "confidence": 0.5}
    Findings.model_validate(good)
    for bad in ({**good, "confidence": 1.2}, {**good, "key_findings": []}, {**good, "extra": 1},
                {**good, "entities": [{"name": "Ada", "type": "alien"}]}):
        with pytest.raises(ValidationError):
            Findings.model_validate(bad)
