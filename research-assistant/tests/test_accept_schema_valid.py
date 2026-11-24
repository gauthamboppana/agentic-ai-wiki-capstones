"""Acceptance 1: 20 of 20 articles give schema-valid output. Retries count; silent fixes do not."""

from research_assistant.data import load_articles
from research_assistant.providers import DEFAULT_FAULTS, FakeModel
from research_assistant.schema import Findings
from research_assistant.score import score


def test_twenty_of_twenty_schema_valid():
    report = score(FakeModel())
    assert report.total == 20
    assert report.valid == 20, report.render()
    for aid, result in report.results.items():
        Findings.model_validate(result.findings.model_dump())  # re-validate what we report
        assert result.findings.article_id == aid


def test_repair_path_was_really_exercised():
    report = score(FakeModel())
    assert set(report.repaired) == set(DEFAULT_FAULTS)
    assert report.results["a17"].retries == 2
    assert report.first_try == 20 - len(DEFAULT_FAULTS)


def test_without_repairs_the_faulty_articles_fail():
    from research_assistant.pipeline import ResearchError, analyse
    failed = []
    for a in load_articles():
        try:
            analyse(a, FakeModel(first_token_delay=0), max_repairs=0)
        except ResearchError:
            failed.append(a["id"])
    assert set(failed) == set(DEFAULT_FAULTS)
