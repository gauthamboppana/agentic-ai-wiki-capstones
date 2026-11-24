"""M5: the bundled test set and the scoring script."""

from research_assistant.data import load_articles
from research_assistant.providers import FakeModel
from research_assistant.score import percentile, score


def test_twenty_original_articles():
    articles = load_articles()
    assert len(articles) == 20
    assert len({a["id"] for a in articles}) == 20
    for a in articles:
        assert len(a["text"].split()) >= 40 and a["expected_entities"]


def test_percentile_nearest_rank():
    values = [float(v) for v in range(1, 11)]
    assert percentile(values, 90) == 9.0
    assert percentile(values, 50) == 5.0
    assert percentile([0.3], 90) == 0.3


def test_report_renders_every_target():
    report = score(FakeModel(first_token_delay=0), load_articles()[:3])
    text = report.render()
    for label in ("Schema-valid outputs : 3/3", "Time to first token", "Entity recall"):
        assert label in text
