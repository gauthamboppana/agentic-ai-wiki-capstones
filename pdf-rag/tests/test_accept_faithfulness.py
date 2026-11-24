"""Target: faithfulness of at least 0.90, every claim checked against the chunks it cites."""

from pdfrag.evaluate import TARGETS


def test_faithfulness_meets_target(report):
    claims = [v for r in report.rows for v in r.verdicts]
    assert len(claims) >= 40
    assert report.faithfulness >= TARGETS["faithfulness"]
