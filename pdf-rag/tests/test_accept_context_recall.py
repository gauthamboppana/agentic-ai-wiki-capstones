"""Target: context recall of at least 0.85 on the 50-question golden set."""

from pdfrag.evaluate import TARGETS


def test_context_recall_meets_target(report):
    assert report.context_recall >= TARGETS["context_recall"]


def test_recall_counts_only_answerable_questions(report):
    scored = [r for r in report.rows if r.recall is not None]
    assert len(scored) == sum(r.item["answerable"] for r in report.rows)
