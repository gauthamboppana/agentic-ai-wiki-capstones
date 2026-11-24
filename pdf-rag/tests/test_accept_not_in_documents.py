"""Target: say "Not in the documents." when retrieval is weak, and only then."""

from pdfrag.models import NOT_FOUND


def test_unanswerable_questions_are_refused(report):
    rows = [r for r in report.rows if not r.item["answerable"]]
    assert rows and all(r.answer.text == NOT_FOUND for r in rows)


def test_answerable_questions_are_not_refused(report):
    assert report.answered >= 0.95


def test_naive_rag_does_not_refuse(chunks, golden):
    from pdfrag.evaluate import evaluate
    from pdfrag.pipeline import Pipeline

    naive = evaluate(Pipeline(chunks, mode="naive"), golden)
    assert naive.not_found < 1.0  # the abstain gate is what makes the difference
