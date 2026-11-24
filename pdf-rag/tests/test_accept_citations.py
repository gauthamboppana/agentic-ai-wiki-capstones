"""Target: every answer cites a page, and every cited page resolves to a real page of a real manual."""

from pdfrag.corpus import page_index
from pdfrag.evaluate import citation_problem
from pdfrag.models import ScriptedModel
from pdfrag.pipeline import Pipeline


def test_every_answer_cites_a_page_that_resolves(report):
    answered = [r for r in report.rows if not r.answer.abstained]
    assert answered
    assert [(r.item["id"], r.citation_problem) for r in answered if r.citation_problem] == []


def test_a_page_that_does_not_exist_fails(docs, chunks):
    a = Pipeline(chunks, model=ScriptedModel(["x"])).answer("What is the factory IP address of the LX-40?")
    a.text, a.abstained = "It is 192.168.10.40. [lx40 p.9]", False
    assert "does not exist" in citation_problem(a, page_index(docs))
    a.text = "It is 192.168.10.40."
    assert citation_problem(a, page_index(docs)) == "no page cited"
