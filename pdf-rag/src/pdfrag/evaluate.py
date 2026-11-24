"""Score a pipeline on the golden set: context recall, faithfulness, citations and abstention."""

from __future__ import annotations

from dataclasses import dataclass, field

from .chunking import Chunk
from .corpus import load_chunks, load_documents, page_index
from .golden import load_golden
from .judge import ClaimSupportJudge, Judge, Verdict, judge_answer
from .models import NOT_FOUND
from .pipeline import Answer, Pipeline
from .text import citations, sentences

TARGETS = {"context_recall": 0.85, "faithfulness": 0.90, "citations": 1.0, "not_found": 1.0}


def context_recall(item: dict, retrieved: list[Chunk]) -> float:
    """Share of the item's evidence quotes found in a retrieved chunk on the right page."""
    found = 0
    for ev in item["evidence"]:
        found += any(c.doc_id == ev["doc"] and c.page == ev["page"] and ev["quote"].lower() in c.text.lower() for c in retrieved)
    return found / len(item["evidence"])


def citation_problem(answer: Answer, pages: set[tuple[str, int]]) -> str:
    """Empty if the answer cites at least one page and every sentence cites pages that exist."""
    if answer.abstained:
        return ""
    if not citations(answer.text):
        return "no page cited"
    for s in sentences(answer.text):
        cited = citations(s)
        if not cited:
            return f"sentence without a citation: {s!r}"
        missing = [c for c in cited if c not in pages]
        if missing:
            return f"{missing[0][0]} p.{missing[0][1]} does not exist"
    return ""


@dataclass
class Row:
    item: dict
    answer: Answer
    recall: float | None
    verdicts: list[Verdict] = field(default_factory=list)
    citation_problem: str = ""


@dataclass
class Report:
    rows: list[Row]

    def _mean(self, xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    @property
    def context_recall(self) -> float:
        return self._mean([r.recall for r in self.rows if r.recall is not None])

    @property
    def faithfulness(self) -> float:
        return self._mean([v.supported for r in self.rows for v in r.verdicts])

    @property
    def citations(self) -> float:
        answered = [r for r in self.rows if not r.answer.abstained]
        return self._mean([not r.citation_problem for r in answered])

    @property
    def not_found(self) -> float:
        return self._mean([r.answer.text == NOT_FOUND for r in self.rows if not r.item["answerable"]])

    @property
    def answered(self) -> float:
        return self._mean([not r.answer.abstained for r in self.rows if r.item["answerable"]])

    @property
    def answer_match(self) -> float:
        """Not a target, a sanity check: the answer contains the value each evidence quote holds."""
        def hit(r: Row) -> bool:
            return all(ev["quote"].split(": ", 1)[-1].rstrip(".").lower() in r.answer.text.lower() for ev in r.item["evidence"])

        return self._mean([hit(r) for r in self.rows if r.item["answerable"]])

    def metrics(self) -> dict[str, float]:
        return {
            "context_recall": self.context_recall,
            "faithfulness": self.faithfulness,
            "citations": self.citations,
            "not_found": self.not_found,
            "answered": self.answered,
            "answer_match": self.answer_match,
        }

    def passed(self) -> bool:
        m = self.metrics()
        return all(m[k] >= t for k, t in TARGETS.items())


def evaluate(pipeline: Pipeline | None = None, golden: list[dict] | None = None, judge: Judge | None = None) -> Report:
    pipeline = pipeline or Pipeline(load_chunks())
    judge = judge or ClaimSupportJudge()
    pages = page_index(load_documents())
    rows = []
    for item in golden or load_golden():
        answer = pipeline.answer(item["question"])
        recall = context_recall(item, answer.retrieval.chunks) if item["answerable"] else None
        verdicts = [] if answer.abstained else judge_answer(answer.text, answer.retrieval.chunks, judge)
        rows.append(Row(item, answer, recall, verdicts, citation_problem(answer, pages)))
    return Report(rows)
