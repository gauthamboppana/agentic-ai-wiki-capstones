"""The question-answering pipeline: route, retrieve (BM25 + dense, fused with RRF), rerank, answer, cite."""

from __future__ import annotations

from dataclasses import dataclass, field

from .bm25 import BM25
from .chunking import Chunk
from .corpus import ALIASES
from .embed import Embedder, HashingEmbedder
from .fusion import rrf
from .models import NOT_FOUND, SYSTEM_PROMPT, ChatModel, FakeModel, build_prompt
from .rerank import LexicalReranker, Reranker
from .store import InMemoryStore, VectorStore, matches
from .text import citations, sentences, tokens


@dataclass
class Retrieval:
    question: str
    where: dict | None
    candidates: list[Chunk]  # fused list handed to the reranker
    ranked: list[tuple[Chunk, float]]  # reranked top_n, best first

    @property
    def top_score(self) -> float:
        return self.ranked[0][1] if self.ranked else 0.0

    @property
    def chunks(self) -> list[Chunk]:
        return [c for c, _ in self.ranked]


@dataclass
class Answer:
    question: str
    text: str
    abstained: bool
    retrieval: Retrieval
    citations: list[tuple[str, int]] = field(default_factory=list)
    reason: str = ""


def check_citations(text: str, contexts: list[Chunk]) -> str:
    """Empty string if every sentence cites a page that was in the context, else what is wrong."""
    allowed = {(c.doc_id, c.page) for c in contexts}
    for s in sentences(text):
        cited = citations(s)
        if not cited:
            return f"sentence without a citation: {s!r}"
        bad = [c for c in cited if c not in allowed]
        if bad:
            return f"citation {bad[0][0]} p.{bad[0][1]} is not one of the context pages"
    return ""


class Pipeline:
    def __init__(
        self,
        chunks: list[Chunk],
        store: VectorStore | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        model: ChatModel | None = None,
        mode: str = "hybrid",  # "hybrid" or "naive" (dense only, no routing, no rerank, no abstain gate)
        k: int = 20,
        top_n: int = 5,
        threshold: float = 0.5,
    ):
        self.mode, self.k, self.top_n, self.threshold = mode, k, top_n, threshold
        self.embedder = embedder or HashingEmbedder()
        self.store = store or InMemoryStore()
        self.store.add(chunks, self.embedder.embed([c.search_text for c in chunks]))
        self.bm25 = BM25(chunks)
        self.reranker = reranker or LexicalReranker(self.bm25.idf, self.bm25.max_idf, ignore=set(ALIASES))
        self.model = model or FakeModel()

    @staticmethod
    def route(question: str) -> dict | None:
        """If the question names exactly one product, search only that manual (a metadata filter)."""
        docs = {ALIASES[t] for t in tokens(question) if t in ALIASES}
        return {"doc_id": docs.pop()} if len(docs) == 1 else None

    def retrieve(self, question: str, where: dict | None = None) -> Retrieval:
        vector = self.embedder.embed([question])[0]
        if self.mode == "naive":
            dense = self.store.search(vector, self.top_n, where)
            return Retrieval(question, where, [c for c, _ in dense], dense)
        where = where if where is not None else self.route(question)
        keyword = [c for c, _ in self.bm25.search(question, self.k, keep=lambda c: matches(c, where))]
        dense = [c for c, _ in self.store.search(vector, self.k, where)]
        fused = [c for c, _ in rrf([keyword, dense])][: self.k]
        return Retrieval(question, where, fused, self.reranker.rerank(question, fused, self.top_n))

    def answer(self, question: str, where: dict | None = None) -> Answer:
        r = self.retrieve(question, where)
        if self.mode != "naive" and r.top_score < self.threshold:
            return Answer(question, NOT_FOUND, True, r, reason=f"weak retrieval (top score {r.top_score:.2f})")
        prompt = build_prompt(question, r.chunks)
        text = self.model.complete(SYSTEM_PROMPT, prompt)
        problem = "" if text.startswith(NOT_FOUND) else check_citations(text, r.chunks)
        if problem:  # one repair attempt, then refuse rather than show an unsupported answer
            text = self.model.complete(SYSTEM_PROMPT, prompt + f"\nYour last answer was rejected: {problem}. Fix it.")
            problem = "" if text.startswith(NOT_FOUND) else check_citations(text, r.chunks)
            if problem:
                return Answer(question, NOT_FOUND, True, r, reason=problem)
        if text.startswith(NOT_FOUND):
            return Answer(question, NOT_FOUND, True, r, reason="model found no answer in the context")
        return Answer(question, text, False, r, citations=list(dict.fromkeys(citations(text))))
