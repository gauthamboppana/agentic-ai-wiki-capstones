from pdfrag.bm25 import BM25
from pdfrag.fusion import rrf
from pdfrag.pipeline import Pipeline


def test_bm25_finds_exact_codes(chunks):
    top, _ = BM25(chunks).search("E12", k=1)[0]
    assert (top.doc_id, top.page, top.kind) == ("h200", 4, "table")


def test_rrf_rewards_chunks_both_lists_agree_on(chunks):
    a, b, c = chunks[:3]
    fused = [ch.chunk_id for ch, _ in rrf([[a, b], [c, b]])]
    assert fused[0] == b.chunk_id  # second in both beats first in one


def test_route_filters_to_the_named_product():
    assert Pipeline.route("What does event 204 mean on the TG-7?") == {"doc_id": "tg7"}
    assert Pipeline.route("Which devices need termination?") is None


def test_reranker_puts_the_answer_first(pipeline):
    r = pipeline.retrieve("How long does the TG-7 buffer readings when both links are down?")
    assert (r.chunks[0].doc_id, r.chunks[0].page) == ("tg7", 2)
    assert r.top_score >= pipeline.threshold
