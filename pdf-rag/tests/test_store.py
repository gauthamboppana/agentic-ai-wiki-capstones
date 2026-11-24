import pytest

from pdfrag.embed import HashingEmbedder
from pdfrag.store import InMemoryStore, SQLiteStore


@pytest.fixture(scope="module", params=["memory", "sqlite"])
def store(request, chunks):
    s = InMemoryStore() if request.param == "memory" else SQLiteStore(":memory:")
    s.add(chunks, HashingEmbedder().embed([c.search_text for c in chunks]))
    return s


def test_filters_by_document_and_kind(store):
    assert {c.doc_id for c in store.chunks({"doc_id": "tg7"})} == {"tg7"}
    tables = store.chunks({"doc_id": "lx40", "kind": "table"})
    assert [c.page for c in tables] == [1, 3, 4]


def test_list_value_means_any_of(store):
    assert {c.page for c in store.chunks({"doc_id": "h200", "page": [3, 4]})} == {3, 4}


def test_filtered_search_only_returns_matching_chunks(store):
    vector = HashingEmbedder().embed(["firmware update"])[0]
    hits = store.search(vector, k=3, where={"doc_id": "lx40"})
    assert hits and all(c.doc_id == "lx40" for c, _ in hits)


def test_both_stores_rank_the_same(chunks):
    vectors = HashingEmbedder().embed([c.search_text for c in chunks])
    a, b = InMemoryStore(), SQLiteStore()
    a.add(chunks, vectors)
    b.add(chunks, vectors)
    q = HashingEmbedder().embed(["accumulator pre-charge"])[0]
    assert [c.chunk_id for c, _ in a.search(q, 5)] == [c.chunk_id for c, _ in b.search(q, 5)]


def test_sqlite_rejects_unknown_filter_fields():
    with pytest.raises(ValueError):
        SQLiteStore().chunks({"text": "x"})
