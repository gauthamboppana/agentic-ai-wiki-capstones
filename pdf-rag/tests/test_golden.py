from pdfrag.golden import synthesise


def test_fifty_questions_mixed_sources(golden):
    assert len(golden) == 50
    assert len({i["id"] for i in golden}) == 50
    sources = {s: sum(i["source"] == s for i in golden) for s in ("synthetic", "hand")}
    assert sources["synthetic"] >= 20 and sources["hand"] >= 20
    assert sum(not i["answerable"] for i in golden) >= 5


def test_synthetic_file_matches_the_generator(docs, golden):
    assert [i for i in golden if i["source"] == "synthetic"] == synthesise(docs)


def test_every_evidence_quote_exists_on_its_page(golden, chunks):
    for item in golden:
        assert bool(item["evidence"]) == item["answerable"], item["id"]
        for ev in item["evidence"]:
            page = [c.text.lower() for c in chunks if c.doc_id == ev["doc"] and c.page == ev["page"]]
            assert any(ev["quote"].lower() in t for t in page), (item["id"], ev)
