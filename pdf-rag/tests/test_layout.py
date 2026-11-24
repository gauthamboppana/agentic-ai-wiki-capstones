from pdfrag.chunking import chunk_document
from pdfrag.layout import reading_order


def test_two_column_page_reads_left_column_then_right(docs):
    page = docs[0].pages[0]  # h200 page 1: blocks are stored in drawing order, not reading order
    texts = [b.text or b.caption for b in reading_order(page)]
    assert texts[0] == "Halden H-200 Booster Pump"
    assert texts.index("1 Safety") < texts.index("2 Specifications")
    assert texts.index("Table 1: H-200 specifications") > texts.index("2 Specifications")


def test_running_headers_and_footers_are_dropped(chunks):
    assert not any("revision C, page" in c.text or c.text.startswith("H-200 manual") for c in chunks)


def test_every_chunk_has_a_page_and_never_crosses_one(docs):
    for doc in docs:
        for chunk in chunk_document(doc):
            page = next(p for p in doc.pages if p.number == chunk.page)
            page_text = " ".join(b.text for b in page.blocks) + " ".join(" ".join(r) for b in page.blocks for r in b.rows)
            first_words = chunk.text.splitlines()[-1].split(":")[-1].split()[:3]
            assert " ".join(first_words).rstrip(".") in page_text


def test_tables_stay_whole_with_labelled_cells(chunks):
    table = next(c for c in chunks if c.kind == "table" and c.doc_id == "h200" and c.page == 4)
    assert table.text.count("Code:") == 5
    assert "Code: E12; Meaning: Pressure sensor fault; Action: Replace the pressure sensor, part number HS-112." in table.text
    assert table.section == "6 Fault codes"
