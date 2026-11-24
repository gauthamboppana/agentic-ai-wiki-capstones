"""Chunking that respects layout: a chunk never crosses a page or a heading, and a table stays whole."""

from __future__ import annotations

from dataclasses import dataclass

from .layout import Block, Document, reading_order


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    page: int
    section: str
    kind: str  # "text" or "table"
    text: str

    @property
    def cite(self) -> str:
        return f"[{self.doc_id} p.{self.page}]"

    @property
    def search_text(self) -> str:
        """What the retrievers index: the heading path gives short chunks the context they lack."""
        return f"{self.doc_title}\n{self.section}\n{self.text}"

    @property
    def meta(self) -> dict:
        return {"doc_id": self.doc_id, "page": self.page, "kind": self.kind, "section": self.section}


def table_text(block: Block) -> str:
    """One line per row, each cell labelled by its column, so a row can be quoted and cited on its own."""
    lines = [block.caption] if block.caption else []
    for row in block.rows:
        cells = [f"{h}: {c}" for h, c in zip(block.header, row)]
        lines.append("; ".join(cells) + ("" if row[-1].endswith(".") else "."))
    return "\n".join(lines)


def chunk_document(doc: Document, max_words: int = 90) -> list[Chunk]:
    chunks: list[Chunk] = []
    section = ""

    def emit(page: int, kind: str, parts: list[str]) -> None:
        if parts:
            n = len(chunks) + 1
            chunks.append(Chunk(f"{doc.doc_id}-p{page}-c{n}", doc.doc_id, doc.title, page, section, kind, "\n".join(parts)))

    for page in doc.pages:
        buffer: list[str] = []
        for block in reading_order(page):
            if block.kind == "heading":
                emit(page.number, "text", buffer)
                buffer = []
                if block.level > 1:  # the document title is not a section
                    section = block.text
            elif block.kind == "table":
                emit(page.number, "text", buffer)
                buffer = []
                emit(page.number, "table", [table_text(block)])
            else:
                if buffer and sum(len(p.split()) for p in buffer) + len(block.text.split()) > max_words:
                    emit(page.number, "text", buffer)
                    buffer = []
                buffer.append(block.text)
        emit(page.number, "text", buffer)  # a chunk never runs over the page break
    return chunks
