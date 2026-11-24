"""The bundled corpus: three short, original technical manuals stored as page-marked layout JSON."""

from __future__ import annotations

from pathlib import Path

from .chunking import Chunk, chunk_document
from .layout import Document, load_layout_json

DATA = Path(__file__).resolve().parents[2] / "data"
MANUALS = DATA / "manuals"

# Product names a question may mention, mapped to the document that covers them (used as a metadata filter).
ALIASES = {"h-200": "h200", "lx-40": "lx40", "tg-7": "tg7"}


def load_documents(folder: Path = MANUALS) -> list[Document]:
    return [load_layout_json(p) for p in sorted(folder.glob("*.json"))]


def load_chunks(folder: Path = MANUALS) -> list[Chunk]:
    return [c for doc in load_documents(folder) for c in chunk_document(doc)]


def page_index(docs: list[Document]) -> set[tuple[str, int]]:
    """Every (doc_id, page) that exists, so a citation can be checked to resolve."""
    return {(d.doc_id, p.number) for d in docs for p in d.pages}
