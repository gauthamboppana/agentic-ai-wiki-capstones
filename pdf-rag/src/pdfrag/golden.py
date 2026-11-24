"""The 50-question golden set: synthetic questions generated from table rows, plus hand-written ones.

Each item names the evidence a good retriever must surface: (doc, page, quote) triples. An unanswerable
item has no evidence and expects "Not in the documents." Regenerate the synthetic half with
`uv run python scripts/make_golden.py`; the hand-written half lives in data/golden/handwritten.json.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from .corpus import DATA, load_documents
from .layout import Document

GOLDEN = DATA / "golden"
PRODUCT = {"h200": "H-200", "lx40": "LX-40", "tg7": "TG-7"}
SYNTHETIC_PER_DOC = 8


def synthesise(docs: list[Document], per_doc: int = SYNTHETIC_PER_DOC, seed: int = 7) -> list[dict]:
    """One question per sampled table row, asking for one cell. Seeded, so the file never drifts."""
    rng = random.Random(seed)
    items: list[dict] = []
    for doc in docs:
        tables = [(page.number, b) for page in doc.pages for b in page.blocks if b.kind == "table"]
        queues = [[(page, table, row) for row in rng.sample(table.rows, len(table.rows))] for page, table in tables]
        picked = []
        while len(picked) < per_doc:  # round robin over the tables, so every table is asked about
            for queue in queues:
                if queue and len(picked) < per_doc:
                    picked.append(queue.pop())
        for page, table, row in sorted(picked, key=lambda r: (r[0], r[2])):
            answer_cols = [i for i, h in enumerate(table.header) if i and h != "Notes"]
            col = answer_cols[len(items) % len(answer_cols)]  # rotate through the answer columns
            subject = table.header[0].lower()
            items.append(
                {
                    "id": f"syn-{len(items) + 1:02d}",
                    "source": "synthetic",
                    "question": f"What is the {table.header[col].lower()} for {subject} {row[0]} on the {PRODUCT[doc.doc_id]}?",
                    "answerable": True,
                    "evidence": [{"doc": doc.doc_id, "page": page, "quote": f"{table.header[col]}: {row[col]}"}],
                }
            )
    return items


def load_golden(folder: Path = GOLDEN) -> list[dict]:
    items = []
    for name in ("synthetic.json", "handwritten.json"):
        items += json.loads((folder / name).read_text(encoding="utf-8"))
    return items


def write_synthetic(folder: Path = GOLDEN) -> Path:
    path = folder / "synthetic.json"
    path.write_text(json.dumps(synthesise(load_documents()), indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
