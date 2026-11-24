"""Layout model: pages made of positioned blocks, and the reading-order rule that turns them into a sequence.

PDF extractors hand back blocks in drawing order, which is rarely reading order on a two-column page.
Every source (the bundled JSON manuals, pypdf, or a layout model you swap in) produces the same
`Page` objects, and `reading_order` decides the order a person would read them in.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Block:
    kind: str  # "heading", "text" or "table"
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 with y growing down the page
    text: str = ""
    level: int = 2  # heading level
    caption: str = ""  # tables only
    header: list[str] = field(default_factory=list)
    rows: list[list[str]] = field(default_factory=list)

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def width(self) -> float:
        return self.bbox[2] - self.bbox[0]


@dataclass
class Page:
    number: int  # 1-based, as printed
    width: float
    height: float
    blocks: list[Block]


@dataclass
class Document:
    doc_id: str
    title: str
    pages: list[Page]


def is_running_text(block: Block, page: Page, margin: float = 0.04) -> bool:
    """Running headers and footers sit in the top or bottom margin and repeat on every page."""
    return block.y0 < page.height * margin or block.y0 > page.height * (1 - margin)


def reading_order(page: Page) -> list[Block]:
    """Full-width blocks split the page into bands; inside a band read the left column, then the right."""
    body = [b for b in page.blocks if not is_running_text(b, page)]
    full = sorted((b for b in body if b.width > page.width * 0.6), key=lambda b: b.y0)
    columns = [b for b in body if b.width <= page.width * 0.6]
    ordered: list[Block] = []
    top = float("-inf")
    for separator in [*full, None]:
        bottom = separator.y0 if separator else float("inf")
        band = [b for b in columns if top <= b.y0 < bottom]
        left = sorted((b for b in band if b.x0 < page.width / 2), key=lambda b: b.y0)
        right = sorted((b for b in band if b.x0 >= page.width / 2), key=lambda b: b.y0)
        ordered += left + right
        if separator:
            ordered.append(separator)
            top = separator.y0
    return ordered


def load_layout_json(path: str | Path) -> Document:
    """Load one page-marked manual from the bundled corpus."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    pages = [
        Page(
            number=p["page"],
            width=raw["page_size"][0],
            height=raw["page_size"][1],
            blocks=[Block(**{**b, "bbox": tuple(b["bbox"])}) for b in p["blocks"]],
        )
        for p in raw["pages"]
    ]
    return Document(doc_id=raw["doc_id"], title=raw["title"], pages=pages)
