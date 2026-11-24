"""Optional: turn a real PDF into the same `Document` layout model, using pypdf (`uv sync --extra pdf`).

pypdf reports each text run with its position and font size. We group runs into lines, lines into
blocks, mark large-font blocks as headings and let `reading_order` sort columns. pypdf does not find
tables; for table-heavy PDFs swap in a layout model or pdfplumber and emit `Block(kind="table")`.
"""

from __future__ import annotations

from pathlib import Path
from statistics import median

from .layout import Block, Document, Page


def _runs(page) -> list[tuple[float, float, float, str]]:
    """(x, y_from_top, font_size, text) for every non-empty text run on a pypdf page."""
    height = float(page.mediabox.height)
    runs: list[tuple[float, float, float, str]] = []

    def visit(text, cm, tm, font_dict, font_size):
        if not text.strip():
            return
        x = tm[4] * cm[0] + tm[5] * cm[2] + cm[4]
        y = tm[4] * cm[1] + tm[5] * cm[3] + cm[5]
        size = abs(font_size * (tm[3] or 1) * (cm[3] or 1))
        runs.append((x, height - y, size, text.strip()))

    page.extract_text(visitor_text=visit)
    return runs


def blocks_from_runs(runs: list[tuple[float, float, float, str]], width: float) -> list[Block]:
    if not runs:
        return []
    body = median(r[2] for r in runs)
    # 1. runs on the same baseline and the same side of the page form a line
    lines: dict[tuple[int, int], list[tuple[float, float, float, str]]] = {}
    for x, y, size, text in runs:
        lines.setdefault((round(y), 0 if x < width / 2 else 1), []).append((x, y, size, text))
    merged = []
    for (_, side), parts in sorted(lines.items()):
        parts.sort()
        x0, y0, size = parts[0][0], parts[0][1], max(p[2] for p in parts)
        text = " ".join(p[3] for p in parts)
        x1 = max(p[0] for p in parts) + len(parts[-1][3]) * size * 0.5
        merged.append((side, x0, y0, x1, size, text))
    # 2. consecutive lines in one column with a small gap and the same style form a block
    blocks: list[Block] = []
    for side in (0, 1):
        column = sorted((m for m in merged if m[0] == side), key=lambda m: m[2])
        current: list = []
        for line in column:
            heading = line[4] >= body * 1.15
            if current:
                prev = current[-1]
                same_style = (prev[4] >= body * 1.15) == heading
                if same_style and not heading and line[2] - prev[2] <= prev[4] * 1.8:
                    current.append(line)
                    continue
                blocks.append(_block(current, body))
            current = [line]
        if current:
            blocks.append(_block(current, body))
    return blocks


def _block(lines: list, body: float) -> Block:
    size = lines[0][4]
    bbox = (min(l[1] for l in lines), lines[0][2], max(l[3] for l in lines), lines[-1][2] + size)
    text = " ".join(l[5] for l in lines)
    if size >= body * 1.15:
        return Block(kind="heading", bbox=bbox, text=text, level=1 if size >= body * 1.6 else 2)
    return Block(kind="text", bbox=bbox, text=text)


def load_pdf(path: str | Path, doc_id: str | None = None) -> Document:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on the optional extra
        raise RuntimeError("PDF support needs the optional extra: uv sync --extra pdf") from exc
    reader = PdfReader(str(path))
    pages = []
    for number, page in enumerate(reader.pages, start=1):
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        pages.append(Page(number=number, width=width, height=height, blocks=blocks_from_runs(_runs(page), width)))
    path = Path(path)
    title = (reader.metadata.title if reader.metadata and reader.metadata.title else path.stem)
    return Document(doc_id=doc_id or path.stem.lower(), title=title, pages=pages)
