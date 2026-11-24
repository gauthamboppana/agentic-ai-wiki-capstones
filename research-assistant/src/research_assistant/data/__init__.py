"""M5: the bundled 20-article test set (short original articles with expected entities)."""

from __future__ import annotations

import json
from functools import cache
from importlib import resources


@cache
def load_articles() -> tuple[dict, ...]:
    text = resources.files(__package__).joinpath("articles.json").read_text(encoding="utf-8")
    return tuple(json.loads(text))


def article_by_id(article_id: str) -> dict | None:
    return next((a for a in load_articles() if a["id"] == article_id), None)
