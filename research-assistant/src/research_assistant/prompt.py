"""M1: the prompt template. Variables are named, checked and filled in one place."""

from __future__ import annotations

import json
import string

from .schema import strict_json_schema

OUTPUT_RULES = [
    "Return one JSON object and nothing else: no prose, no code fences.",
    "key_findings: one to five short sentences, each a fact stated in the article.",
    "entities: people, organisations, places and dates named in the article, each with a type.",
    "confidence: a number from 0 to 1 for how well the article supports your findings.",
    "Do not invent facts. If the article is thin, return fewer findings and a lower confidence.",
]

SYSTEM = "You are a careful research assistant. You read one article and report what it says as JSON."

TEMPLATE = string.Template(
    """Read the article and extract structured findings.

<article id="$article_id" title="$title">
$article
</article>

Rules:
$rules

The JSON must match this schema:
$schema
"""
)

REPAIR_TEMPLATE = string.Template(
    """Your last reply could not be used: $error
Reply again with only the corrected JSON object for article "$article_id", matching the schema exactly."""
)


def build_prompt(article_id: str, title: str, article: str, rules: list[str] | None = None) -> str:
    """Fill the template. `substitute` raises KeyError if a variable is missing."""
    return TEMPLATE.substitute(
        article_id=article_id,
        title=title,
        article=article.strip(),
        rules="\n".join(f"- {r}" for r in (rules or OUTPUT_RULES)),
        schema=json.dumps(strict_json_schema()["schema"], indent=None),
    )


def build_repair_prompt(article_id: str, error: str) -> str:
    return REPAIR_TEMPLATE.substitute(article_id=article_id, error=error)
