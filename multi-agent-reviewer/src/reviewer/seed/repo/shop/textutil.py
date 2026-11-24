"""Text helpers."""

import re


def slugify(name: str) -> str:
    """Turn a product name into a URL slug: 'Blue Mug!' -> 'blue-mug'."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower())
    return slug.lstrip("-")
