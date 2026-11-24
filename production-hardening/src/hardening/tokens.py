"""Token counting and credit accounting.

Credits are a unit of account computed only from token counts. They are not money:
change RATES to match the ratios of whatever models you run.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_TOKEN = re.compile(r"\w+|[^\w\s]")


def count_tokens(text: str) -> int:
    """A stable, offline stand-in for a tokenizer: words and punctuation marks."""
    return len(_TOKEN.findall(text or ""))


# Credits per 1,000 tokens, by model tier. Cached input is billed at a tenth of fresh input.
RATES: dict[str, dict[str, float]] = {
    "small": {"input": 2.0, "cached_input": 0.2, "output": 8.0},
    "large": {"input": 10.0, "cached_input": 1.0, "output": 40.0},
}


@dataclass
class Usage:
    input_tokens: int = 0          # all prompt tokens, cached ones included
    cached_input_tokens: int = 0   # the part of input_tokens served from the prefix cache
    output_tokens: int = 0


def credits(tier: str, usage: Usage) -> float:
    rate = RATES[tier]
    fresh = usage.input_tokens - usage.cached_input_tokens
    total = (fresh * rate["input"]
             + usage.cached_input_tokens * rate["cached_input"]
             + usage.output_tokens * rate["output"])
    return round(total / 1000, 6)
