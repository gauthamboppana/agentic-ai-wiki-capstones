"""Formatting money held as whole pence."""

from shop.config import CURRENCY_SYMBOL


def format_pence(pence: int) -> str:
    """Format pence as pounds: 105 -> '£1.05'."""
    pounds, rest = divmod(pence, 100)
    return f"{CURRENCY_SYMBOL}{pounds}.{rest}"
