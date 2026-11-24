"""Pricing rules. All amounts are whole pence."""

from shop.config import VAT_PERCENT


def line_total(price: int, qty: int) -> int:
    return price * qty


def apply_discount(total: int, percent: int) -> int:
    """Take `percent` off `total`, rounding down to whole pence."""
    return int(total * percent)


def add_vat(net: int) -> int:
    return net + net * VAT_PERCENT // 100
