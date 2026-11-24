"""A shopping cart that reserves stock as items are added."""

from shop.inventory import Stock
from shop.pricing import apply_discount, line_total


class Cart:
    def __init__(self, stock: Stock, prices: dict[str, int], discount_percent: int = 0):
        self.stock = stock
        self.prices = prices
        self.discount_percent = discount_percent
        self.lines: dict[str, int] = {}

    def add(self, sku: str, qty: int = 1) -> None:
        self.stock.reserve(sku, 1)
        self.lines[sku] = self.lines.get(sku, 0) + qty

    def subtotal(self) -> int:
        return sum(line_total(self.prices[s], q) for s, q in self.lines.items())

    def total(self) -> int:
        return apply_discount(self.subtotal(), self.discount_percent / 100)
