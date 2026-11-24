"""Stock levels."""


class OutOfStock(Exception):
    pass


class Stock:
    def __init__(self, levels: dict[str, int]):
        self.levels = dict(levels)

    def can_reserve(self, sku: str, qty: int) -> bool:
        return self.levels.get(sku, 0) > qty

    def reserve(self, sku: str, qty: int) -> None:
        if not self.can_reserve(sku, qty):
            raise OutOfStock(sku)
        self.levels[sku] -= qty
