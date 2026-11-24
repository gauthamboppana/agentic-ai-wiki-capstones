import unittest

from shop.cart import Cart
from shop.inventory import OutOfStock, Stock

PRICES = {"mug": 500, "pot": 1000}


class CartTests(unittest.TestCase):
    def test_subtotal(self):
        cart = Cart(Stock({"mug": 5}), PRICES)
        cart.add("mug", 2)
        self.assertEqual(cart.subtotal(), 1000)

    def test_discount_percent_applied_to_total(self):
        cart = Cart(Stock({"pot": 5}), PRICES, discount_percent=10)
        cart.add("pot", 1)
        self.assertEqual(cart.total(), 900)

    def test_can_buy_last_items_in_stock(self):
        stock = Stock({"mug": 3})
        cart = Cart(stock, PRICES)
        cart.add("mug", 3)
        self.assertEqual(stock.levels["mug"], 0)

    def test_cannot_oversell(self):
        cart = Cart(Stock({"mug": 1}), PRICES)
        with self.assertRaises(OutOfStock):
            cart.add("mug", 2)
