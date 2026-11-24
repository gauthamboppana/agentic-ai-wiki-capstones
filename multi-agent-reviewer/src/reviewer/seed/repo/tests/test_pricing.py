import unittest

from shop.pricing import add_vat, line_total


class PricingTests(unittest.TestCase):
    def test_line_total(self):
        self.assertEqual(line_total(250, 4), 1000)

    def test_add_vat(self):
        self.assertEqual(add_vat(1000), 1200)
