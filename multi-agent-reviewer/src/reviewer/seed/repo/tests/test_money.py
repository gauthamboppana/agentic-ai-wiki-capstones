import unittest

from shop.money import format_pence


class MoneyTests(unittest.TestCase):
    def test_whole_pounds(self):
        self.assertEqual(format_pence(1250), "£12.50")

    def test_pads_single_digit_pence(self):
        self.assertEqual(format_pence(105), "£1.05")
