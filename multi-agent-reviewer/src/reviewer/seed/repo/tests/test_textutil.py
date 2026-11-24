import unittest

from shop.textutil import slugify


class SlugifyTests(unittest.TestCase):
    def test_simple_name(self):
        self.assertEqual(slugify("Tea Pot"), "tea-pot")

    def test_no_trailing_hyphen(self):
        self.assertEqual(slugify("Blue Mug!"), "blue-mug")
