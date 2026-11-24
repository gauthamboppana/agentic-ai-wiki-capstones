import unittest
from datetime import date

from shop.dates import is_weekend


class DateTests(unittest.TestCase):
    def test_sunday_is_weekend(self):
        self.assertTrue(is_weekend(date(2026, 11, 22)))

    def test_saturday_is_weekend(self):
        self.assertTrue(is_weekend(date(2026, 11, 21)))

    def test_monday_is_not_weekend(self):
        self.assertFalse(is_weekend(date(2026, 11, 23)))
