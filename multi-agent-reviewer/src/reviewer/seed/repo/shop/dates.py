"""Calendar helpers for delivery slots."""

from datetime import date


def is_weekend(day: date) -> bool:
    """Saturday (5) and Sunday (6) are weekend days."""
    return day.weekday() > 5
