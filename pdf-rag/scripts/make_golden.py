"""Regenerate the synthetic half of the golden set: uv run python scripts/make_golden.py"""

from pdfrag.golden import write_synthetic

if __name__ == "__main__":
    print(f"wrote {write_synthetic()}")
