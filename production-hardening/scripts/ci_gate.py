"""CI entry point: run the trajectory eval suite and exit non-zero on any regression.

    uv run python scripts/ci_gate.py                      # PASS, exit 0
    uv run python scripts/ci_gate.py --regress dual-llm   # an injected regression, exit 1
"""
import sys

from hardening.cli import main

if __name__ == "__main__":
    sys.exit(main(["gate", *sys.argv[1:]]))
