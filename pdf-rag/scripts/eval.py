"""Score the pipeline on the golden set: uv run python scripts/eval.py [--mode naive] [--json]"""

import sys

from pdfrag.cli import main

if __name__ == "__main__":
    sys.exit(main(["eval", *sys.argv[1:]]))
