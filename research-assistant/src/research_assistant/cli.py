"""Command line: `python -m research_assistant.cli demo | score | analyse <id or file> | serve`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import article_by_id
from .pipeline import ProgressLine, ResearchError, analyse
from .providers import ScriptedModel, get_provider
from .score import score

BROKEN_CASES = {
    "truncated, then fixed": ["{\"article_id\": \"a01\", \"key_findings\": [\"The turb", None],
    "refused, then fixed": ["I'm sorry, but I can't help with that request.", None],
    "refused every time": ["I'm sorry, but I can't help with that request."] * 3,
}


def _good_reply(article: dict) -> str:
    from .providers import extract
    return json.dumps(extract(article["id"], article["text"]))


def broken_cases() -> list[str]:
    """Feed three broken replies and show the retry or the clear error. Returns report lines."""
    article = article_by_id("a01")
    lines = []
    for label, script in BROKEN_CASES.items():
        model = ScriptedModel([r if r is not None else _good_reply(article) for r in script])
        try:
            result = analyse(article, model)
            lines.append(f"  {label:<22}: recovered after {result.retries} repair(s)")
        except ResearchError as err:
            lines.append(f"  {label:<22}: clear error, no crash: {err}")
    return lines


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="research_assistant")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="score all 20 articles and run the three broken cases")
    sub.add_parser("score", help="score the 20-article test set")
    a = sub.add_parser("analyse", help="analyse one bundled article id or a text file")
    a.add_argument("target")
    s = sub.add_parser("serve", help="serve the SSE endpoint on localhost")
    s.add_argument("--port", type=int, default=8000)
    args = ap.parse_args(argv)
    provider = get_provider()

    if args.cmd == "serve":
        from .sse import serve
        serve(port=args.port, provider=provider)
        return 0
    if args.cmd == "analyse":
        article = article_by_id(args.target)
        if article is None:
            path = Path(args.target)
            article = {"id": path.stem, "title": path.stem, "text": path.read_text(encoding="utf-8")}
        try:
            result = analyse(article, provider, on_event=ProgressLine(sys.stderr.write))
        except ResearchError as err:
            print(err, file=sys.stderr)
            return 1
        print(result.findings.model_dump_json(indent=2))
        return 0

    print(f"Provider: {provider.name}")
    progress = ProgressLine(sys.stderr.write)
    report = score(provider, on_event=lambda aid, ev: progress(ev) if ev["event"] != "progress" else None)
    print("\n== Acceptance report ==")
    print(report.render())
    if args.cmd == "demo":
        print("Failure handling     : three broken cases")
        print("\n".join(broken_cases()))
    return 0 if report.valid == report.total else 1


if __name__ == "__main__":
    raise SystemExit(main())
