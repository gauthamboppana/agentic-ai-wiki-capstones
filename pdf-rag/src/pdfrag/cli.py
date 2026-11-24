"""Command line: `python -m pdfrag.cli demo | eval | ask "question"`."""

from __future__ import annotations

import argparse
import json
import sys

from .corpus import load_chunks
from .evaluate import TARGETS, evaluate
from .golden import load_golden
from .models import get_model
from .pipeline import Pipeline

LABELS = {
    "context_recall": "Context recall",
    "faithfulness": "Faithfulness",
    "citations": "Answers citing a page that resolves",
    "not_found": "Unanswerable questions refused",
    "answered": "Answerable questions answered",
    "answer_match": "Answers containing the expected value",
}


def print_report(metrics: dict[str, float], title: str) -> None:
    print(title)
    for key, value in metrics.items():
        target = TARGETS.get(key)
        mark = "" if target is None else ("  PASS" if value >= target else "  FAIL") + f" (target {target:.2f})"
        print(f"  {LABELS[key]:<40} {value:.2f}{mark}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pdfrag")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("demo", help="answer three questions, then score hybrid and naive retrieval on the golden set")
    ev = sub.add_parser("eval", help="score the pipeline on the golden set")
    ev.add_argument("--mode", choices=["hybrid", "naive"], default="hybrid")
    ev.add_argument("--json", action="store_true")
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    args = parser.parse_args(argv)

    chunks = load_chunks()
    if args.command == "ask":
        a = Pipeline(chunks, model=get_model()).answer(args.question)
        print(a.text)
        print(f"  ({a.reason or 'cites ' + ', '.join(f'{d} p.{p}' for d, p in a.citations)})")
        return 0
    if args.command == "eval":
        report = evaluate(Pipeline(chunks, mode=args.mode, model=get_model()))
        if args.json:
            print(json.dumps(report.metrics(), indent=1))
        else:
            print_report(report.metrics(), f"{args.mode} pipeline on {len(report.rows)} golden questions")
        return 0 if report.passed() or args.mode == "naive" else 1

    golden = load_golden()
    hybrid = Pipeline(chunks, model=get_model())
    for q in ("What is the factory IP address of the LX-40?", "What should I do when the H-200 shows fault E02?", "Does the TG-7 support LoRaWAN sensors?"):
        a = hybrid.answer(q)
        print(f"Q: {q}\nA: {a.text}\n")
    synthetic = sum(i["source"] == "synthetic" for i in golden)
    unanswerable = sum(not i["answerable"] for i in golden)
    print(f"Golden set: {len(golden)} questions ({synthetic} synthetic, {len(golden) - synthetic} hand-written, {unanswerable} unanswerable)\n")
    report = evaluate(hybrid, golden)
    print_report(report.metrics(), "Hybrid: BM25 + dense, RRF, rerank, abstain gate")
    print_report(evaluate(Pipeline(chunks, mode="naive", model=get_model()), golden).metrics(), "\nNaive: dense top 5, no rerank, no abstain gate")
    print("\nAll acceptance targets met." if report.passed() else "\nSome acceptance targets missed.")
    return 0 if report.passed() else 1


if __name__ == "__main__":
    sys.exit(main())
