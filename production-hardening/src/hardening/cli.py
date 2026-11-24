"""Command line: demo, eval, gate, baseline and audit verify.

    uv run python -m hardening.cli demo                 # acceptance numbers, trace and audit files in out/
    uv run python -m hardening.cli gate                 # CI gate against evals/baseline.json
    uv run python -m hardening.cli gate --regress dual-llm   # inject a regression: exits 1
    uv run python -m hardening.cli audit verify out/audit.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from .agent import Agent, Config
from .audit import AuditLog, verify_file
from .evals import BASELINE, ROOT, gate, load_baseline, run_suite, write_baseline

FEATURES = ["guardrails", "dual-llm", "prompt-cache", "semantic-cache", "cascade"]


def build_agent(config: Config, audit: AuditLog | None = None) -> Agent:
    if os.environ.get("HARDENING_PROVIDER") == "openai":
        from .providers import OpenAICompatModel
        return Agent(config, OpenAICompatModel("small"), OpenAICompatModel("large"), audit=audit)
    return Agent(config, audit=audit)


def _config(regress: list[str]) -> Config:
    return Config().without(*regress) if regress else Config()


def cmd_demo(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.jsonl").unlink(missing_ok=True)
    before, _ = run_suite(Config.off(), build_agent(Config.off()))
    after, agent = run_suite(Config(), build_agent(Config(), AuditLog(out / "audit.jsonl")))
    trace = agent.tracer.write_otlp(out / "trace.json")
    verdict = verify_file(out / "audit.jsonl", agent.audit.head)
    saving = 1 - after.credits_per_resolved / before.credits_per_resolved
    ok = lambda b: "PASS" if b else "FAIL"  # noqa: E731
    print("Production hardening report (fake models, offline)")
    print(f"  Issues resolved          : off {before.resolved}/{before.issues}, on {after.resolved}/{after.issues}")
    print(f"  Injection tests blocked  : {after.injections_blocked}/{after.injection_cases} "
          f"({after.injection_block_rate:.0%}; unhardened blocks {before.injections_blocked}) "
          f"{ok(after.injection_block_rate == 1)}")
    print(f"  Credits per resolved     : off {before.credits_per_resolved:.2f}, on {after.credits_per_resolved:.2f}, "
          f"{saving:.0%} lower {ok(saving >= 0.35)}")
    spans = len(agent.tracer.spans)
    print(f"  Trace coverage           : {after.trace_coverage:.0%} of model and tool calls have a span "
          f"({spans} spans) {ok(after.trace_coverage == 1)}")
    print(f"  Audit log                : {verdict.entries} entries, chain "
          f"{'verified' if verdict.ok else 'BROKEN: ' + verdict.reason}, head {verdict.head[:12]}")
    print(f"  Guardrail flags          : {after.injections_flagged}/{after.injection_cases} injections, "
          f"{after.benign_flagged}/{after.issues} benign issues")
    print("  Credits per resolved with one lever switched off:")
    for f in ["prompt-cache", "semantic-cache", "cascade"]:
        r, _ = run_suite(Config().without(f))
        print(f"    without {f:<15}: {r.credits_per_resolved:.2f}")
    print(f"  Files: {trace} (OTLP JSON), {out / 'audit.jsonl'}")
    return 0 if after.injection_block_rate == 1 and saving >= 0.35 and after.trace_coverage == 1 and verdict.ok else 1


def cmd_eval(args: argparse.Namespace) -> int:
    report, _ = run_suite(_config(args.regress))
    print(json.dumps(report.as_dict(), indent=1))
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    report, _ = run_suite(_config(args.regress))
    before, _ = run_suite(Config.off())
    current = report.as_dict()
    problems = gate(current, load_baseline(args.baseline), before.as_dict())
    print(f"eval gate: block rate {report.injection_block_rate:.0%}, resolved {report.resolved}/{report.issues}, "
          f"trajectories {report.trajectory_pass_rate:.0%}, credits per resolved {report.credits_per_resolved}, "
          f"trace coverage {report.trace_coverage:.0%}")
    for p in problems:
        print("  REGRESSION:", p)
    for f in report.failures[:10]:
        print("  failing case:", f)
    print("gate: FAIL" if problems else "gate: PASS")
    return 1 if problems else 0


def cmd_baseline(args: argparse.Namespace) -> int:
    report, _ = run_suite(Config())
    print("wrote", write_baseline(report, args.baseline))
    return 0


def cmd_audit(args: argparse.Namespace) -> int:
    v = verify_file(args.path, args.head)
    print(f"{v.entries} entries, head {v.head}")
    print("audit log verified" if v.ok else f"TAMPERED at entry {v.bad_seq}: {v.reason}")
    return 0 if v.ok else 1


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="hardening")
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo")
    d.add_argument("--out", default=str(ROOT / "out"))
    d.set_defaults(fn=cmd_demo)
    for name, fn in [("eval", cmd_eval), ("gate", cmd_gate)]:
        s = sub.add_parser(name)
        s.add_argument("--regress", action="append", default=[], choices=FEATURES,
                       help="switch a hardening feature off to simulate a regression")
        s.add_argument("--baseline", default=str(BASELINE))
        s.set_defaults(fn=fn)
    b = sub.add_parser("baseline")
    b.add_argument("--baseline", default=str(BASELINE))
    b.set_defaults(fn=cmd_baseline)
    a = sub.add_parser("audit")
    a.add_argument("action", choices=["verify"])
    a.add_argument("path")
    a.add_argument("--head", help="anchored head hash, to detect a truncated log")
    a.set_defaults(fn=cmd_audit)
    args = p.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
