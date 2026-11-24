"""Command line: run, resume, audit and an end-to-end demo that prints the acceptance numbers.

    uv run python -m reviewer.cli demo
    uv run python -m reviewer.cli run ISSUE-3 --workdir /tmp/rv --approve prompt
    uv run python -m reviewer.cli run ISSUE-3 --workdir /tmp/rv --crash-after writer
    uv run python -m reviewer.cli resume ISSUE-3-ab12cd --workdir /tmp/rv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from reviewer.app import Reviewer, seed_issues
from reviewer.approval import AutoApprove, CliPrompt, Decision, Defer, RejectAll
from reviewer.model import ScriptedModel

POLICIES = {"auto": AutoApprove, "prompt": CliPrompt, "defer": Defer, "reject": RejectAll}


def show(result) -> None:
    detail = result.state.get("stop_detail", "")
    print(f"run {result.run_id}: stop reason = {result.stop_reason}" + (f" ({detail})" if detail else ""))
    print(f"  path: {' -> '.join(result.state.get('history', []))}")
    if result.interrupt:
        print(result.interrupt["diff"])
        print(f"  resume with: resume {result.run_id} --decision approve|reject")


def demo() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        rv = Reviewer(tmp / "main", ScriptedModel())
        print(f"Seeded suite before: {len(rv.suite()['failed'])} failing tests")
        resolved = 0
        for issue in seed_issues():
            r = rv.run(issue["id"])
            suite = rv.suite()
            recalled = r.state.get("plan", {}).get("recalled", [])
            print(f"{issue['id']}: {r.stop_reason:<9} nodes={r.steps:<2} attempts={r.state['attempt'] + 1} "
                  f"files={','.join(r.state.get('written', []))} suite passing={len(suite['passed'])}/"
                  f"{len(suite['passed']) + len(suite['failed'])}" + (f" recalled={','.join(recalled)}" if recalled else ""))
            resolved += r.stop_reason == "resolved"
        report = rv.audit()

        # Resume: kill a real process after the writer node, then resume in this one.
        work = tmp / "resume"
        killed = subprocess.run([sys.executable, "-m", "reviewer.cli", "run", "ISSUE-4", "--workdir", str(work),
                                 "--crash-after", "writer", "--approve", "auto"], capture_output=True, text=True,
                                stdin=subprocess.DEVNULL)
        crashed = Reviewer(work, ScriptedModel())
        run_id = crashed.checkpoints.runs()[-1][0]
        before = crashed.checkpoints.latest(run_id)
        resumed = crashed.resume(run_id)

        # Guard: a fixer that undoes its own edit forever.
        loop = {"ISSUE-1": {"files": ["shop/textutil.py"], "finding": "?", "lesson": "", "cycle": True, "attempts": [
            [{"path": "shop/textutil.py", "find": 'lstrip("-")', "replace": 'lstrip("-" )'}],
            [{"path": "shop/textutil.py", "find": 'lstrip("-" )', "replace": 'lstrip("-")'}]]}}
        guard = Reviewer(tmp / "guard", ScriptedModel(loop), max_steps=12).run("ISSUE-1")

        rejected = Reviewer(tmp / "reject", ScriptedModel(), RejectAll()).run("ISSUE-2")
        rej_audit = Reviewer(tmp / "reject", ScriptedModel()).audit()

        print("\nAcceptance report")
        print(f"  resolved issues:     {resolved}/{len(seed_issues())} (multi-file: ISSUE-3, ISSUE-4)")
        print(f"  unauthorised writes: {report['unauthorised'] + rej_audit['unauthorised']} "
              f"({report['writes']} writes, {report['approvals']} approvals; rejected run wrote {rej_audit['writes']})")
        print(f"  resume:              killed with exit code {killed.returncode} after '{before.node}', "
              f"resumed at '{before.next_node}', stop reason = {resumed.stop_reason}")
        print(f"  max-iterations:      stop reason = {guard.stop_reason} ({guard.state['stop_detail']})")
        ok = (resolved == len(seed_issues()) and report["unauthorised"] == 0 and rej_audit["unauthorised"] == 0
              and resumed.stop_reason == "resolved" and guard.stop_reason == "max_iterations")
        print("ALL TARGETS MET" if ok else "SOME TARGETS MISSED")
        return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="reviewer")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("demo", help="run everything on a temporary copy and print the acceptance report")
    r = sub.add_parser("run", help="work one issue")
    r.add_argument("issue")
    r.add_argument("--workdir", type=Path, required=True)
    r.add_argument("--approve", choices=sorted(POLICIES), default="prompt")
    r.add_argument("--crash-after", help="kill the process right after this node's checkpoint")
    r.add_argument("--max-steps", type=int, default=20)
    s = sub.add_parser("resume", help="continue a run from its last checkpoint")
    s.add_argument("run_id")
    s.add_argument("--workdir", type=Path, required=True)
    s.add_argument("--decision", choices=["approve", "reject"])
    s.add_argument("--approve", choices=sorted(POLICIES), default="prompt")
    for name in ("audit", "runs", "issues"):
        sub.add_parser(name).add_argument("--workdir", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "demo":
        return demo()
    if args.cmd == "run":
        rv = Reviewer(args.workdir, policy=POLICIES[args.approve](), max_steps=args.max_steps)
        show(rv.run(args.issue, crash_after=args.crash_after, hard_crash=True))
    elif args.cmd == "resume":
        rv = Reviewer(args.workdir, policy=POLICIES[args.approve]())
        decision = Decision(args.decision, by="human (cli flag)") if args.decision else None
        show(rv.resume(args.run_id, decision))
    elif args.cmd == "audit":
        print(json.dumps(Reviewer(args.workdir).audit(), indent=2))
    elif args.cmd == "runs":
        for run_id, issue_id, status in Reviewer(args.workdir).checkpoints.runs():
            print(f"{run_id}  {issue_id}  {status}")
    elif args.cmd == "issues":
        for issue in seed_issues():
            print(f"{issue['id']}  {issue['title']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
