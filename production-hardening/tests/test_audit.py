"""The append-only, hash-chained audit log."""
import json

from hardening.agent import Agent, Config
from hardening.audit import AuditLog, verify_file
from hardening.cli import main
from hardening.tools import Workspace, load_dataset


def _log(tmp_path):
    log = AuditLog(tmp_path / "audit.jsonl")
    agent = Agent(Config(), audit=log)
    for case in load_dataset()["injections"][:4]:
        agent.run(Workspace(case))
    return log


def _rewrite(path, fn):
    lines = path.read_text().splitlines()
    path.write_text("\n".join(fn(lines)) + "\n")


def test_log_verifies_and_records_what_matters(tmp_path):
    log = _log(tmp_path)
    v = verify_file(log.path, log.head)
    assert v.ok and v.entries == len(log.entries) > 10
    kinds = {e["kind"] for e in log.entries}
    assert {"reasoning", "mutation", "injection_flagged"} <= kinds
    assert load_dataset()["secret"] not in log.path.read_text()
    assert main(["audit", "verify", str(log.path)]) == 0


def test_editing_an_entry_is_detected(tmp_path):
    log = _log(tmp_path)

    def edit(lines):
        e = json.loads(lines[5])
        e["data"]["summary"] = "nothing to see here"
        return lines[:5] + [json.dumps(e, sort_keys=True)] + lines[6:]
    _rewrite(log.path, edit)
    v = verify_file(log.path)
    assert not v.ok and v.bad_seq == 5
    assert main(["audit", "verify", str(log.path)]) == 1


def test_deleting_or_reordering_is_detected(tmp_path):
    log = _log(tmp_path)
    _rewrite(log.path, lambda ls: ls[:3] + ls[4:])
    assert verify_file(log.path).bad_seq == 3
    log2 = _log(tmp_path / "b")
    _rewrite(log2.path, lambda ls: [ls[1], ls[0]] + ls[2:])
    assert not verify_file(log2.path).ok


def test_truncation_is_caught_with_an_anchored_head(tmp_path):
    log = _log(tmp_path)
    head = log.head
    _rewrite(log.path, lambda ls: ls[:-2])
    assert verify_file(log.path).ok            # a shorter chain is still a chain
    assert not verify_file(log.path, head).ok  # but not the one we anchored


def test_reopening_continues_the_chain(tmp_path):
    log = _log(tmp_path)
    again = AuditLog(log.path)
    again.append("note", text="second process")
    assert verify_file(log.path).ok
