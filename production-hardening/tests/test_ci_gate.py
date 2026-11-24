"""The CI gate passes on the current build and fails on an injected regression."""
import subprocess
import sys
from pathlib import Path

import pytest

from hardening.cli import main
from hardening.evals import gate, load_baseline

ROOT = Path(__file__).resolve().parents[1]


def test_gate_passes_on_the_current_build():
    assert main(["gate"]) == 0


@pytest.mark.parametrize("feature", ["dual-llm", "guardrails", "cascade", "prompt-cache"])
def test_gate_fails_when_a_hardening_feature_regresses(feature, capsys):
    assert main(["gate", "--regress", feature]) == 1
    assert "REGRESSION" in capsys.readouterr().out


def test_gate_rules_on_numbers():
    base = load_baseline()
    assert gate(base, base) == []
    worse = {**base, "credits_per_resolved": base["credits_per_resolved"] * 1.2}
    assert any("credits_per_resolved" in p for p in gate(worse, base))
    assert gate({**base, "injection_block_rate": 0.9375}, base)


def test_ci_script_exit_codes():
    ok = subprocess.run([sys.executable, "scripts/ci_gate.py"], cwd=ROOT, capture_output=True, text=True)
    bad = subprocess.run([sys.executable, "scripts/ci_gate.py", "--regress", "dual-llm"], cwd=ROOT,
                         capture_output=True, text=True)
    assert ok.returncode == 0 and "gate: PASS" in ok.stdout
    assert bad.returncode == 1 and "gate: FAIL" in bad.stdout


def test_sample_workflow_runs_the_gate():
    wf = (ROOT / ".github" / "workflows" / "eval-gate.yml").read_text()
    assert "scripts/ci_gate.py" in wf and "uv run pytest" in wf
