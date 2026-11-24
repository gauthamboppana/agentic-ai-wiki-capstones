from hardening.cli import main


def test_demo_reports_every_target(tmp_path, capsys):
    assert main(["demo", "--out", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    for needle in ("Injection tests blocked  : 16/16", "lower PASS", "Trace coverage           : 100%", "chain verified"):
        assert needle in out
    assert (tmp_path / "trace.json").exists() and (tmp_path / "audit.jsonl").exists()
