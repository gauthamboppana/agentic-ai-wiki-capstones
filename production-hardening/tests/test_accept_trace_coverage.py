"""Acceptance: every model and tool call has a span."""
import json

from hardening.agent import Agent, Config
from hardening.evals import trace_coverage
from hardening.model import ModelRequest
from hardening.tools import Workspace, load_dataset


def test_full_coverage_on_the_whole_suite(hardened):
    report, agent = hardened
    assert report.trace_coverage == 1.0
    assert len(agent.tracer.find("model.call")) == len(agent.small.calls) + len(agent.large.calls)


def test_model_spans_carry_usage_and_credits(hardened):
    _, agent = hardened
    for s in agent.tracer.find("model.call"):
        a = s.attributes
        assert a["gen_ai.request.model"] and a["gen_ai.usage.input_tokens"] > 0 and "credits" in a


def test_blocked_tool_calls_have_spans_too():
    agent = Agent(Config().without("dual_llm"))   # the planner sees the payload, the tool policy refuses it
    case = next(c for c in load_dataset()["injections"] if c["id"] == "ISS-112")
    agent.run(Workspace(case))
    assert agent.tracer.find("tool.call", lambda s: s.attributes.get("tool.blocked") is True)


def test_an_untraced_call_is_caught():
    agent = Agent(Config())
    ws = Workspace(load_dataset()["issues"][0])
    agent.run(ws)
    agent.small.complete(ModelRequest("x", [{"role": "user", "content": "Resolve issue ISS-001."}]))
    cov = trace_coverage(agent.tracer, [agent.small, agent.large], [ws], 0)
    assert cov["coverage"] < 1.0 and len(cov["untraced_model_calls"]) == 1


def test_otlp_export_is_one_tree_per_run(tmp_path):
    agent = Agent(Config())
    agent.run(Workspace(load_dataset()["issues"][6]))
    doc = json.loads(agent.tracer.write_otlp(tmp_path / "t.json").read_text())
    spans = doc["resourceSpans"][0]["scopeSpans"][0]["spans"]
    roots = [s for s in spans if not s["parentSpanId"]]
    assert [r["name"] for r in roots] == ["agent.run"]
    assert {s["traceId"] for s in spans} == {roots[0]["traceId"]}
