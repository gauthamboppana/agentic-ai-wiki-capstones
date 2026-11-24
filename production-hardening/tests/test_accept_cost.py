"""Acceptance: credits per resolved issue are at least 35% lower with hardening on."""
from hardening.agent import Config
from hardening.evals import run_suite


def test_cost_per_resolved_issue_is_at_least_35_percent_lower(hardened, unhardened):
    after, before = hardened[0], unhardened[0]
    assert after.resolved == before.resolved == after.issues     # same set, nothing lost
    saving = 1 - after.credits_per_resolved / before.credits_per_resolved
    assert saving >= 0.35, saving


def test_each_cost_lever_pulls_its_weight(hardened):
    full = hardened[0].credits_per_resolved
    for lever in ("prompt_cache", "semantic_cache", "cascade"):
        assert run_suite(Config().without(lever))[0].credits_per_resolved > full, lever


def test_the_cascade_escalates_unsure_reads(hardened):
    _, agent = hardened
    large = agent.tracer.find("model.call", lambda s: s.attributes["model.tier"] == "large")
    assert 1 <= len(large) <= 6
    assert all(s.attributes["model.task"] == "read" for s in large)


def test_caches_hit(hardened):
    _, agent = hardened
    hits = agent.tracer.find("cache.lookup", lambda s: s.attributes["cache.hit"])
    assert {s.attributes["cache.kind"] for s in hits} == {"exact", "semantic"}
    cached = sum(s.attributes["gen_ai.usage.cached_input_tokens"] for s in agent.tracer.find("model.call"))
    assert cached > 0
