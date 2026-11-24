"""Acceptance 2: time to first token is logged per request and the p90 is under 1.5 s."""

from research_assistant.providers import FakeModel
from research_assistant.score import TTFT_TARGET_S, percentile, score


def test_ttft_p90_under_target():
    report = score(FakeModel(first_token_delay=0.02))
    assert len(report.ttfts) == 20
    assert all(t > 0 for t in report.ttfts)
    print(f"TTFT p90 {report.ttft_p90 * 1000:.0f} ms")
    assert report.ttft_p90 < TTFT_TARGET_S


def test_ttft_is_first_chunk_not_whole_reply():
    # A slow stream with a fast first chunk: TTFT must stay small even though the whole reply takes longer.
    report = score(FakeModel(first_token_delay=0.01, chunk_delay=0.005, chunk_size=40))
    some = next(iter(report.results.values()))
    assert some.ttft_s < some.attempts[0].total_s
    assert percentile(report.ttfts, 90) < 0.5


def test_slow_provider_would_fail_the_target():
    report = score(FakeModel(first_token_delay=0.0), [])
    assert report.ttfts == []
    slow = score(FakeModel(first_token_delay=1.6, faults={}), [{"id": "z1", "title": "t", "text": "One. Two."}])
    assert slow.ttft_p90 >= TTFT_TARGET_S
