"""M5 and M6: run the test set and score it against the acceptance targets."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from .data import load_articles
from .pipeline import Event, ResearchError, ResearchResult, analyse
from .providers import Provider

TTFT_TARGET_S = 1.5


def percentile(values: list[float], p: float) -> float:
    """Nearest-rank percentile: the smallest value with at least p per cent of values at or below it."""
    if not values:
        return float("nan")
    ordered = sorted(values)
    return ordered[max(0, math.ceil(p / 100 * len(ordered)) - 1)]


@dataclass
class Report:
    results: dict[str, ResearchResult] = field(default_factory=dict)
    failures: dict[str, ResearchError] = field(default_factory=dict)
    recall: dict[str, float] = field(default_factory=dict)

    @property
    def total(self) -> int:
        return len(self.results) + len(self.failures)

    @property
    def valid(self) -> int:
        return len(self.results)

    @property
    def first_try(self) -> int:
        return sum(1 for r in self.results.values() if r.retries == 0)

    @property
    def repaired(self) -> list[str]:
        return [k for k, r in self.results.items() if r.retries > 0]

    @property
    def ttfts(self) -> list[float]:
        every = [*(r.ttft_s for r in self.results.values()), *(e.attempts[0].ttft_s for e in self.failures.values())]
        return [t for t in every if t is not None]

    @property
    def ttft_p90(self) -> float:
        return percentile(self.ttfts, 90)

    def render(self) -> str:
        p50, p90 = percentile(self.ttfts, 50), self.ttft_p90
        mean_recall = sum(self.recall.values()) / max(1, len(self.recall))
        lines = [
            f"Schema-valid outputs : {self.valid}/{self.total} (first try {self.first_try}, "
            f"repaired {len(self.repaired)}: {', '.join(self.repaired) or 'none'})",
            f"Repair retries used  : {sum(r.retries for r in self.results.values())}",
            f"Time to first token  : p50 {p50 * 1000:.0f} ms, p90 {p90 * 1000:.0f} ms "
            f"(target under {TTFT_TARGET_S * 1000:.0f} ms) {'PASS' if p90 < TTFT_TARGET_S else 'FAIL'}",
            f"Entity recall        : {mean_recall:.0%} of expected names found",
        ]
        for aid, err in self.failures.items():
            lines.append(f"FAILED {aid}: {err}")
        return "\n".join(lines)


def entity_recall(result: ResearchResult, expected: Iterable[str]) -> float:
    expected = [e.lower() for e in expected]
    found = {e.name.lower() for e in result.findings.entities}
    return sum(1 for e in expected if e in found) / max(1, len(expected))


def score(provider: Provider, articles: Iterable[dict] | None = None,
          on_event: Callable[[str, Event], None] | None = None) -> Report:
    report = Report()
    for article in load_articles() if articles is None else articles:
        hook = (lambda ev, a=article["id"]: on_event(a, ev)) if on_event else None
        try:
            result = analyse(article, provider, on_event=hook)
        except ResearchError as err:
            report.failures[article["id"]] = err
            continue
        report.results[article["id"]] = result
        report.recall[article["id"]] = entity_recall(result, article.get("expected_entities", []))
    return report
