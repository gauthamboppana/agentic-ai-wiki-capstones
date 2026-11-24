"""M3 and M4: stream one article through the model, validate, repair up to twice, or fail clearly."""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field

from pydantic import ValidationError

from .prompt import SYSTEM, build_prompt, build_repair_prompt
from .providers import Message, Provider
from .schema import Findings

MAX_REPAIRS = 2
REFUSAL_MARKERS = ("i can't", "i cannot", "i'm sorry", "i am sorry", "unable to help", "i won't")

Event = dict  # {"event": name, "data": {...}}, always JSON-safe so it can go straight onto an SSE stream


@dataclass
class Attempt:
    n: int
    ttft_s: float | None
    total_s: float
    chars: int
    ok: bool
    error_kind: str | None = None
    error: str | None = None


@dataclass
class ResearchResult:
    findings: Findings
    attempts: list[Attempt]
    status: str = "ok"  # "ok", or "unsure" when confidence is under the threshold

    @property
    def retries(self) -> int:
        return len(self.attempts) - 1

    @property
    def ttft_s(self) -> float | None:
        """Time to first token of the first request: what the user waits before anything moves."""
        return self.attempts[0].ttft_s


class ResearchError(Exception):
    """Raised when every attempt failed. Carries the attempts so callers can report them."""

    def __init__(self, article_id: str, attempts: list[Attempt]):
        self.article_id = article_id
        self.attempts = attempts
        last = attempts[-1]
        super().__init__(
            f"Could not get valid findings for article {article_id!r} after {len(attempts)} attempts "
            f"(1 try and {len(attempts) - 1} repairs). Last problem: {last.error_kind}: {last.error}"
        )


def check(text: str, article_id: str) -> tuple[Findings | None, str | None, str | None]:
    """Strict check, no silent fixes: the reply must be one JSON object that matches the schema.

    Returns (findings, None, None) or (None, error_kind, message).
    """
    body = text.strip()
    if not body:
        return None, "empty", "the reply was empty"
    if not body.startswith("{"):
        if any(m in body.lower() for m in REFUSAL_MARKERS):
            return None, "refused", f"the model declined: {body[:80]!r}"
        return None, "not_json", f"the reply must start with '{{' but started with {body[:30]!r}"
    try:
        data = json.loads(body)
    except json.JSONDecodeError as e:
        cut_off = e.pos >= len(body) - 1 or e.msg.startswith("Unterminated") or body.count("{") > body.count("}")
        kind = "truncated" if cut_off else "not_json"
        return None, kind, f"invalid JSON at character {e.pos}: {e.msg}"
    try:
        findings = Findings.model_validate(data)
    except ValidationError as e:
        problems = "; ".join(f"{'.'.join(map(str, x['loc'])) or 'root'}: {x['msg']}" for x in e.errors()[:5])
        return None, "schema", problems
    if findings.article_id != article_id:
        return None, "schema", f"article_id is {findings.article_id!r}, expected {article_id!r}"
    return findings, None, None


def run(article: dict, provider: Provider, max_repairs: int = MAX_REPAIRS, unsure_below: float | None = None,
        clock: Callable[[], float] = time.perf_counter) -> Generator[Event, None, ResearchResult | ResearchError]:
    """Yields progress events; returns the result, or a ResearchError (it does not raise, so streams can close cleanly)."""
    aid = article["id"]
    messages: list[Message] = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": build_prompt(aid, article.get("title", ""), article["text"])},
    ]
    attempts: list[Attempt] = []
    for n in range(1, max_repairs + 2):
        yield {"event": "start", "data": {"article_id": aid, "attempt": n, "provider": provider.name}}
        start, ttft, parts = clock(), None, []
        try:
            for chunk in provider.stream(messages):
                if not chunk:
                    continue
                if ttft is None:
                    ttft = clock() - start
                parts.append(chunk)
                yield {"event": "token", "data": {"attempt": n, "text": chunk}}
                yield {"event": "progress", "data": {"attempt": n, "chars": sum(map(len, parts))}}
            text = "".join(parts)
            findings, kind, error = check(text, aid)
        except Exception as e:  # network errors and timeouts count as a failed attempt, not a crash
            text, findings, kind, error = "".join(parts), None, "provider_error", f"{type(e).__name__}: {e}"
        attempts.append(Attempt(n, ttft, clock() - start, len(text), findings is not None, kind, error))
        if findings is not None:
            status = "unsure" if unsure_below is not None and findings.confidence < unsure_below else "ok"
            result = ResearchResult(findings, attempts, status)
            yield {"event": "result", "data": {"status": status, "retries": result.retries,
                                               "ttft_ms": _ms(result.ttft_s), "findings": findings.model_dump()}}
            return result
        yield {"event": "invalid", "data": {"attempt": n, "kind": kind, "error": error,
                                            "will_retry": n <= max_repairs}}
        messages = messages + [
            {"role": "assistant", "content": text or "(empty reply)"},
            {"role": "user", "content": build_repair_prompt(aid, f"{kind}: {error}")},
        ]
    err = ResearchError(aid, attempts)
    yield {"event": "error", "data": {"article_id": aid, "message": str(err)}}
    return err


def analyse(article: dict, provider: Provider, on_event: Callable[[Event], None] | None = None,
            **kwargs) -> ResearchResult:
    """Run one article to completion. Raises ResearchError when the retries run out."""
    gen = run(article, provider, **kwargs)
    while True:
        try:
            event = next(gen)
        except StopIteration as stop:
            outcome = stop.value
            break
        if on_event:
            on_event(event)
    if isinstance(outcome, ResearchError):
        raise outcome
    return outcome


def _ms(seconds: float | None) -> int | None:
    return None if seconds is None else round(seconds * 1000)


class ProgressLine:
    """A one-line progress display for terminals: rewrites itself with a carriage return."""

    def __init__(self, write: Callable[[str], object]):
        self.write = write

    def __call__(self, event: Event) -> None:
        d = event["data"]
        if event["event"] == "progress":
            self.write(f"\r  attempt {d['attempt']} | {d['chars']:>4} chars received")
        elif event["event"] == "invalid":
            self.write(f"\r  attempt {d['attempt']} | {d['kind']}: {d['error'][:60]} | "
                       f"{'repairing' if d['will_retry'] else 'giving up'}\n")
        elif event["event"] == "result":
            self.write(f"\r  done | retries {d['retries']} | ttft {d['ttft_ms']} ms | "
                       f"confidence {d['findings']['confidence']}\n")
        elif event["event"] == "error":
            self.write(f"\r  failed | {d['message']}\n")
