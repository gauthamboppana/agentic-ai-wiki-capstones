"""Faithfulness judges: is each claim in an answer supported by the chunks it cites?

`ClaimSupportJudge` is deterministic and free. A claim is one sentence of the answer; it is supported
when every number in it appears in the cited text and most of its content terms do too. It cannot read
meaning (a negated claim with the same words passes), so it is calibrated against hand labels like any
judge. `ModelJudge` asks a model for a verdict instead; calibrate it the same way before trusting it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from .chunking import Chunk
from .models import ChatModel
from .text import citations, sentences, strip_citations, terms, tokens

JUDGE_PROMPT = (
    "You check one claim against evidence from a manual.\n"
    "Reply SUPPORTED if the evidence states everything in the claim, including every number.\n"
    "Reply UNSUPPORTED otherwise. Reply with that one word only."
)


@dataclass
class Verdict:
    claim: str
    cites: list[tuple[str, int]]
    supported: bool
    reason: str = ""


class Judge(Protocol):
    def supports(self, claim: str, evidence: str) -> tuple[bool, str]: ...


class ClaimSupportJudge:
    def __init__(self, min_coverage: float = 0.8):
        self.min_coverage = min_coverage

    def supports(self, claim: str, evidence: str) -> tuple[bool, str]:
        have_tokens, have_terms = set(tokens(evidence)), set(terms(evidence))
        numbers = [t for t in tokens(claim) if any(ch.isdigit() for ch in t)]
        missing = [n for n in numbers if n not in have_tokens]
        if missing:
            return False, f"number not in the cited text: {missing[0]}"
        claim_terms = set(terms(claim))
        if not claim_terms:
            return False, "no content to check"
        coverage = len(claim_terms & have_terms) / len(claim_terms)
        if coverage < self.min_coverage:
            return False, f"only {coverage:.0%} of the claim's terms are in the cited text"
        return True, ""


class ModelJudge:
    """A model as the judge. Parse strictly: anything but a clear SUPPORTED counts as unsupported."""

    def __init__(self, model: ChatModel):
        self.model = model

    def supports(self, claim: str, evidence: str) -> tuple[bool, str]:
        reply = self.model.complete(JUDGE_PROMPT, f"Claim: {claim}\n\nEvidence:\n{evidence}").strip().upper()
        word = reply.split()[0].strip(".:") if reply else ""
        return word == "SUPPORTED", "" if word == "SUPPORTED" else f"judge said {reply[:40]!r}"


def judge_answer(answer: str, contexts: list[Chunk], judge: Judge) -> list[Verdict]:
    """One verdict per sentence. A claim is checked only against the chunks on the pages it cites."""
    verdicts = []
    for s in sentences(answer):
        cites = citations(s)
        evidence = "\n".join(c.section + "\n" + c.text for c in contexts if (c.doc_id, c.page) in cites)
        if not evidence:
            verdicts.append(Verdict(strip_citations(s), cites, False, "cites no page in the context"))
            continue
        ok, why = judge.supports(strip_citations(s), evidence)
        verdicts.append(Verdict(strip_citations(s), cites, ok, why))
    return verdicts


def calibrate(judge: Judge, labelled: list[dict]) -> dict:
    """Agreement with hand labels, plus the two error kinds. Missed unsupported claims are the costly ones."""
    agree = false_pass = false_fail = 0
    for item in labelled:
        got, _ = judge.supports(item["claim"], item["evidence"])
        agree += got == item["supported"]
        false_pass += got and not item["supported"]
        false_fail += item["supported"] and not got
    n = len(labelled)
    return {"n": n, "agreement": agree / n, "false_pass": false_pass, "false_fail": false_fail}


def load_calibration(path: Path | None = None) -> list[dict]:
    from .golden import GOLDEN

    return json.loads((path or GOLDEN / "judge_calibration.json").read_text(encoding="utf-8"))
