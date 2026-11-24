"""Input and output guardrails.

- redact_pii: emails, phone numbers, card numbers (Luhn-checked), UK National Insurance numbers.
- detect_injection: heuristics for indirect prompt injection in tool results.
- check_output: what may leave the agent in a mutation (no secrets, only allowlisted links).
- ToolPolicy: which tools a task may call, and on which issue.

Heuristics are a tripwire, not a wall. The structural defence is the dual-LLM split in
dual_llm.py: the planner never reads untrusted text, so a missed injection has nothing to steer.
"""
from __future__ import annotations

import base64
import re
from dataclasses import dataclass, field
from typing import Any

from .model import URL, ZERO_WIDTH

EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
PHONE = re.compile(r"(?<![\w-])\+?\d[\d ()-]{8,}\d\b")
CARD = re.compile(r"\b(?:\d[ -]?){13,19}\b")
NINO = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b")


def _luhn(digits: str) -> bool:
    total, alt = 0, False
    for d in reversed(digits):
        n = int(d)
        if alt:
            n = n * 2 - 9 if n > 4 else n * 2
        total, alt = total + n, not alt
    return total % 10 == 0


def redact_pii(text: str) -> tuple[str, dict[str, int]]:
    counts: dict[str, int] = {}

    def sub(kind: str, pattern: re.Pattern[str], s: str, check=lambda m: True) -> str:
        def repl(m: re.Match[str]) -> str:
            if not check(m):
                return m.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            return f"[{kind}]"
        return pattern.sub(repl, s)

    text = sub("CARD", CARD, text, lambda m: _luhn(re.sub(r"\D", "", m.group(0))))
    text = sub("EMAIL", EMAIL, text)
    text = sub("NINO", NINO, text)
    text = sub("PHONE", PHONE, text)
    return text, counts


INJECTION_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("override", re.compile(r"(?i)\b(ignore|disregard|forget)\b.{0,30}\b(instructions|rules|prompt|everything)\b")),
    ("override", re.compile(r"(?i)\bignorez\b.{0,30}\binstructions\b")),
    ("override", re.compile(r"(?i)\b(new|updated) (policy|instructions?)\b|\byou are now\b|\bmaintenance mode\b")),
    ("role", re.compile(r"(?im)^\s*(system|assistant|developer)\s*:|<\|im_start\|>|\[/?INST\]")),
    ("role", re.compile(r"(?i)\b(note|message) to the (ai|assistant|agent|model)\b|\bassistant\s*,|\bagents? (processing|summari[sz]ing|reading)\b")),
    ("role", re.compile(r"(?i)\btool result\s*:|\bnext step required\b|\bbefore replying\b")),
    ("exfil", re.compile(r"!\[[^\]]*\]\(https?://")),
    ("exfil", re.compile(r"(?i)\b(escalation|internal|secret) code\b|\bapi key\b")),
    ("exfil", re.compile(r"(?is)\b(include|add|append|embed)\b.{0,80}\b(reply|response|answer)\b.{0,120}https?://"
                         r"|https?://\S+.{0,80}\b(to|in) (your|the|their) (reply|response|answer)\b")),
    ("hidden", re.compile(r"(?is)<!--.*?-->|display\s*:\s*none|visibility\s*:\s*hidden")),
]


@dataclass
class InjectionVerdict:
    flagged: bool
    reasons: list[str] = field(default_factory=list)


def detect_injection(text: str, tool_names: list[str]) -> InjectionVerdict:
    reasons: list[str] = []
    if ZERO_WIDTH.search(text):
        reasons.append("hidden:zero-width")
    clean = ZERO_WIDTH.sub("", text)
    for blob in re.findall(r"[A-Za-z0-9+/=]{24,}", clean):
        try:
            decoded = base64.b64decode(blob, validate=True).decode("utf-8")
        except Exception:
            continue
        reasons.append("encoded:base64")
        clean += "\n" + decoded
    for kind, rule in INJECTION_RULES:
        if rule.search(clean):
            reasons.append(kind)
    for name in tool_names:
        if re.search(rf"\b{re.escape(name)}\b", clean):
            reasons.append(f"tool-mention:{name}")
    return InjectionVerdict(bool(reasons), sorted(set(reasons)))


@dataclass
class OutputVerdict:
    allowed: bool
    text: str
    reasons: list[str] = field(default_factory=list)


def check_output(text: str, secrets: list[str], allowed_domains: list[str]) -> OutputVerdict:
    """Block secrets outright. Strip images and links to domains not on the allowlist. Redact PII."""
    reasons = [f"secret:{s[:7]}" for s in secrets if s in text]
    if reasons:
        return OutputVerdict(False, "", reasons)
    out = re.sub(r"!\[[^\]]*\]\([^)]*\)", lambda m: (reasons.append("image"), "")[1], text)

    def keep(m: re.Match[str]) -> str:
        host = re.sub(r"^https?://", "", m.group(0)).split("/")[0].lower()
        if any(host == d or host.endswith("." + d) for d in allowed_domains):
            return m.group(0)
        reasons.append(f"link:{host}")
        return "[link removed]"
    out = re.sub(r"\[([^\]]*)\]\((https?://[^)]+)\)", lambda m: m.group(1) + " " + m.group(2), out)
    out = URL.sub(keep, out)
    out, pii = redact_pii(out)
    reasons += [f"pii:{k}" for k in pii]
    return OutputVerdict(True, re.sub(r"\s{2,}", " ", out).strip(), reasons)


@dataclass
class ToolPolicy:
    """Least privilege for the issue-resolution task."""
    allowed: frozenset[str] = frozenset({"read_issue", "search_kb", "fetch_page", "add_label", "post_reply"})

    def check(self, name: str, args: dict[str, Any], issue_id: str) -> str | None:
        if name not in self.allowed:
            return f"tool {name} is not allowed for this task"
        if "issue_id" in args and args["issue_id"] != issue_id:
            return f"{name} may only act on {issue_id}"
        if name == "fetch_page" and not str(args.get("url", "")).startswith("https://docs.example.org/"):
            return "fetch_page is limited to the documentation site"
        return None
