"""Models. The default is a scripted, deterministic policy keyed on the issue, so runs repeat exactly.

Every model has one method: `decide(role, payload) -> dict`, where role is planner, investigator,
fixer or critic. An OpenAI-compatible HTTP adapter is included for real runs; tests never use it.
"""

from __future__ import annotations

import json
import os
import urllib.request

# One script per seeded issue: what the planner names, what the investigator concludes,
# and the fixer's edits for each attempt. ISSUE-3's first attempt is deliberately partial.
SCRIPTS: dict[str, dict] = {
    "ISSUE-1": {
        "files": ["shop/textutil.py"],
        "finding": "lstrip only removes leading hyphens; punctuation at the end leaves one behind.",
        "attempts": [[{"path": "shop/textutil.py", "find": 'return slug.lstrip("-")', "replace": 'return slug.strip("-")'}]],
        "lesson": "Trim separators from both ends of generated slugs.",
    },
    "ISSUE-2": {
        "files": ["shop/money.py"],
        "finding": "The pence remainder is printed without zero padding.",
        "attempts": [[{"path": "shop/money.py", "find": "{pounds}.{rest}", "replace": "{pounds}.{rest:02d}"}]],
        "lesson": "Pad pence to two digits when formatting money.",
    },
    "ISSUE-3": {
        "files": ["shop/pricing.py", "shop/cart.py"],
        "finding": "apply_discount multiplies by the percent instead of taking it off, and the cart divides by 100 first.",
        "attempts": [
            [{"path": "shop/pricing.py", "find": "return int(total * percent)",
              "replace": "return total * (100 - percent) // 100"}],
            [{"path": "shop/cart.py", "find": "apply_discount(self.subtotal(), self.discount_percent / 100)",
              "replace": "apply_discount(self.subtotal(), self.discount_percent)"}],
        ],
        "lesson": "Keep one unit for discounts (whole percent) across the cart and the pricing rules.",
    },
    "ISSUE-4": {
        "files": ["shop/inventory.py", "shop/cart.py"],
        "finding": "Cart.add reserves 1 instead of qty, and can_reserve uses > where the boundary needs >=.",
        "attempts": [[
            {"path": "shop/inventory.py", "find": "self.levels.get(sku, 0) > qty", "replace": "self.levels.get(sku, 0) >= qty"},
            {"path": "shop/cart.py", "find": "self.stock.reserve(sku, 1)", "replace": "self.stock.reserve(sku, qty)"},
        ]],
        "lesson": "Off by one at a boundary comparison: check whether > should be >=.",
    },
    "ISSUE-5": {
        "files": ["shop/dates.py"],
        "finding": "weekday() > 5 excludes Saturday, which is 5.",
        "attempts": [[{"path": "shop/dates.py", "find": "day.weekday() > 5", "replace": "day.weekday() >= 5"}]],
        "lesson": "Off by one at a boundary comparison: Saturday is weekday 5.",
    },
}


class ScriptedModel:
    """A deterministic stand-in for the model. Same issue, same attempt, same answer."""

    def __init__(self, scripts: dict[str, dict] | None = None):
        self.scripts = scripts if scripts is not None else SCRIPTS
        self.calls: list[tuple[str, str]] = []

    def decide(self, role: str, payload: dict) -> dict:
        issue_id = payload["issue"]["id"]
        self.calls.append((role, issue_id))
        script = self.scripts.get(issue_id)
        if script is None:
            return {"planner": {"files": [], "hypothesis": "unknown issue"}, "investigator": {"finding": "no idea"},
                    "critic": {"verdict": "retry", "feedback": "unknown issue"}}.get(role, {"edits": [], "summary": "no idea"})
        if role == "planner":
            recalled = [m["issue_id"] for m in payload.get("memories", [])]
            return {"files": script["files"], "recalled": recalled,
                    "hypothesis": f"Look at {', '.join(script['files'])}" + (f", as in {', '.join(recalled)}" if recalled else "")}
        if role == "investigator":
            return {"finding": script["finding"]}
        if role == "fixer":
            attempts, n = script["attempts"], payload["attempt"]
            if n >= len(attempts):
                if not script.get("cycle"):
                    return {"edits": [], "summary": "no further ideas"}
                n %= len(attempts)
            return {"edits": attempts[n], "summary": f"attempt {payload['attempt'] + 1}: " + script["finding"]}
        if role == "critic":
            ok = not payload["failed"] and not payload["regressions"]
            return {"verdict": "pass" if ok else "retry", "lesson": script["lesson"],
                    "feedback": "" if ok else f"still failing: {payload['failed'] or payload['regressions']}"}
        raise ValueError(f"unknown role {role}")


ROLE_PROMPTS = {
    "planner": 'Plan a fix. Reply with JSON: {"files": [paths to inspect], "hypothesis": "..."}',
    "investigator": 'Explain the cause from the files and failing tests. Reply with JSON: {"finding": "..."}',
    "fixer": 'Propose minimal edits. Reply with JSON: {"edits": [{"path": "...", "find": "exact old text", '
             '"replace": "new text"}], "summary": "..."}. Reply with an empty edits list if you cannot fix it.',
    "critic": 'Judge the test results. Reply with JSON: {"verdict": "pass" or "retry", "feedback": "...", '
              '"lesson": "one sentence to remember for similar issues"}',
}


class OpenAICompatModel:
    """Any OpenAI-compatible chat endpoint: a hosted API, or a local open-weights model via Ollama, vLLM or LM Studio.

    Configured from environment variables only: REVIEWER_BASE_URL, REVIEWER_MODEL and REVIEWER_API_KEY.
    """

    def __init__(self, base_url: str | None = None, model: str | None = None):
        self.base_url = (base_url or os.environ.get("REVIEWER_BASE_URL", "http://localhost:11434/v1")).rstrip("/")
        self.model = model or os.environ.get("REVIEWER_MODEL", "llama3.1")

    def decide(self, role: str, payload: dict) -> dict:
        body = {"model": self.model, "response_format": {"type": "json_object"}, "messages": [
            {"role": "system", "content": f"You are the {role} in a code review team. {ROLE_PROMPTS[role]}"},
            {"role": "user", "content": json.dumps(payload)}]}
        headers = {"Content-Type": "application/json"}
        if key := os.environ.get("REVIEWER_API_KEY"):
            headers["Authorization"] = f"Bearer {key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            text = json.load(resp)["choices"][0]["message"]["content"]
        return json.loads(text)


def default_model():
    return OpenAICompatModel() if os.environ.get("REVIEWER_PROVIDER") == "openai" else ScriptedModel()
