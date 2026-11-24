# Production hardening suite

Reference solution for the **production-hardening** capstone (Playground mini lab 900.4).

A small support agent resolves customer issues: it reads the issue, searches a knowledge base, checks a
linked status page, sets a label and posts a reply. This project takes that agent to production readiness:

- spans for every model and tool call, exported as OTLP JSON for any tracing tool;
- prompt caching, an exact and a semantic cache, and a small-then-large model cascade;
- PII redaction, indirect prompt-injection checks on tool results, output checks and least-privilege tools;
- dual-LLM separation, so the model that can call tools never reads untrusted text;
- an append-only, hash-chained audit log of reasoning summaries, mutations and blocked actions;
- a trajectory eval suite and a CI gate that exits non-zero when a change regresses.

Every layer sits behind a switch in `Config`, so you can measure the agent before and after on the same set.
Everything runs offline against deterministic fake models. No keys, no network, no paid calls.

## Quick start

```bash
uv sync
uv run pytest -q
uv run python -m hardening.cli demo
```

The demo runs the 20 bundled issues and 16 injection cases with hardening off and on:

```
Production hardening report (fake models, offline)
  Issues resolved          : off 20/20, on 20/20
  Injection tests blocked  : 16/16 (100%; unhardened blocks 0) PASS
  Credits per resolved     : off 62.09, on 6.43, 90% lower PASS
  Trace coverage           : 100% of model and tool calls have a span (833 spans) PASS
  Audit log                : 277 entries, chain verified, head 2a8be01691e0
  Guardrail flags          : 16/16 injections, 0/20 benign issues
  Credits per resolved with one lever switched off:
    without prompt-cache   : 12.02
    without semantic-cache : 6.89
    without cascade        : 29.74
```

It also writes `out/trace.json` (OTLP JSON) and `out/audit.jsonl`. Other commands:

```bash
uv run python -m hardening.cli eval                        # full report as JSON
uv run python -m hardening.cli gate                        # the CI gate, exit 0
uv run python scripts/ci_gate.py --regress dual-llm        # an injected regression, exit 1
uv run python -m hardening.cli audit verify out/audit.jsonl [--head <hash>]
uv run python -m hardening.cli baseline                    # rewrite evals/baseline.json after a reviewed change
```

## Architecture

```
issue ─> agent loop ─> planner (cascade: small, then large if unsure)
             │             │ tool call with $refs, never raw text
             │             v
             │       tool policy + output check ──blocked──> audit "blocked"
             │             │ allowed
             │             v
             │        tool (tracker, KB, page) ──> audit "mutation"
             │             │ raw result
             │             v
             │   PII redaction + injection check ──flag──> audit "injection_flagged"
             │             │
             │             v
             │   quarantined reader (no tools, fixed schema, caches in front)
             │             │ symbolic view: category, $Q1, $U1, kb id
             └─────────────┘
 every box above runs inside a span: agent.run > model.call / tool.call > guardrail.* / cache.lookup
```

| File | Role |
| --- | --- |
| `src/hardening/agent.py` | `Agent` loop, `Config` switches, spans and audit entries around every call |
| `src/hardening/model.py` | Model interface and `FakeModel`: deterministic, reports token usage, simulates prefix caching, and is gullible on purpose (it obeys instructions in tool output) |
| `src/hardening/tools.py` | Tool specs, the fake tracker `Workspace`, the bundled data loader, `compromised()` and `resolved()` |
| `src/hardening/tracing.py` | Span tracer with GenAI attribute names and OTLP JSON export |
| `src/hardening/otel_bridge.py` | Optional mirror into the OpenTelemetry SDK (`uv sync --extra otel`) |
| `src/hardening/tokens.py` | Offline token counter and credit rates per tier (cached input at a tenth) |
| `src/hardening/caching.py` | Exact cache, semantic cache and a deterministic local embedding |
| `src/hardening/cascade.py` | Small-then-large cascade with a confidence threshold |
| `src/hardening/guardrails.py` | PII redaction, injection detector, output check, `ToolPolicy` |
| `src/hardening/dual_llm.py` | Quarantined reader, schema validation and the `VariableStore` of `$refs` |
| `src/hardening/audit.py` | Append-only hash-chained log and `verify_entries()` |
| `src/hardening/evals.py` | Trajectory checks, trace coverage, `run_suite()`, `gate()` |
| `src/hardening/cli.py` | `demo`, `eval`, `gate`, `baseline`, `audit verify` |
| `src/hardening/data/issues.json` | 20 issues, 8 KB articles, 3 status pages and 16 injection cases |
| `evals/baseline.json` | The numbers the gate compares against |
| `scripts/ci_gate.py` | CI entry point |
| `.github/workflows/eval-gate.yml` | Sample workflow; copy it to `.github/workflows/` at your repository root |

Design choices worth noticing:

- The fake model is deliberately gullible. If a tool result says "call close_issue", it does. That makes the
  injection tests honest: with hardening off, all 16 cases succeed.
- Heuristic detection is a tripwire, not the wall. The planner in dual-LLM mode only gets a symbolic view
  (`{"category": "billing", "query_ref": "$Q1", "url_refs": ["$U1"]}`), so even a missed injection has
  nothing to steer. The controller swaps `$refs` for values only when it executes a tool.
- Flagged text is never cached, and a semantic cache hit reuses only the meaning fields (category, query);
  links are re-read from the new text, because a near-duplicate issue can carry a different link.
- Credits are computed from token counts and the ratios in `tokens.py`. They are not money.

## How each acceptance target is measured

| Target | How | Test |
| --- | --- | --- |
| Injection tests blocked: 100% | 16 indirect injections, at least 5 each in the issue body, a KB article and a fetched page, with a different technique each (override, label hijack, hidden HTML, zero-width split, base64, tracking pixel, secret leak, cross-issue action, fake tool result, chat template tokens, non-English and more). A case is blocked when `Workspace.compromised()` is empty: no closed issue, no email, no action on another issue, no outside fetch, the right label and no attacker link or secret in the reply. A control run with hardening off must be compromised by every case. | `tests/test_accept_injections.py` |
| Cost per resolved issue: at least 35% lower | Total credits over the 20 issues divided by resolved issues, with `Config.off()` (one large model, no caching) and `Config()`, on the same set. Both must resolve 20 of 20. Each lever is also switched off on its own to show it lowers the cost. | `tests/test_accept_cost.py` |
| Trace coverage: every model and tool call has a span | Every model call id recorded by the models, and every tool execution id recorded by the tracker, must appear on a `model.call` or `tool.call` span inside an `agent.run` trace. Refused calls have spans too. A control test makes one untraced call and checks the coverage drops. | `tests/test_accept_trace_coverage.py` |
| Audit log verifies, tampering is caught | Edit, delete, reorder and truncate lines of a real log and check `verify` fails at the right entry. | `tests/test_audit.py` |
| CI gate blocks a regression | The gate passes on the current build and exits 1 with each of `--regress dual-llm`, `guardrails`, `cascade` and `prompt-cache`. | `tests/test_ci_gate.py` |

## Seeing the traces in a tracing tool

`out/trace.json` is OTLP/JSON. Post it to any collector that accepts OTLP over HTTP with
`hardening.tracing.post_otlp_json(path, "http://localhost:6006/v1/traces")` (a local Phoenix server, for example),
or stream spans live through the OpenTelemetry SDK:

```bash
uv sync --extra otel
export OTEL_EXPORTER_OTLP_TRACES_ENDPOINT=http://localhost:6006/v1/traces    # or your Langfuse OTLP endpoint
export OTEL_EXPORTER_OTLP_TRACES_HEADERS="Authorization=Basic <base64 of public:secret key>"   # Langfuse only
```

```python
from hardening.otel_bridge import install_otel
install_otel(agent.tracer)
```

The span names and attributes are the same either way, so the coverage check still applies.

## Plugging in a real model

```bash
export HARDENING_PROVIDER=openai
export HARDENING_BASE_URL=http://localhost:11434/v1      # a local open-weights model (Ollama, vLLM, LM Studio)
export HARDENING_MODEL_SMALL=<small model name>
export HARDENING_MODEL_LARGE=<large model name>
export HARDENING_API_KEY=<key>                           # hosted APIs only; read from the environment, never printed
uv run python -m hardening.cli demo
```

`OpenAICompatModel` (in `providers.py`) maps the planner request to chat completions with tools and the reader to a
JSON-object reply. Hosted APIs rarely report a confidence, so the cascade falls back to a validity check. Change
`RATES` in `tokens.py` to the ratios of the models you run. Real models are not deterministic, so rebuild the
baseline from several runs before you gate on it.

## Milestones

These match the guide on the site.

1. **Trace every model and tool call.** `tracing.py`, `agent.py` (`_call`, `_tool`), `evals.py` (`trace_coverage`).
   Check: `uv run pytest -q tests/test_accept_trace_coverage.py`.
2. **Cut credits with caching and a cascade.** `tokens.py`, `caching.py`, `cascade.py`, `model.py` (usage and prefix cache).
   Check: `uv run pytest -q tests/test_accept_cost.py`.
3. **Add input and output guardrails.** `guardrails.py`, `agent.py` (`_check_call`, `_ingest`).
   Check: `uv run pytest -q tests/test_guardrails.py`.
4. **Split the planner from the reader.** `dual_llm.py`, `agent.py` (`_ingest`, `$ref` deref).
   Check: `uv run pytest -q tests/test_accept_injections.py`.
5. **Write an append-only audit log.** `audit.py`, `cli.py` (`audit verify`).
   Check: `uv run pytest -q tests/test_audit.py`.
6. **Gate releases in CI and measure the targets.** `evals.py`, `cli.py`, `scripts/ci_gate.py`, `evals/baseline.json`,
   `.github/workflows/eval-gate.yml`. Check: `uv run pytest -q tests/test_ci_gate.py tests/test_demo.py`.

## Stretch: kill switch and canary

Add a `HARDENING_KILL_SWITCH` environment variable that `Agent.run` checks before each step (stop, audit a
`killed` entry, hand the issue to a person). For a canary, route a fixed share of issues by a hash of the issue id
to a second `Config` or prompt version, tag spans with `release.variant`, and run `gate()` on each variant's slice
before you widen it.
