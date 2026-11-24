# Structured research assistant

Reference solution for the **research-assistant** capstone (Playground mini lab 900.1).

The assistant reads an article, streams its progress, and returns findings as validated JSON: key findings,
named entities and a confidence score. Broken replies (truncated, refused, wrong shape) are sent back to the
model with the exact problem, up to two times, and then the run fails with a clear error instead of crashing.
The same code path works with a deterministic local fake (the default), a hosted API or a local
open-weights model behind an OpenAI-compatible endpoint.

Everything runs offline and free. No keys are needed for the tests or the demo.

## Quick start

```bash
uv sync
uv run pytest -q
uv run python -m research_assistant.cli demo
```

The demo scores all 20 bundled articles, prints a progress line per article, then the acceptance report and the
three broken cases:

```
Schema-valid outputs : 20/20 (first try 16, repaired 4: a04, a09, a13, a17)
Repair retries used  : 5
Time to first token  : p50 20 ms, p90 20 ms (target under 1500 ms) PASS
Entity recall        : 100% of expected names found
Failure handling     : three broken cases
  truncated, then fixed : recovered after 1 repair(s)
  refused, then fixed   : recovered after 1 repair(s)
  refused every time    : clear error, no crash: Could not get valid findings for article 'a01' after 3 attempts ...
```

Other commands:

```bash
uv run python -m research_assistant.cli analyse a17        # one bundled article, JSON on stdout, progress on stderr
uv run python -m research_assistant.cli analyse notes.txt  # any text file
uv run python -m research_assistant.cli score              # the scoring script on its own
uv run python -m research_assistant.cli serve --port 8000  # SSE endpoint on localhost, open / in a browser
curl -N "http://127.0.0.1:8000/analyse?id=a04"             # watch the events, including one repair
```

## Architecture

```
article ──> prompt template ──> provider.stream(messages) ──> tokens ──> SSE / progress line
                                       ^                         │
                                       │                         v
                              repair prompt <── invalid ── strict check (JSON + Pydantic schema)
                              (max 2 times)                      │ valid
                                                                 v
                                                ResearchResult (findings, attempts, TTFT)
```

| File | Role |
| --- | --- |
| `src/research_assistant/prompt.py` | Prompt template with named variables (article id, title, text, rules, schema) and the repair prompt |
| `src/research_assistant/schema.py` | Pydantic models `Findings` and `Entity`, and `strict_json_schema()` for strict structured output |
| `src/research_assistant/providers.py` | Provider protocol, `FakeModel`, `ScriptedModel`, `OpenAICompatible`, `get_provider()` |
| `src/research_assistant/pipeline.py` | `run()` event generator, `check()`, `analyse()`, `ResearchError`, `ProgressLine` |
| `src/research_assistant/sse.py` | SSE framing and a stdlib HTTP endpoint (`GET /analyse?id=`, `POST /analyse`) |
| `src/research_assistant/web_fastapi.py` | Optional FastAPI version of the endpoint (`uv sync --extra web`) |
| `src/research_assistant/data/` | 20 short original articles with the entities each one names |
| `src/research_assistant/score.py` | Scoring script: validity, retries, TTFT p50 and p90, entity recall |
| `src/research_assistant/cli.py` | `demo`, `score`, `analyse`, `serve` |

Design choices worth noticing:

- `run()` is a generator of JSON-safe events (`start`, `token`, `progress`, `invalid`, `result`, `error`). The CLI
  progress line, the SSE endpoint and the tests all consume the same events, so there is one code path.
- `run()` returns a `ResearchError` instead of raising it, so a stream can always end with an `error` frame.
  `analyse()` is the convenience wrapper that raises.
- The check is strict on purpose: no stripping of code fences or leading prose. A reply that needs fixing goes
  back to the model as a repair, so it is counted as a retry rather than hidden.
- The fake model is stateless: it works out the attempt number from the conversation, and `DEFAULT_FAULTS`
  makes four articles misbehave on their first try (a04 truncated, a09 prose before the JSON, a13 confidence
  out of range, a17 truncated and then a missing field), so the repair path runs on every test run.

## How each acceptance target is measured

| Target | How | Test |
| --- | --- | --- |
| Schema-valid outputs: 20 of 20 | Every article goes through `score()`. Each returned `Findings` is re-validated and its `article_id` checked. Retries count, silent fixes are impossible because `check()` does none. A control test with `max_repairs=0` shows the four faulty articles really fail without repair. | `tests/test_accept_schema_valid.py` |
| Time to first token under 1.5 s | `run()` stamps the time from sending the request to the first non-empty chunk, per attempt. The report uses the first attempt of each request and the nearest-rank 90th percentile. The fake path is asserted under 1.5 s, and a slow fake shows the check can fail. | `tests/test_accept_ttft.py` |
| No crash on truncated or refused output | Three broken cases with `ScriptedModel`: truncated then good (one repair), refused twice then good (two repairs), and truncated, refused, truncated (a `ResearchError` naming the article, the attempt count and the last problem). The SSE stream ends with an `error` frame, not an exception. Provider exceptions such as timeouts also count as failed attempts. | `tests/test_accept_failures.py` |

Run just the acceptance tests with:

```bash
uv run pytest -q tests/test_accept_schema_valid.py tests/test_accept_ttft.py tests/test_accept_failures.py
```

## Plugging in a real model

Set environment variables. The pipeline code does not change.

```bash
# A local open-weights model with Ollama (or vLLM, LM Studio, a llama.cpp server)
export RA_PROVIDER=openai
export RA_BASE_URL=http://localhost:11434/v1
export RA_MODEL=<a model you have pulled>
uv run python -m research_assistant.cli demo

# A hosted API that speaks the OpenAI chat completions format
export RA_PROVIDER=openai
export RA_BASE_URL=https://<host>/v1
export RA_MODEL=<model name>
export RA_API_KEY=<your key>        # read from the environment only, never printed
```

If the host rejects `response_format` of type `json_schema`, set `RA_RESPONSE_FORMAT=json_object` or `none`.
The strict check and the repair loop still guarantee the shape. For a provider with its own API, write a class
with a `name` and a `stream(messages)` method that yields text chunks, and return it from `get_provider()`.

The TTFT target is meant for the hosted path: run `cli score` against your host to get a real p90.

## Stretch: saying "unsure"

`analyse(article, provider, unsure_below=0.6)` marks a valid result with `status="unsure"` when its confidence is
under the threshold, so the caller can say it is unsure instead of answering. See `tests/test_repair.py`.

## Milestones

These match the milestone guide on the site.

- **M1. Write the prompt template and the strict schema**: `src/research_assistant/prompt.py`,
  `src/research_assistant/schema.py`, `tests/test_prompt_schema.py`
- **M2. Add the provider switch and a fake model**: `src/research_assistant/providers.py`, `tests/test_providers.py`
- **M3. Stream progress as server-sent events**: `src/research_assistant/pipeline.py` (`run`, `ProgressLine`),
  `src/research_assistant/sse.py`, `tests/test_streaming.py`
- **M4. Validate strictly and repair up to twice**: `src/research_assistant/pipeline.py` (`check`, repair loop,
  `ResearchError`), `tests/test_repair.py`
- **M5. Bundle the test set and the scoring script**: `src/research_assistant/data/`,
  `src/research_assistant/score.py`, `tests/test_scoring.py`
- **M6. Measure the acceptance targets**: `src/research_assistant/cli.py`, `tests/test_accept_schema_valid.py`,
  `tests/test_accept_ttft.py`, `tests/test_accept_failures.py`
