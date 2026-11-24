# Gen AI Academy capstones: reference solutions

Reference solutions for the four Gen AI Academy capstone projects. Each folder is a complete, runnable Python project with tests that check the capstone's acceptance targets.

| Capstone | Folder | What it builds |
|---|---|---|
| 1. Structured research assistant | [`research-assistant`](research-assistant) | Streams progress and returns schema-valid JSON, with repair retries and a provider switch |
| 2. Hybrid-search PDF RAG with page citations | [`pdf-rag`](pdf-rag) | Layout-aware parsing, BM25 plus dense retrieval with RRF, reranking and page citations |
| 3. Stateful multi-agent reviewer | [`multi-agent-reviewer`](multi-agent-reviewer) | Planner, investigator, fixer and critic over MCP tools, with checkpoints and an approval gate |
| 4. Production hardening suite | [`production-hardening`](production-hardening) | Tracing, caching and a model cascade, guardrails, dual-LLM separation, an audit log and a CI eval gate |

Every guide, with milestones, hints and checkpoints, is on the Gen AI Academy site under Capstones. Try each milestone yourself before you read the code here.

## Run one

You need Python 3.11 or later and [uv](https://docs.astral.sh/uv/).

```sh
cd research-assistant
uv sync
uv run pytest -q
```

Each folder's README gives its demo command, explains how each acceptance target is measured and lists its milestones.

## No keys, no cost

Everything runs offline against a deterministic fake model, so the tests are repeatable and free. To try a real model, see "Plug in a real model" in each README. Keys are read only from environment variables. Never commit them.

## CI

`.github/workflows/tests.yml` runs every capstone's tests, plus the production-hardening eval gate, on each push and pull request.
