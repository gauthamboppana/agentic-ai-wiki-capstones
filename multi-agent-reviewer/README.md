# Multi-agent reviewer

A team of four agents (planner, investigator, fixer and critic) that works through issues in a small seeded
code repository. Every tool comes from an MCP server over stdio, the run state is checkpointed to SQLite after
every node so a killed run can resume, past fixes are kept as episodic memory, and nothing is written to the
repository until a reviewer approves the diff. Every run ends with an explicit stop reason.

Everything runs offline. The default model is a deterministic `ScriptedModel`, so tests repeat exactly.

## Quick start

```bash
uv sync
uv run pytest -q
uv run python -m reviewer.cli demo
```

The demo works all five seeded issues on a temporary copy of the repo and prints the acceptance report:

```
Acceptance report
  resolved issues:     5/5 (multi-file: ISSUE-3, ISSUE-4)
  unauthorised writes: 0 (7 writes, 7 approvals; rejected run wrote 0)
  resume:              killed with exit code 137 after 'writer', resumed at 'critic', stop reason = resolved
  max-iterations:      stop reason = max_iterations (stopped after 12 node runs without finishing (limit 12))
ALL TARGETS MET
```

Work one issue yourself, approving at the prompt, then kill and resume a run:

```bash
uv run python -m reviewer.cli run ISSUE-3 --workdir /tmp/rv --approve prompt
uv run python -m reviewer.cli run ISSUE-4 --workdir /tmp/rv --approve auto --crash-after writer
uv run python -m reviewer.cli runs --workdir /tmp/rv
uv run python -m reviewer.cli resume <run id> --workdir /tmp/rv
uv run python -m reviewer.cli audit --workdir /tmp/rv
```

`--approve defer` stops the run at the gate with `awaiting_approval`; answer it later with
`resume <run id> --decision approve` or `--decision reject`.

## Architecture

```
planner -> investigator -> fixer -> gate -> writer -> critic -> finish -> END
              |              |        |                  |
              |              |        +-- reject -> END   +-- retry -> investigator
              |              +-- edit did not apply -> investigator
              |              +-- no edits -> END (gave_up)
              +-- issue tests already pass -> END (already_fixed)
any node: step limit reached -> END (max_iterations)
gate, no decision yet -> pause (awaiting_approval), resume later with a decision
```

| File | Role |
| --- | --- |
| `src/reviewer/graph.py` | A small state graph: nodes, conditional edges, a checkpoint after every node, interrupts, the max-iterations guard and stop reasons. |
| `src/reviewer/agents.py` | The nodes: planner, investigator, fixer, gate (approval interrupt), writer, critic and finish, and the edges between them. |
| `src/reviewer/model.py` | `ScriptedModel` (default, deterministic) and `OpenAICompatModel` (optional, real). |
| `src/reviewer/mcp_server.py` | Two stdio MCP servers: `workspace` (fs_list, fs_read, fs_write, tests_run) and `tracker` (issues_list, issues_get, issues_comment, issues_close). |
| `src/reviewer/mcp_client.py` | Starts the servers as subprocesses, lists their tools and routes each call. |
| `src/reviewer/approval.py` | Diffs, decisions (approve, edit, reject), approval policies, the ledger and the audit. |
| `src/reviewer/checkpoint.py` | SQLite checkpoints, one row per node run. |
| `src/reviewer/memory.py` | Episodic memory of past fixes, retrieved by word overlap. |
| `src/reviewer/app.py` | Wires it together on a work directory: a copy of the seeded repo and one `state.db`. |
| `src/reviewer/cli.py` | `demo`, `run`, `resume`, `runs`, `audit` and `issues`. |
| `src/reviewer/seed/` | The seeded `shop` repo with five bugs (two of them span two files) and `issues.json`. |

Two design points matter most. The fixer only proposes edits; it never writes. Writes happen in the writer
node, through the `fs_write` tool, and the server itself refuses any write whose `approval_id` is missing,
already used, for another path, a rejection, or for different content (it compares SHA-256 hashes). And the
code, not the model, has the final word on a fix: the critic can only pass a change if the issue's tests pass
and nothing that passed before now fails.

## How each acceptance target is measured

| Target | Test | How |
| --- | --- | --- |
| Resolved issues: at least five, multi-file, tests pass after each | `tests/test_resolved_issues.py` | Runs all five seeded issues in order on one repo. After each, the issue's own tests pass and the failing count drops by exactly one; ISSUE-3 and ISSUE-4 each write two files. |
| Unauthorised writes: zero | `tests/test_approvals.py` | Audits every row in the `writes` table against an approval with the same path and content hash, and checks every changed file against its last approved write, which catches writes that bypassed the server. A rejected change leaves the file untouched and the write log empty. |
| Resume: a killed run continues from its last checkpoint | `tests/test_resume.py` | Starts a real process that calls `os._exit(137)` right after the writer node's checkpoint, then resumes in a new process. The resumed run runs only `critic` and `finish` and repeats no writes. Also crashes after planner, fixer and gate. |
| Max-iterations guard and stop reasons | `tests/test_guard.py` | A fixer that undoes its own edit forever is stopped at the limit with `max_iterations` and a detail naming the limit. |
| MCP over stdio | `tests/test_mcp.py` | Sends raw JSON-RPC lines to each server: `initialize`, `tools/list`, `tools/call`, a path escape and an unknown method. |

## Plug in a real model

Any OpenAI-compatible chat endpoint works: a hosted API, or a local open-weights model through Ollama, vLLM or
LM Studio. Settings come only from environment variables, and the key is never printed or logged.

```bash
export REVIEWER_PROVIDER=openai
export REVIEWER_BASE_URL=http://localhost:11434/v1   # Ollama; or your host's /v1 URL
export REVIEWER_MODEL=llama3.1
export REVIEWER_API_KEY=...                         # only if your host needs one
uv run python -m reviewer.cli run ISSUE-1 --workdir /tmp/rv-real --approve prompt
```

Each role gets a short system prompt asking for JSON (see `ROLE_PROMPTS` in `model.py`). Tests never use it.

## Swapping in the heavier tools

- **Official MCP SDK.** `uv sync --extra mcp`, then rewrite each server with `FastMCP` and `@mcp.tool()`
  around the same `Workspace` and `Tracker` methods, and replace `McpClient` with `mcp.ClientSession` over
  `stdio_client`. The tool names and arguments stay the same, so `agents.py` does not change.
- **LangGraph.** `uv add langgraph langgraph-checkpoint-sqlite`. Build a `StateGraph` with the same node
  functions (wrap each as `lambda s: fn(s, ctx)`), use `add_conditional_edges` with the same routers, compile
  with a `SqliteSaver` and `interrupt_before=["gate"]`, and set `recursion_limit` for the step guard.
- **Postgres checkpoints.** Give `SqliteCheckpointer` and `Ledger` a `psycopg` connection instead; the SQL is
  plain enough to run unchanged.

## Milestones

These match the guide on the site.

- **M1. Build the graph engine with a step guard.** `src/reviewer/graph.py`. Check: `uv run pytest -q tests/test_guard.py`.
- **M2. Serve the tools over MCP.** `src/reviewer/mcp_server.py`, `src/reviewer/mcp_client.py`. Check: `uv run pytest -q tests/test_mcp.py`.
- **M3. Add the four agents and their edges.** `src/reviewer/agents.py`, `src/reviewer/model.py`, `src/reviewer/seed/`. Check: `uv run pytest -q tests/test_resolved_issues.py`.
- **M4. Put an approval gate before every write.** `src/reviewer/approval.py`, the `gate` and `writer` nodes in `src/reviewer/agents.py`, `fs_write` in `src/reviewer/mcp_server.py`. Check: `uv run pytest -q tests/test_approvals.py`.
- **M5. Checkpoint, resume and remember.** `src/reviewer/checkpoint.py`, `src/reviewer/memory.py`, `src/reviewer/app.py`. Check: `uv run pytest -q tests/test_resume.py`.
- **M6. Measure the acceptance targets.** `src/reviewer/cli.py`, `tests/`. Check: `uv run pytest -q` and `uv run python -m reviewer.cli demo`.
