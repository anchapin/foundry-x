# Tutorial: Your first 30 minutes with FoundryX

> A guided end-to-end walkthrough for new contributors. You will clone the
> repo, plant a deterministic offline trace, inspect it with the trace CLI,
> read every event the agent loop emits, and see how those traces feed the
> KPIs and regression gate. No live model required — every command below
> runs offline against the bundled `MockModelAdapter` fixture.
>
> Companion documents (read after this tutorial, not before):
>
> - [AGENTS.md](../AGENTS.md) — operational rules for AI collaborators.
> - [CONTRIBUTING.md](../CONTRIBUTING.md) — PR workflow for humans.
> - [docs/ARCHITECTURE.md](./ARCHITECTURE.md) — runtime architecture map.
> - [docs/OPERATOR.md](./OPERATOR.md) — the operator workflow this loop mirrors.
> - [docs/CONTEXT.md](./CONTEXT.md) — glossary and the `kind` event vocabulary.

## What you will build

By the end of this tutorial you will have:

1. A working FoundryX install.
2. A trace session in `logs/traces.db` that you can inspect with
   `foundry-x-trace show <session_id>`.
3. A working mental model of the six trace events every agent session
   produces: `task_received`, `user_prompt`, `model_request`,
   `model_response`, `tool_call`, `outcome`.
4. A passing run of the test suite, including the tutorial's own CI test.

Budget: **under 30 minutes**, fully offline.

---

## Step 1 — Clone and install (≈ 5 min)

```bash
git clone https://github.com/anchapin/foundry-x.git
cd foundry-x
uv sync                       # install deps (ADR-0002)
cp .env.example .env          # no edits needed for this tutorial
```

FoundryX uses [`uv`](https://docs.astral.sh/uv/) for dependency management.
The `.env` copy is harmless even unedited — the offline path in this
tutorial never reads model credentials.

Verify the CLI is wired:

```bash
uv run foundry-x-trace --help
```

You should see the subcommand list (`sessions`, `show`, `seed-sample-trace`,
`prune`, …). If not, re-run `uv sync` — the console scripts are registered
in `pyproject.toml` §`[project.scripts]`.

---

## Step 2 — Plant a deterministic offline trace (≈ 1 min)

The trace store under `logs/` is the ground truth of what an agent did
([PHILOSOPHY.md §1](./PHILOSOPHY.md)). Before standing up a live model,
plant a deterministic sample session so the CLI has something to render:

```bash
uv run foundry-x-trace seed-sample-trace
```

Expected output (the `session_id` is freshly generated each run):

```
seeded session_id=<UUID>
```

This plants one session containing every event kind the production
Runner emits ([`src/foundry_x/trace/cli.py`](../src/foundry_x/trace/cli.py)
`_seed_sample_trace`). All planted payloads are literal placeholder text —
no tokens, keys, or PEM blocks ([docs/SECURITY.md](./SECURITY.md) §Secrets).

---

## Step 3 — List and inspect the session (≈ 2 min)

List every session the store knows about:

```bash
uv run foundry-x-trace sessions
```

```
session_id  started_at  harness_version  model_id
<UUID>      2026-07-18T17:33:23Z  seed-sample      seeded-llama-sample  ...
```

Copy the `session_id` and render its timeline:

```bash
uv run foundry-x-trace show <session_id>
```

```
Session: <session_id>
Events: 7

  #1   +0.0s  task_received    Refactor the auth helper to drop the legacy token cache.
  #2   +0.0s  user_prompt
  #3   +0.0s  model_request
  #4   +0.0s  model_response   {'role': 'assistant', ...} [tokens:60]
  #5   +0.0s  tool_call        read_file (12ms)
  #6   +0.0s  tool_result      read_file (12ms)
  #7   +0.0s  outcome          success
```

This is the shape of **every** FoundryX agent session: one lifecycle
bracket (`task_received` → `outcome`) wrapping zero or more agent-loop
round-trips (`model_request` → `model_response` → `tool_call` →
`tool_result`).

---

## Step 4 — Read the trace events (≈ 8 min)

The full `kind` vocabulary is canonical in
[docs/CONTEXT.md §Event kinds](./CONTEXT.md#event-kinds). The six events
below are the ones you will see in nearly every session; learn them first.

| # | `kind` | What it means | Payload to look at |
| --- | --- | --- | --- |
| 1 | **`task_received`** | The raw `--task` argument the operator passed in, recorded *before* the agent loop opens. | `payload.prompt` — the original task text. |
| 2 | **`user_prompt`** | The task as fed into the model, plus the size of the tool surface the agent sees. | `payload.content`, `payload.tool_count`. |
| 3 | **`model_request`** | One chat-completion round-trip to the model. Emitted once per step. | `payload.step` (loop index), `payload.message_count`, `payload.tool_count`. |
| 4 | **`model_response`** | The assistant message the model returned, plus any tool calls it emitted. | `payload.message`, `payload.tool_calls`, `payload.tokens_used`. |
| 5 | **`tool_call`** | One tool execution. Emitted once per `ToolCall` the model emits, bracketed by hook fan-out. | `payload.name`, `payload.arguments`, `payload.duration_ms`, `payload.hook_overhead_ms`. |
| 6 | **`outcome`** | Terminal event, always emitted in `finally`. The Digester attributes the session's status from this. | `payload.status` (`success` \| `truncated` \| `failed`), `payload.reason`, `payload.steps`. |

### How to think about the event flow

```
task_received          ← operator's intent (lifecycle open)
   │
   └─ user_prompt      ← task enters the model conversation
        │
        └─ model_request → model_response   ← one round-trip (step N)
             │
             └─ tool_call → tool_result     ← one tool execution (step N)
        (loop repeats until final answer, max_steps, or budget)
   │
outcome                ← terminal status (lifecycle close)
```

A session with three tool calls will have **three** `model_request` /
`model_response` pairs and **three** `tool_call` / `tool_result` pairs —
the `step` index on each payload tells you which round-trip it belongs to.

### Failure signals

Most `kind`s are benign, but a few carry failure signals on their own
([docs/CONTEXT.md §Failure-signalling subset](./CONTEXT.md#failure-signalling-subset)):

- `task_failed` / `task_aborted` — the session did not complete normally.
- `model_error` — the model call raised.
- `hook_registry_error` — security hooks (including the prompt-injection
  firewall) are silently disabled for this session. Treat as degraded.
- Any payload with an `error`, `traceback`, or `exception` key.

When you read a real session, scan for these first.

---

## Step 5 — See how traces feed the KPIs (≈ 3 min)

The trace store is not just a log — it is the input to the evolution loop.
Compute the three PRD success-metric KPIs from the session you just
planted:

```bash
uv run foundry-kpis
```

```
| KPI | Value |
| --- | --- |
| Cycle Time (seconds) | N/A |
| Regression Rate | 0.00 |
| Improvement Rate | 0.00 |
| Hooks Disabled Count | 0 |
| Hooks Disabled Rate | 0.00 |
| Token Budget Hit Rate | 0.00 |
```

`Cycle Time` is `N/A` because the seeded session has no `critic_verdict`
event — the loop in [docs/CONTEXT.md §The loop](./CONTEXT.md#the-loop)
hasn't run end-to-end on it. A real harness-evolution iteration produces
that verdict and the metric fills in.

Aggregate Critic verdicts across the whole store with:

```bash
uv run fx-trace regression-report
```

```
- Total verdicts: 0
- Approvals: 0
- Rejections: 0
```

Empty today, populated once the `Evolver` → `Critic` loop has run against
a benchmark suite (see [docs/OPERATOR.md](./OPERATOR.md)).

---

## Step 6 — Run the test suite (≈ 5 min)

```bash
uv run ruff check .           # lint (also enforced by pre-commit)
uv run pytest -m "not benchmark"
```

Unit tests run offline against the deterministic `MockModelAdapter`
fixture defined in [`tests/conftest.py`](../tests/conftest.py). The
tutorial's own CI test lives at
[`tests/test_tutorial.py`](../tests/test_tutorial.py) — it drives
`run_task` end-to-end with the mock adapter and asserts the trace store
grew and contains the six events you just learned. If that test passes,
the path you walked in Steps 2–4 is verified for every PR.

Optional: stand up the pre-commit hooks you will use for real
contributions:

```bash
uv run pre-commit install
uv run pre-commit run --all-files
```

---

## Step 7 — Where to go next (≈ 5 min)

You now have the vocabulary to read the rest of the documentation without
getting lost. Pick the path that matches what you want to do:

- **Observe your own agent** →
  [`src/foundry_x/trace/logger.py`](../src/foundry_x/trace/logger.py) and
  [`src/foundry_x/execution/runner.py`](../src/foundry_x/execution/runner.py).
- **Analyze failures** →
  [`src/foundry_x/evolution/digester.py`](../src/foundry_x/evolution/digester.py)
  and [docs/OPERATOR.md](./OPERATOR.md) (degradation modes).
- **Evolve the harness** →
  [`src/foundry_x/evolution/evolver.py`](../src/foundry_x/evolution/evolver.py)
  and [ADR-0004](./adr/0004-self-modification-guardrails.md).
- **Run a local model** → [`infra/llama-cpp/README.md`](../infra/llama-cpp/README.md).
- **Make a contribution** → [CONTRIBUTING.md](../CONTRIBUTING.md).

When you are ready to open a PR, the operational rules that govern this
repo live in [AGENTS.md](../AGENTS.md) (for agents) and
[CONTRIBUTING.md](../CONTRIBUTING.md) (for humans). Both apply to everyone.

---

## Troubleshooting

- **`uv run foundry-x-trace ...` prints "No sessions found."**
  You have not planted or run a session yet. Run `seed-sample-trace`
  (Step 2) or `fx-runner --task "..."` against a live model.
- **`uv sync` fails to resolve.** Ensure you are on Python 3.11+
  (`python --version`). The lockfile (`uv.lock`) is committed; never run
  `pip install` directly (ADR-0002).
- **`foundry-kpis` shows all zeros.** That is correct for a store with
  only seeded sessions — the KPIs measure the evolution loop, which has
  not run yet. Plant more sessions or run a benchmark to populate them.
- **A test fails after a fresh clone.** Re-read the failure output; do
  not paper over it. AGENTS.md §2: "If a test fails, the change is not
  done."
