# Quickstart: from git clone to first passing benchmark (under 15 minutes)

> A fast-track path for experienced developers. Clone, install, run a benchmark,
> done. No conceptual context required — just follow the steps.
>
> **Budget: under 15 minutes.** Fully offline — no live model, no API keys.
>
> This path uses the deterministic `MockModelAdapter` fixture. If you want to
> understand *why* it works, read the full
> [Tutorial](./TUTORIAL.md) afterwards.
>
> Companion documents (read after, not before):
>
> - [TUTORIAL.md](./TUTORIAL.md) — the full 30-minute guided walk-through.
> - [AGENTS.md](../AGENTS.md) — operational rules for AI collaborators.
> - [CONTRIBUTING.md](../CONTRIBUTING.md) — PR workflow for humans.

## What you will accomplish

By the end of this quickstart you will have:

1. A working FoundryX install.
2. A passing benchmark run against the offline mock adapter.
3. (Optional) A deterministic trace planted in `logs/traces.db`.

---

## Step 1 — Clone and install (≈ 3 min)

```bash
git clone https://github.com/anchapin/foundry-x.git
cd foundry-x
uv sync                       # install deps (ADR-0002)
cp .env.example .env          # no edits needed for the offline path
```

Verify the CLI is wired:

```bash
uv run foundry-x-trace --help
```

You should see the subcommand list (`sessions`, `show`, `seed-sample-trace`,
`prune`, …). If not, re-run `uv sync`.

---

## Step 2 — Run the benchmark suite (≈ 5 min)

FoundryX validates harness changes against a suite of benchmark tasks
(ADR-0004, ADR-0005). Each task runs an agent in an isolated workspace,
feeds it an input file, and asserts the agent's output matches a golden
expected file.

Run all benchmarks against the offline mock adapter:

```bash
uv run pytest -m benchmark
```

To run a single simple task instead of the full suite (faster, same
fixture):

```bash
uv run pytest benchmarks/tasks/test_nth_fibonacci.py -m benchmark
```

Every benchmark in `benchmarks/tasks/` is marked `@pytest.mark.benchmark`.
All run offline using the same `MockModelAdapter` fixture as the test suite
— no live model or network access required.

---

## Step 3 — Plant a trace (optional, ≈ 2 min)

The trace store (`logs/traces.db`) is the ground truth of every agent
session. Plant a deterministic sample session so the CLI has something to
render:

```bash
uv run foundry-x-trace seed-sample-trace
```

Inspect it:

```bash
uv run foundry-x-trace sessions
uv run foundry-x-trace show <session_id>
```

See [TUTORIAL.md Step 3](./TUTORIAL.md#step-3--list-and-inspect-the-session-2-min)
for the expected output shape.

---

## Step 4 — Where to go next (≈ 2 min)

You have a working install and a passing benchmark. Next steps:

- **Run the full test suite:**
  ```bash
  uv run ruff check .
  uv run pytest -m "not benchmark"
  ```
- **Understand the trace events:**
  [TUTORIAL.md Step 4](./TUTORIAL.md#step-4--read-the-trace-events-8-min)
  (the 8-min conceptual deep-dive — optional but recommended).
- **Evolve the harness:**
  [`foundry-evolve`](../src/foundry_x/evolution/evolver.py) and
  [docs/OPERATOR.md](./OPERATOR.md).
- **Run a local model:**
  [`infra/llama-cpp/README.md`](../infra/llama-cpp/README.md).
- **Make a contribution:**
  [CONTRIBUTING.md](../CONTRIBUTING.md).

When you open a PR, the operational rules live in
[AGENTS.md](../AGENTS.md) (for agents) and
[CONTRIBUTING.md](../CONTRIBUTING.md) (for humans).

---

## Troubleshooting

- **`uv run pytest -m benchmark` fails with "no such option -m".**
  Run from the repository root: `cd foundry-x` first.
- **`uv sync` fails to resolve.** Ensure you are on Python 3.11+
  (`python --version`). Never run `pip install` directly (ADR-0002).
- **A benchmark test fails after a fresh clone.** Re-read the failure
  output; do not paper over it. AGENTS.md §2: "If a test fails, the
  change is not done."
- **`foundry-x-trace ...` prints "No sessions found."**
  You have not planted or run a session yet. Run `seed-sample-trace`
  (Step 3) or `fx-runner --task "..."` against a live model.
