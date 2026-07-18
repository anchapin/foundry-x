# AGENTS.md

> Operational ground rules for AI coding agents (Claude, GPT, local models
> driven by FoundryX itself, etc.) collaborating on this repository.
>
> - Humans: see [CONTRIBUTING.md](./CONTRIBUTING.md).
> - Why these rules exist: see [docs/PHILOSOPHY.md](./docs/PHILOSOPHY.md).
> - What you can and cannot touch in `harness/`: see [docs/SECURITY.md](./docs/SECURITY.md).

## 1. Read first, then act

Before you write code in this repo, read in this order:

1. `README.md` — what this is.
2. `docs/PRD.md` — product requirements and KPIs.
3. `docs/ROADMAP.md` — current phase and milestones.
4. `docs/PHILOSOPHY.md` — the principles you must not violate.
5. `docs/SECURITY.md` — guardrails, especially for `harness/` edits.
   `harness/manifest.json` controls which hooks are active; adding or
   removing a hook file requires updating the manifest.
6. `docs/CONTEXT.md` — glossary of project terms and the `kind`
   vocabulary produced by the `TraceLogger` (relevant whenever a
   payload, event name, or subsystem name is ambiguous).
7. `docs/ARCHITECTURE.md` — runtime architecture map (Runner,
   TraceLogger, Digester, Evolver, Critic and how they connect).
8. `docs/OPERATOR.md` — human-side workflow that mirrors the agent
   loop in §3 below.
9. `docs/MODEL_CONFIG.md` — the full set of model-side env vars
   (`OPENCODE_SERVER_URL`, `FOUNDRY_TOKEN_BUDGET`, `FOUNDRY_TASK_TIMEOUT`,
   `FOUNDRY_REQUEST_TIMEOUT_S`, …) and the resolution order.
10. `docs/adr/` — read the relevant ADR before changing that area:
    - `harness/` → ADR-0004 | `pyproject.toml` / deps → ADR-0002
    - `src/foundry_x/trace/` → ADR-0007, ADR-0003 | `benchmarks/` → ADR-0004, ADR-0005
    - Module-boundary models → ADR-0006 | `src/foundry_x/execution/` → ADR-0010
11. The relevant module under `src/foundry_x/`.

If you have not read the ADR for the subsystem you are about to change,
stop and read it. Speculation is not evidence.

## 2. Hard rules (do not violate)

These are non-negotiable. If a task appears to require violating one, stop
and ask the human.

- **Never edit `harness/system_prompt.txt`, `harness/hooks/*`, or
  `harness/skills/*` as a code change.** These files are the agent's
  DNA. They are evolved by the `Evolver` -> `Critic` loop, not
  hand-edited. If you think the harness needs to change, produce a
  `ProposedEdit` and route it through the evolution pipeline. See
  ADR-0004.
- **Never bypass the `Critic` gate.** No "I'll just push and run tests
  later." Every harness edit ships through the Critic or it does not
  ship. The `Critic` runs in an isolated sandbox and evaluates harness
  edits against the full pytest suite *plus* the benchmark suite
  (`benchmarks/tasks/`, marked `@pytest.mark.benchmark`). Regressing a
  previously-passing benchmark blocks the gate. See ADR-0004.
- **Never run destructive commands** (`rm -rf`, `git reset --hard`,
  force-push to a branch other than your own throwaway, dropping a
  database) without an explicit rollback path stated in the response.
- **Never commit secrets.** No API keys, tokens, or `.env` contents.
  `.env.example` is the template; real values live in `.env` (gitignored).
- **Never assume a library is available** without checking
  `pyproject.toml` and `uv.lock` first. If it is not there, add it via
  `uv add <package>` and explain why in the PR. The lockfile (`uv.lock`)
  is committed — always run `uv sync` after adding a dependency so the
  lockfile stays in sync with `pyproject.toml`.
- **Never silently swallow an exception.** Log it via the project's
  `TraceLogger`, surface it, or re-raise. Bare `except: pass` is a bug.
- **Never widen scope.** A bug fix is not a refactor. A feature is not a
  re-architecture. If you discover adjacent issues, file them and move on.
- **Only merge agent-authored PRs after CI is green.** Agents may merge
  their own PRs only when every required check is passing, the branch is
  cleanly mergeable, and the PR does not contain harness hand-edits. Two
  human approvals are still required for harness hand-edits.
- **Never pretend a benchmark passed.** If a test fails, the change is
  not done. Re-read the failure, do not paper over it.
- **Never merge directly to `main`.** All changes go through PRs targeting
  `develop`. The `main` branch is protected and requires PR reviews,
  status checks (CI + benchmark gate), and linear history.

## 3. The FoundryX way

This project is itself an agent harness foundry. The way we work here
mirrors the way our product works:

1. **Observe.** Read the trace (`logs/`) and the existing code before
   proposing a change. The trace store is ground truth. Note: `logs/` is
   gitignored; it contains live SQLite/JSONL trace data from agent runs.
2. **Digest.** Write a small failure report ("the existing approach
   breaks when X because Y").
3. **Propose the smallest viable change.** One file if possible. One
   concern per commit.
4. **Evaluate.** Run the test suite. If you changed the harness, the
   Critic must pass on a benchmark.
5. **Commit atomically.** Conventional Commits. The subject must answer
   "what changed and why" in one sentence.
6. **Hand off.** Open a PR. Summarize the trace evidence that motivated
   the change. Wait for human review.

## 4. Tooling you are expected to use

- **Package manager:** `uv` (ADR-0002). Never `pip install` directly.
- **Pre-commit hooks:** installed via `uv run pre-commit install`.
  Run on demand with `uv run pre-commit run --all-files`. Hooks
  include ruff, ruff-format, gitleaks (secret scan), and standard
  hygiene checks. See `.pre-commit-config.yaml`.
- **Lint:** `uv run ruff check .` — must pass before commit (and is
  also enforced by pre-commit). Always run before pytest.
- **Test:** `uv run pytest` — must pass before commit. Run after lint.
  - Single test: `uv run pytest tests/path/to_test.py::test_name`
  - Single benchmark: `uv run pytest benchmarks/tasks/test_name.py -m benchmark`
  - Benchmarks live alongside unit tests in `benchmarks/tasks/` and are
    marked `@pytest.mark.benchmark` (ADR-0004, ADR-0005).
  - Run the full benchmark suite: `uv run pytest -m benchmark`
  - `benchmarks/fixtures/` contains large benchmark inputs and is
    excluded from both ruff and pytest on purpose
    (`pyproject.toml` `extend-exclude` + `norecursedirs`). Don't lint
    or import from it; copy fixtures into `tests/` if you need them.
- **Critic harness-load smoke test:** before the pytest suite, the
  Critic runs `harness/scripts/load_check.py` to confirm the harness
  imports and registers cleanly. If your hook or skill change fails
  there but passes pytest, the harness can't be loaded at runtime.
- **CLI tools:**
  - `uv run fx-runner --task "..."` — run a single agent task session
  - `uv run foundry-x-trace` (alias `foundry-trace`) — inspect trace
    sessions, render timelines, grep events, prune old sessions
    (`--help` for flags)
  - `uv run foundry-kpis` — compute PRD success-metric KPIs from traces
  - `uv run fx-trace regression-report` — aggregate Critic verdicts
  - `uv run foundry-evolve` — run one evolution iteration against a failure report
  - `uv run foundry-sweep` — run a parametric sweep of harness variants
- **Type discipline:** Python 3.11+ syntax. `pydantic` for all
  structured data at module boundaries (ADR-0006). No `Any` without
  a comment explaining why.
- **Logging:** the project's own `TraceLogger`. Do not sprinkle
  `print()` or generic `logging.info` in library code; route through
  trace events so the evolution loop can see them.
- **Search:** prefer `rg` over `grep`.
- **Shell:** prefer `workdir` over `cd`. Quote paths with spaces.

## 5. Commit and PR etiquette

- **Branch from `develop` and target `develop`.** The `main` branch is
  protected — all work branches off `develop` and PRs target `develop`.
  Use `git checkout -b fix/issue-NNN-description develop` to create branches.
- One logical change per commit. If a refactor is needed to make the
  feature possible, that is two commits in two PRs.
- Commit subject: 50 chars, imperative, no trailing period.
  Example: `feat(trace): persist tool-call latency histogram`.
- PR description must include: motivation, evidence (trace excerpt or
  test output), risk, and the ADR(s) it advances or supersedes.
- Keep PRs <400 lines of diff where possible. If a change is larger,
  write the plan to `docs/adr/NNNN-...md` first.

## 6. When you are stuck

If after one focused attempt you cannot make progress:

1. Stop. Do not retry the same action with random tweaks.
2. State the failure concretely: what you tried, what you observed,
   what you expected.
3. List the assumptions that might be wrong.
4. Ask the human a specific question.

This mirrors operating rule 3 of `harness/system_prompt.txt` — we
practice what we preach.

## 7. The self-reference loop

The agent harness in this repo is written using tools shaped by the harness.
Keep the two layers strictly separate:

- **`src/foundry_x/`** — the *foundry*: Python code that wraps and evolves agents.
  When you read `src/foundry_x/execution/runner.py`, that is the code that talks to the agent.
- **`harness/`** — the *artifact being evolved*: the agent's own DNA (system prompt, hooks, skills).
  When you read `harness/system_prompt.txt`, that is the agent you are talking to.

Mixing these up is the most common mistake newcomers make.
