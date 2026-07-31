# AGENTS.md

> Operational ground rules for AI coding agents collaborating on this repository.
>
> - Humans: see [CONTRIBUTING.md](./CONTRIBUTING.md).
> - Why these rules exist: see [docs/PHILOSOPHY.md](./docs/PHILOSOPHY.md).
> - Harness guardrails: see [docs/SECURITY.md](./docs/SECURITY.md).

## 1. Read first, then act

First session here? Work through [`docs/TUTORIAL.md`](./docs/TUTORIAL.md)
(<30 min, fully offline): plants a trace, walks the six core events the
`TraceLogger` emits, reads the KPIs.

Before writing code, read in this order:

1. `README.md` — what this is.
2. `docs/PRD.md` — product requirements and KPIs.
3. `docs/PHILOSOPHY.md` — principles you must not violate.
4. `docs/SECURITY.md` — guardrails, especially for `harness/` edits.
   `harness/manifest.json` controls which hooks are active; adding or
   removing a hook file requires updating the manifest.
5. `docs/CONTEXT.md` — glossary and the `kind` vocabulary produced by the
   `TraceLogger`. The §Event kinds table is the canonical trace event
   payload contract. Adding a new `kind` is a vocabulary change that must
   ship in the same PR as the code that emits it.
6. `docs/ARCHITECTURE.md` — runtime map (Runner, TraceLogger, Digester,
   Evolver, Critic and how they connect).
7. `docs/MODEL_CONFIG.md` — model-side env vars and resolution order.
   Three resource caps guard against runaway loops: `FOUNDRY_TASK_TIMEOUT`
   (wall-clock, default 600 s), `FOUNDRY_TOKEN_BUDGET` (total tokens,
   unset), `FOUNDRY_MAX_EVENTS_PER_SESSION` (event count, unset).
8. `docs/adr/` — read the relevant ADR before changing that area:
   - `harness/` → ADR-0004 | deps → ADR-0002 | `trace/` → ADR-0003, ADR-0007
   - `benchmarks/` → ADR-0004, ADR-0005 | models → ADR-0006 | `execution/` → ADR-0010
   - `evolution/` → ADR-0010 | Conventional Commits → ADR-0008
   - Run `ls docs/adr/` for the full current set (0001–0035).
9. The relevant module under `src/foundry_x/`.

If you have not read the ADR for the subsystem you are about to change,
stop and read it. Speculation is not evidence.

## 2. Hard rules (do not violate)

If a task appears to require violating one of these, stop and ask the human.

- **Never edit `harness/system_prompt.txt`, `harness/hooks/*`, or
  `harness/skills/*` as a code change.** These are the agent's DNA —
  evolved by the `Evolver` → `Critic` loop, not hand-edited. Route
  harness changes through `ProposedEdit` (ADR-0004).
- **Never bypass the `Critic` gate.** Every harness edit ships through
  the Critic or it does not ship. The Critic evaluates harness edits
  against the full pytest suite *plus* the benchmark suite
  (`benchmarks/tasks/`, `@pytest.mark.benchmark`). Regressing a
  previously-passing benchmark blocks the gate. Diffs touching only
  `docs/`, `.pre-commit-config.yaml`, or `pyproject.toml` skip the
  benchmark suite; all other changes run the full gate.
  `foundry-evolve evolve --no-verify` skips the gate locally (audit-logged
  via a synthetic "skipped" `CriticVerdict`) but cannot ship a harness
  edit to `main`.
- **Never run destructive commands** (`rm -rf`, `git reset --hard`,
  force-push to a branch other than your own throwaway, dropping a
  database) without an explicit rollback path stated in the response.
- **Never commit secrets.** `.env.example` is the template; real values
  live in `.env` (gitignored).
- **Never assume a library is available** without checking `pyproject.toml`
  and `uv.lock` first. Add new deps via `uv add <package>`, then `uv sync`
  so the lockfile stays in sync. Explain why in the PR.
- **Never silently swallow an exception.** Log via `TraceLogger`, surface,
  or re-raise. Bare `except: pass` is a bug.
- **Never widen scope.** A bug fix is not a refactor. File adjacent
  issues and move on.
- **Never pretend a benchmark passed.** If a test fails, the change is
  not done.
- **Branch from `develop` and target `develop`.** `main` is protected
  (PR reviews, status checks, linear history). Agents may merge their
  own PRs only when all CI checks pass, the branch is cleanly mergeable,
  and the PR has no harness hand-edits (those need two human approvals).

## 3. The FoundryX way

This project is an agent harness foundry — we work the way the product works:

1. **Observe.** Read the trace (`logs/`) and existing code before proposing
   changes. `logs/` is gitignored; it contains live SQLite/JSONL trace data.
2. **Digest.** Write a small failure report ("breaks when X because Y").
3. **Propose the smallest viable change.** One file if possible. One concern
   per commit.
4. **Evaluate.** Run the test suite. Harness changes must pass the Critic gate.
5. **Commit atomically.** Conventional Commits. Subject answers "what changed
   and why" in one sentence.
6. **Hand off.** Open a PR with trace evidence. Wait for human review.

## 4. Tooling

- **Package manager:** `uv` (ADR-0002). Never `pip install` directly.
  Run `uv sync` after clone and after adding any dependency.
- **Pre-commit:** `uv run pre-commit install`. Run on demand with
  `uv run pre-commit run --all-files`. Hooks: ruff (with `--fix
  --exit-non-zero-on-fix`, so staged files get modified and must be
  re-added), ruff-format, gitleaks, and standard hygiene checks.
- **Lint:** `uv run ruff check .` — must pass before commit. `ruff`
  line-length is **100** here, not the default 88 (see `[tool.ruff]` in
  `pyproject.toml`). Fix format with `uv run ruff format .`. The
  `lint.yml` CI workflow runs both `ruff check .` and
  `ruff format --check` as separate jobs. The `ci.yml` workflow's
  `pre-commit` job also enforces both via pre-commit hooks. Both
  workflows run `tests/docs/test_doc_links.py` (broken cross-doc refs)
  and `tests/docs/test_ci_gate_claims.py` (CI enforcement claims in docs
  backed by real workflows).
- **Test:** `uv run pytest` — must pass before commit. Run after lint.
  Pytest discovers both `tests/` and `benchmarks/` (`testpaths` in
  `pyproject.toml`); benchmark tasks are gated by `@pytest.mark.benchmark`
  (ADR-0004, ADR-0005).
  - Unit tests only: `uv run pytest -m "not benchmark"`
  - Single test: `uv run pytest tests/path/to_test.py::test_name`
  - Single benchmark: `uv run pytest benchmarks/tasks/test_name.py -m benchmark`
  - Full benchmark suite: `uv run pytest -m benchmark`
  - List benchmark tasks without running: `uv run pytest --co -q -m benchmark`
  - **Test fixtures** (`tests/conftest.py`, `benchmarks/conftest.py`):
    - `model_adapter` — session-scoped `ModelAdapter` selected by
      `TEST_MODEL_MODE`. Defaults to `mock` (deterministic, no network)
      so unit tests and most benchmarks run offline. Set
      `TEST_MODEL_MODE=real` to drive `OpenAICompatibleAdapter` against
      `OPENCODE_SERVER_URL` / `LLAMACPP_HOST` (matches
      `.github/workflows/test.yml::test-real-model`). **Session-scoped**:
      do not re-configure it mid-test.
    - `mock_adapter` — fresh `MockModelAdapter` per test; ignores
      `TEST_MODEL_MODE`. Use when configuring per-test responses.
    - `benchmark_workspace` — per-test isolated `tmp_path` the benchmark
      task treats as the agent's entire filesystem. Indirect-parametrize
      with a `benchmarks/fixtures/<name>/` directory to seed inputs (raises
      `FileNotFoundError` if the fixture directory is missing).
  - `benchmarks/fixtures/` is excluded from ruff, ruff-format, and pytest
    on purpose (`pyproject.toml` `extend-exclude` + `norecursedirs`).
    Don't lint or import from it; copy fixtures into the
    `benchmark_workspace` (or `tests/`) if you need them outside a
    benchmark task.
- **Harness load check:** the Critic (and CI in both `ci.yml` and
  `test.yml`) runs `harness/scripts/load_check.py` before pytest to confirm
  the harness imports and registers cleanly. If your hook or skill change
  fails there but passes pytest, the harness can't be loaded at runtime.
- **CLI tools** (seven registered in `pyproject.toml` §`[project.scripts]`):

  | Command | Purpose |
  | --- | --- |
  | `uv run fx-runner --task "..."` | Run a single agent task session (flags: `--harness-dir`, `--trace-path`, `--workspace-root`, `--model-id`) |
  | `uv run foundry-x-trace` / `foundry-trace` | Trace inspection: `sessions`, `show`, `events-grep`, `render-failure`, `seed-sample-trace`, `prune` |
  | `uv run fx-trace` | KPI reports, regression reports, session summaries, tool-latency percentiles |
  | `uv run foundry-kpis` | Three PRD KPIs (cycle time, regression rate, improvement rate) + token-budget metric |
  | `uv run foundry-evolve evolve --session-id <id>` | One evolution iteration (Digester→Evolver→Critic). Also: `approve <uuid>`, `apply <uuid>`, `--background`, `--no-verify` |
  | `uv run foundry-sweep` | Parametric sweep of harness variants (Phase 3) |

  Pass `--help` to any command for the full flag surface.
- **Operational notes:**
  - `logs/` grows without bound. Prune with
    `uv run foundry-x-trace prune --keep-last N` or `--older-than DAYS`
    (both support `--dry-run`). Add `--vacuum` on a sqlite retention pass
    to reclaim the `traces.db-wal` sidecar (issue #896).
  - Real-model E2E benchmarks: `infra/scripts/run_benchmark.sh --task "..."`
    (see script header for env vars like `LLAMACPP_HOST`, `LLAMACPP_NGL`).
  - Trace backend: SQLite default (`./logs/traces.db`, WAL mode). Switch
    with `FOUNDRY_TRACE_BACKEND=jsonl` and a `.jsonl` path. Same schema
    either way (ADR-0003).
  - Dependency audit: `pip-audit` runs weekly in CI and on PRs touching
    `pyproject.toml`/`uv.lock`/`requirements*.txt`. Run locally with
    `uvx --from "pip-audit==2.10.1" pip-audit`.
- **Type discipline:** Python 3.11+. `pydantic` for all structured data
  at module boundaries (ADR-0006). No `Any` without a comment explaining why.
- **Logging:** route through `TraceLogger`, not `print()` or generic
  `logging.info` in library code. The evolution loop depends on trace events.
- **Search:** prefer `rg` over `grep`. **Shell:** prefer `workdir` over `cd`.

## 5. Commit and PR etiquette

- One logical change per commit. If a refactor is needed to make the
  feature possible, that is two commits in two PRs.
- Commit subject: 50 chars, imperative, no trailing period.
  Example: `feat(trace): persist tool-call latency histogram`.
- PR description must include: motivation, evidence (trace excerpt or
  test output), risk, and the ADR(s) it advances or supersedes.
- Keep PRs <400 lines of diff where possible. If larger, write the plan
  to `docs/adr/NNNN-...md` first.

## 6. The self-reference loop

The agent harness in this repo is written using tools shaped by the harness.
Keep the two layers strictly separate:

- **`src/foundry_x/`** — the *foundry*: Python code that wraps and evolves agents.
  `src/foundry_x/execution/runner.py` is the code that talks to the agent.
- **`harness/`** — the *artifact being evolved*: the agent's own DNA
  (`system_prompt.txt`, `hooks/`, `skills/`). All three are version-controlled
  and evolved by the Evolver→Critic loop. Skills are JSON tool definitions the
  agent can invoke; hooks are Python middleware that runs around every tool call.

Mixing these up is the most common mistake newcomers make.

## 7. Known discrepancies

- **Python version**: CI installs Python 3.14 (`uv python install 3.14` in
  workflow files). `.python-version` says `3.12`. `pyproject.toml` requires
  `>=3.11`. When in doubt, match what CI uses (3.14).
- **Stale AGENTS files**: `AGENTS_BASE_*.md`, `AGENTS_LOCAL_*.md`, and
  `AGENTS_REMOTE_*.md` in the repo root are stale backups from a previous
  session — not the canonical `AGENTS.md`.
- **README CLI table is incomplete**: `README.md` documents only three
  console scripts, but `pyproject.toml` §`[project.scripts]` registers
  seven. Treat §4 above, not the README table, as authoritative.
