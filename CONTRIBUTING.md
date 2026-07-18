# Contributing to FoundryX

Thank you for your interest in improving FoundryX. This document covers
the human side of collaboration. AI coding agents reading this should
consult [AGENTS.md](./AGENTS.md) instead.

## Code of conduct

Be kind. Disagree on substance, not on people. Assume good faith. We
are building tools that will run for a long time; we can afford to be
patient with each other.

## Ground rules

- **One PR, one concern.** Smaller is faster to review and faster to
  revert.
- **Evidence-led.** PRs that change behavior must include a test, a
  trace excerpt, or a benchmark result demonstrating the change.
- **Decisions are written down.** Any non-trivial choice gets an ADR
  in `docs/adr/` before the code lands. The PR can be opened against
  the ADR; the ADR and the code may land in the same PR.
- **The harness is evolved, not edited.** `harness/` changes must go
  through the `Evolver` -> `Critic` loop (see `docs/ROADMAP.md` and
  ADR-0004). Hand-edits require explicit justification in the PR and
  sign-off from a second human reviewer.
- **Secrets stay out of git.** `.env.example` is the template; real
  values belong in `.env` (gitignored). See `docs/SECURITY.md`.

## Development setup

```bash
git clone https://github.com/anchapin/foundry-x.git
cd foundry-x
uv sync                       # install deps
cp .env.example .env          # then edit with your model endpoint
uv run pre-commit install     # install git hooks (ruff, gitleaks, ...)
uv run pytest                 # smoke tests
uv run ruff check .           # lint
```

## Before you push

Every PR must pass the full suite (unit + benchmark) before merge.

```bash
uv run ruff check .
uv run pytest
```

## Workflow

1. **Branch** off `develop`. Naming:
    - `feat/short-name`
    - `fix/short-name`
    - `chore/short-name`
    - `docs/short-name`
    - `adr/NNNN-title`

2. **Write code + tests together.** TDD is encouraged but not
   mandated; untested code will be asked to add tests before merge.

3. **Run the local checks.**
   ```bash
   uv run pre-commit run --all-files   # ruff, secrets, hygiene
   uv run ruff check .                 # lint only
   uv run pytest                       # tests
   ```

4. **Commit** with [Conventional Commits](https://www.conventionalcommits.org/).
   Examples:
   - `feat(trace): add tool-call latency histogram`
   - `fix(evolver): prevent duplicate ProposedEdit on retry`
   - `docs(adr): record uv adoption`

5. **Push and open a PR** against `develop`. Fill out the PR template:
   motivation, evidence, risk, ADRs touched.

6. **CI must be green** before review.

7. **Merge gate.** CI must be green and the branch cleanly mergeable.
   Agent-authored PRs that do not touch `harness/` may be self-merged
   by the agent once CI passes (AGENTS.md §2). PRs containing harness
   hand-edits require two human approvals before merge.

## Adding a dependency

1. Check that it is not already transitively available.
2. Prefer libraries that work with `pydantic` v2 and that have no
   native extensions outside `uv`'s resolver.
3. Add it with `uv add <package>` — this updates both
   `pyproject.toml` and the lockfile.
4. Mention the rationale in the PR. If the choice is non-trivial,
   write an ADR (see `docs/adr/0001-record-architecture-decisions.md`).

## Adding a new skill or hook

Skills and hooks live under `harness/skills/` and `harness/hooks/`.
Because these are evolved artifacts, hand-additions must:

1. Include a unit test under `tests/harness/`.
2. Pass the Critic on the benchmark suite.
3. Be proposed as a `ProposedEdit` and merged via the evolution
   pipeline. The hand-add path is a documented escape hatch, not the
   default.

## Human review workflow for harness edits

Every proposed change to `harness/` — whether produced by the
`Evolver` or drafted by hand — must pass human review before it is
applied to the live harness. This gate exists because harness changes
directly affect agent behaviour; a bad edit can degrade capability or
introduce regressions silently (see [ADR-0004](docs/adr/0004-self-modification-guardrails.md)).

### The end-to-end loop

```
task failure → TraceLogger → Digester → Evolver → ProposedEdit
                                                   ↓
                                              Critic (automated gate)
                                                   ↓
                                         Human review → merge → harness updated
```

1. **Evolver** produces a `ProposedEdit`: a `target_file`, a
   `rationale`, and a `unified_diff` (see [ADR-0006](docs/adr/0006-pydantic-for-module-boundaries.md)).
2. **Critic** evaluates the edit in a sandbox: it copies the harness,
   applies the diff, runs `harness/scripts/load_check.py`, then runs
   the full pytest suite including `@pytest.mark.benchmark` tasks
   ([ADR-0004](docs/adr/0004-self-modification-guardrails.md),
   [ADR-0005](docs/adr/0005-pytest-as-evaluation-framework.md)).
3. A human reviewer evaluates the edit's substantive quality (see
   §Evaluating an edit below).
4. The edit is merged (or rejected), and the harness is updated.

### When to review

Review is required **before** applying any edit to `harness/`. This
applies to:

- Evolver-produced `ProposedEdit`s (automated gate passes, but the edit
  still needs a human to assess the rationale and confirm it targets the
  right file).
- Hand-drafted edits to `harness/system_prompt.txt`,
  `harness/manifest.json`, `harness/hooks/`, or `harness/skills/`.

Hand-edits require explicit justification in the PR body and a second
human approval ([ADR-0004](docs/adr/0004-self-modification-guardrails.md)).

### Evaluating an edit

When reviewing a `ProposedEdit` (or a hand-drafted harness diff),
evaluate it against these criteria:

**Correctness**

- Does the `unified_diff` apply cleanly? (`git apply` must succeed in
  the Critic sandbox; a broken patch is caught automatically, but verify
  the intent is sound.)
- Does the target file exist and is the path confined to the harness
  tree? (`system_prompt.txt`, `manifest.json`, `hooks/`, `skills/` —
  see [ADR-0012](docs/adr/0012-manifest-json-as-evolver-target.md).)
- Is the `rationale` consistent with the failure report or benchmark
  evidence? A diff with no clear connection to a documented failure is
  a red flag.

**Security** (see [ADR-0009](docs/adr/0009-security-evals-benchmark-family.md))

- Does the diff contain prompt-injection patterns?
  (`ignore previous instructions`, role-tag sequences, etc.)
  The Critic scans for these automatically, but reviewers should still
  read the diff content.
- Does the edit touch `manifest.json`? If so, verify the JSON remains
  valid and that the declared hooks/skills match the on-disk files
  ([ADR-0012](docs/adr/0012-manifest-json-as-evolver-target.md)).

**Scope**

- Is the diff focused (one concern per edit)? Large, unfocused diffs
  are harder to review and risk introducing unrelated changes.
- Does the diff stay within the line cap (default 200 lines)?
  The Critic enforces this; a reviewer flagging an unusually large
  diff should ask whether it can be split.

**Benchmarks**

- Did the Critic's `passed_checks` include the relevant benchmark tags?
  A `benchmark:security` failure means a control was weakened (see
  [ADR-0009](docs/adr/0009-security-evals-benchmark-family.md)).
- Did any previously-passing benchmark newly fail? A regression in the
  regression baseline blocks the gate ([ADR-0004](docs/adr/0004-self-modification-guardrails.md)).

### Rollback procedure

If an approved harness edit causes regressions in production or the
benchmark suite:

1. **Revert the commit** that applied the edit:
   ```bash
   git revert <commit-hash>
   git push
   ```
2. **File a failure report** in the trace store (see `foundry-trace`
   CLI) documenting what broke, so the Evolver can propose a targeted
   fix rather than re-introducing the same change.
3. **Notify the reviewer** who approved the edit, citing the regression
   evidence. Their sign-off is part of the feedback loop.
4. **Update the regression baseline** if the regression was caught by
   the Critic (the baseline file is at `logs/critic_baseline.json` per
   [ADR-0004](docs/adr/0004-self-modification-guardrails.md)).

## Reporting security issues

Please see [docs/SECURITY.md](./docs/SECURITY.md) for the disclosure
process. Do **not** file public issues for security-sensitive bugs.

## License

By contributing, you agree that your contributions will be licensed
under the project's license (see `LICENSE`).
