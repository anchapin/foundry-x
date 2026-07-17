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

## Reporting security issues

Please see [docs/SECURITY.md](./docs/SECURITY.md) for the disclosure
process. Do **not** file public issues for security-sensitive bugs.

## License

By contributing, you agree that your contributions will be licensed
under the project's license (see `LICENSE`).
