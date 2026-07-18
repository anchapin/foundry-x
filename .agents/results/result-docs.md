# Documentation Curation Result

## Status

BLOCKED — the required `oma` CLI is not installed or available on `PATH` in this worktree environment.

## Summary

- Read issue #875 and its comments (no comments were present).
- Read the repository guidance, `docs/SECURITY.md`, `docs/ARCHITECTURE.md`, `docs/OPERATOR.md`, and ADR-0004 before considering documentation changes.
- Diff intake: cached diff was empty; fallback range was `HEAD~1..HEAD` (the preceding commit changed `AGENTS.md`).
- `oma docs verify --json`: could not run (`command not found: oma`).
- `oma docs sync HEAD~1..HEAD --json`: could not run (`command not found: oma`).
- Per the documentation-curation workflow, no manual grep-based substitute was used and no documentation patch was applied.

## Drift counts

- Before: unavailable; the mandatory `oma docs verify --json` baseline could not execute.
- After: not run; no documentation changes were applied.

## Files changed

- None in the requested documentation scope.
- This required result artifact: `.agents/results/result-docs.md`.

## Acceptance criteria checklist

- [ ] Documentation clearly explains the degradation mode — blocked because `oma` is unavailable and the workflow requires exiting rather than substituting manual drift analysis.
- [ ] Documentation lists affected security controls — not applied.
- [ ] Documentation provides detection and recovery steps — not applied.
- [x] `uv run ruff check docs/` — passed (no Python files under `docs/`).
- [x] `uv run ruff format --check docs/` — passed (no Python files under `docs/`).
- [ ] `oma docs verify --json` baseline and re-verification — blocked by missing CLI.
- [ ] Commit, push, PR creation, and closing-reference verification — not attempted because the required curation preflight could not complete.

## Blocker / recovery

Install or expose the repository's `oma` CLI, then rerun `oma docs verify --json` and `oma docs sync <range> --json` from this worktree. No PR number exists because no commit or PR was created.
