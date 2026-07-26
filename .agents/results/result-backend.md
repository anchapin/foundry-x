# Backend Result — Issue #867

## Status

**PR:** [anchapin/foundry-x#887](https://github.com/anchapin/foundry-x/pull/887)
**Branch:** `fix/issue-867-add-failure-kinds` → `develop`
**Outcome:** complete; local checks green; PR opened; `Closes #867` x1.

## Summary

Added `model_error` and `hook_registry_error` to the `Digester.FAILURE_KINDS`
frozenset so the first-failure walk reports model faults and
security-degraded-hook sessions directly (was previously masked behind the
later downstream `tool_error`). Updated the canonical vocabulary in
`docs/CONTEXT.md` and added regression tests pinning the corrected
first-failure precedence and the new kinds in `FAILURE_KINDS`.

## Files changed

| Path | +/- | Purpose |
| --- | --- | --- |
| `src/foundry_x/evolution/digester.py` | +11 / -0 | Add `model_error`, `hook_registry_error` to `FAILURE_KINDS` with inline rationale citing runner.py:1592 / :675. |
| `docs/CONTEXT.md` | +14 / -4 | Extend the `FAILURE_KINDS` enumeration in the "Failure-signalling subset" subsection; required by the existing `tests/docs/test_event_kinds.py::test_failure_subset_cross_references_digester_constants` guard. |
| `tests/test_digester.py` | +159 / -1 | Extend `test_any_failure_kind_signals_via_kind_field` parametrize; add 6 regression tests covering vocabulary pin, first-failure precedence, and signal/cause contract for both new kinds. |
| `scripts/check_pr_closing_refs.sh` | +61 / -0 | Chore: developer-tool script (copied from sibling worktrees so step 12 of the workflow can run locally). |

Total: 4 files, +245 / -5 (split across 2 atomic commits — fix and chore).

## Acceptance criteria checklist

- [x] `FAILURE_KINDS` includes `model_error` and `hook_registry_error` in `src/foundry_x/evolution/digester.py`.
- [x] A session with `model_error` at step N followed by `tool_error` at step M classifies `model_error` as the first failure — pinned by `test_model_error_precedes_tool_error_in_first_failure_walk`.
- [x] A session with `hook_registry_error` flags degradation in the Digester output — pinned by `test_hook_registry_error_kind_triggers_first_failure_classification` and `test_hook_registry_error_precedes_tool_error_in_first_failure_walk`.
- [x] `uv run pytest tests/test_digester.py tests/docs/test_event_kinds.py` passes locally (115 tests, 0 failures).
- [x] `uv run ruff check src/ docs/ tests/` clean.
- [x] `uv run ruff format --check src/ docs/ tests/` clean (no formatting changes required).
- [x] PR body contains exactly one `Closes` line — verified via `bash scripts/check_pr_closing_refs.sh 887 1`.
- [x] PR targets `develop`, not `main`.

## Out of scope (per "Never widen scope", AGENTS.md §2)

- `docs/SECURITY.md` does not enumerate `FAILURE_KINDS`. It references individual kinds (`task_aborted`, `firewall_exception`) but no closed set, so no update needed for this issue.
- The pre-existing `task_aborted` source/docs drift in `FAILURE_KINDS` (CONTEXT.md says it is in the set, source does not include it) is left as a separate concern.

## Pre-existing environmental failures (NOT regressions)

`tests/execution/test_bash_skill_executor.py`, `tests/execution/test_runner_stream.py`, and the async fixtures in `tests/execution/test_runner_outcome_preservation.py` fail in this sandbox with `async def functions are not natively supported` (no `pytest-asyncio` installed). Confirmed pre-existing by `git stash` + rerun on the base commit before any change was applied.

## Blockers

None.
