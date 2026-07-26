## Motivation

CONTEXT.md §"Failure-signalling subset" cites two digester constants with line-number ranges that no longer match the source file. `FAILURE_KINDS` is cited at `digester.py:60-69` but actually lives at lines 61-88; `FAILURE_PAYLOAD_KEYS` is cited at `digester.py:70-76` but actually lives at lines 90-96. The constants grew when `task_aborted` (#901), `model_error` (#867), and `hook_registry_error` (#867) were added to `FAILURE_KINDS`, shifting every line down by ~20.

A developer following CONTEXT.md to locate the failure vocabulary now lands on the wrong lines, increasing time from failure discovery to fix. The existing test `tests/docs/test_event_kinds.py` verifies the vocabulary *values* (set membership) but not the *line-number citations*, so the drift passes CI undetected.

This follows the precedent of issues #33 and #34, where documentation accuracy was filed under `kpi-cycle-time` because correct docs directly reduce the time from failure discovery to fix proposal.

## Evidence

- `docs/CONTEXT.md:184` — cites `FAILURE_KINDS` at `digester.py:60-69`
- `src/foundry_x/evolution/digester.py:61-88` — actual `FAILURE_KINDS` range (frozenset at 61, closing paren at 88)
- `docs/CONTEXT.md:205` — cites `FAILURE_PAYLOAD_KEYS` at `digester.py:70-76`
- `src/foundry_x/evolution/digester.py:90-96` — actual `FAILURE_PAYLOAD_KEYS` range (frozenset at 90, closing paren at 96)
- `tests/docs/test_event_kinds.py` — imports `FAILURE_KINDS`/`FAILURE_PAYLOAD_KEYS` and checks set membership, no line-citation validation

## Risk

Low. Docs-only correction + one regression test. No code or behavior change.

## Acceptance Criteria

1. CONTEXT.md cites `FAILURE_KINDS` at the exact line range where the frozenset definition begins and ends
2. CONTEXT.md cites `FAILURE_PAYLOAD_KEYS` at the exact line range where the frozenset definition begins and ends
3. A regression test in `tests/docs/test_event_kinds.py` parses the `digester.py:NN-NN` citation pattern from CONTEXT.md and asserts the cited range contains the named constant definition
4. Test fails if a future constant extension shifts lines without a CONTEXT.md update

## ADR(s)

ADR-0001 — complies (documentation correction)
ADR-0008 — complies (conventional-commits discipline)
