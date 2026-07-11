"""Pinning tests for the three Evolver guardrails (issue #94).

Issues #17 and #18 added the guardrails themselves (rate-limit, diff-size,
target-tree confinement) but did not pin the *behavior* with regression
tests covering the exact failure modes called out in the acceptance
criteria of #94:

1. A ``target_file`` of ``'../src/foundry_x/x.py'`` or any absolute path
   must be rejected at the pydantic boundary (ADR-0006) so an out-of-tree
   edit cannot reach the Critic.
2. A unified diff whose line count exceeds ``max_diff_lines`` must be
   rejected with ``EvolverGuardError`` before any proposal work runs.
3. ``Evolver.propose()`` must honor the per-run rate limit: the
   ``(max_proposals_per_hour + 1)``-th call inside the rolling window
   raises ``EvolverGuardError`` — never a partial edit, never silently
   accepted.

Each test below names one criterion and uses the exact input string the
issue called out, so a regression that loosens any of the three guards
flips one of these tests red.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from foundry_x.evolution.evolver import (
    Evolver,
    EvolverGuardError,
    ProposedEdit,
)

# A minimal non-blank unified diff; these tests pin guardrails, not the
# diff parser. ``+one\n`` is the smallest legal body the pydantic model
# accepts and keeps the assertions independent of any future diff parser
# changes.
_TINY_DIFF = "+one\n"


# ---------------------------------------------------------------------------
# Criterion 1 — target_file confinement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "target_file",
    [
        # Exact string from the issue acceptance criteria.
        "../src/foundry_x/x.py",
        # ``harness/../`` resolves back out of the harness tree — must
        # not silently land inside ``harness/``.
        "harness/../src/foundry_x/evolution/evolver.py",
        # Absolute path: anywhere on the host filesystem is off-limits.
        "/etc/passwd",
        "/absolute/path/harness/hooks/a.py",
    ],
)
def test_proposed_edit_rejects_out_of_tree_target_file(target_file: str) -> None:
    """A ``target_file`` outside ``harness/{system_prompt.txt,hooks/,skills/}``
    raises ``ValidationError`` at the pydantic boundary.

    Pin for issue #94 acceptance criterion 1. The Evolver must never let
    an edit target the project source tree, the host filesystem, or any
    other non-harness location reach the Critic.
    """
    with pytest.raises(ValidationError):
        ProposedEdit(
            target_file=target_file,
            rationale="x",
            unified_diff=_TINY_DIFF,
        )


@pytest.mark.parametrize(
    "target_file",
    [
        "harness/system_prompt.txt",
        "harness/hooks/a.py",
        "harness/hooks/sub/b.py",
        "harness/skills/x.md",
        "harness/skills/nested/y.md",
    ],
)
def test_proposed_edit_accepts_harness_tree_target_file(target_file: str) -> None:
    """A ``target_file`` inside ``harness/{system_prompt.txt,hooks/,skills/}``
    is accepted and canonicalised.

    Companion to the rejection cases above: the confinement is *positive*
    as well — the three allowed shapes round-trip exactly. Without this
    pair the rejection cases could pass for a model that simply refuses
    every input.
    """
    edit = ProposedEdit(
        target_file=target_file,
        rationale="x",
        unified_diff=_TINY_DIFF,
    )
    assert edit.target_file == target_file


# ---------------------------------------------------------------------------
# Criterion 2 — diff-size guardrail
# ---------------------------------------------------------------------------


def test_diff_exceeding_line_cap_raises_evolver_guard_error() -> None:
    """A diff whose line count exceeds ``max_diff_lines`` is rejected.

    Pin for issue #94 acceptance criterion 2. With ``max_diff_lines=5``,
    a 6-line diff must surface ``EvolverGuardError``; a 5-line diff at
    the cap is the boundary and must still pass.
    """
    evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=5)
    oversized = "\n".join(f"+line {i}" for i in range(6))
    edit = ProposedEdit(
        target_file="harness/hooks/a.py",
        rationale="x",
        unified_diff=oversized,
    )
    with pytest.raises(EvolverGuardError, match="diff too large"):
        evolver._validate_edit(edit)


def test_diff_at_exact_line_cap_passes() -> None:
    """A diff whose line count equals ``max_diff_lines`` is accepted.

    Boundary case: ``>=`` semantics on the cap. Without this test a
    regression that flips the comparison to ``>=`` would silently
    shrink the allowed diff size by one line on every evolution run.
    """
    evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=5)
    at_cap = "\n".join(f"+l{i}" for i in range(5))
    edit = ProposedEdit(
        target_file="harness/hooks/a.py",
        rationale="x",
        unified_diff=at_cap,
    )
    evolver._validate_edit(edit)  # must not raise


# ---------------------------------------------------------------------------
# Criterion 3 — propose() per-run rate limit
# ---------------------------------------------------------------------------


def test_propose_n_plus_first_call_raises_rate_limit_error() -> None:
    """The ``(max_proposals_per_hour + 1)``-th ``propose()`` within the
    rolling window raises ``EvolverGuardError``.

    Pin for issue #94 acceptance criterion 3. With
    ``max_proposals_per_hour=2``, two pre-existing timestamps fill the
    window; the next ``propose()`` must be rejected by the guard *before*
    any body work runs (the body currently raises ``NotImplementedError``,
    so the ``EvolverGuardError`` here proves the guard fired first).
    """
    evolver = Evolver(max_proposals_per_hour=2, max_diff_lines=200)
    evolver._record_proposals(2)  # fill the window to the cap
    with pytest.raises(EvolverGuardError, match="rate limit exceeded"):
        evolver.propose(Path("/nonexistent/harness"), failure=object())


def test_propose_under_cap_reaches_body_not_guard() -> None:
    """With headroom in the rate-limit window, ``propose()`` reaches its
    body rather than being blocked by the guard.

    Companion to the over-cap case: this test pins the *negative* of
    criterion 3 — the guard does not over-fire. ``NotImplementedError``
    (today's body) surfacing proves the guard ran AND passed AND yielded
    control to the body.
    """
    evolver = Evolver(max_proposals_per_hour=3, max_diff_lines=200)
    with pytest.raises(NotImplementedError):
        evolver.propose(Path("/nonexistent/harness"), failure=object())


# ---------------------------------------------------------------------------
# Defaults — referenced by SECURITY.md §"Rate limits"; pin them so a
# silent loosening of the cap is caught at PR time.
# ---------------------------------------------------------------------------


def test_evolver_defaults_match_security_doc() -> None:
    """The default caps mirror SECURITY.md: 10 proposals/hour, 200 lines.

    A regression that raises either default diverges from the prose and
    from this test. The Critic workflow references SECURITY.md directly,
    so any drift here is also a documentation/code inconsistency.
    """
    evolver = Evolver()
    assert evolver.max_proposals_per_hour == 10
    assert evolver.max_diff_lines == 200
