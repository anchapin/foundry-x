"""Single-step evolution loop chaining Digester → Evolver → Critic (issue #255).

``run_evolution_step`` is the pipeline described in CONTEXT.md "The loop"::

    Digester.digest → Evolver.propose → Critic.evaluate

Each stage is严格 ordered so the loop never calls a downstream component
unless the upstream stage emitted a non-clean signal.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from foundry_x.evolution.critic import Critic, CriticVerdict
from foundry_x.evolution.digester import Digester, FailureReport
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.execution.runner import resolve_harness_version
from foundry_x.trace.logger import TraceEvent


class EvolutionResult(BaseModel):
    """Structured result of a single evolution-step pipeline run (ADR-0006).

    Returned by :func:`run_evolution_step`. The ``verdict`` field is only
    present when the pipeline ran the full Digester → Evolver → Critic chain;
    it is absent when the report is clean (short-circuit) or when the
    Evolver returned no edits.

    Issue #604 adds ``evolver_duration_ms``: wall-clock milliseconds spent
    inside ``evolver.propose()``, measured via ``time.time()`` deltas. It
    is ``None`` when the evolver is not reached (clean report or no edits).

    Issue #609 adds ``started_at`` and ``completed_at`` ISO-8601 timestamps
    stamped around the pipeline for KPI history trend computation.
    """

    session_id: str
    failure_report: FailureReport
    failure_class: str = Field(
        description="Copied from failure_report.proposed_class for KPI attribution"
    )
    proposed_edits: list[ProposedEdit] = Field(default_factory=list)
    verdict: CriticVerdict | None = None
    evolver_duration_ms: float | None = None
    harness_version: str | None = None
    started_at: str
    completed_at: str


def _edits_to_diff(edits: list[ProposedEdit]) -> str:
    """Concatenate a list of ProposedEdit unified diffs into one patch string."""
    if not edits:
        return ""
    return "\n".join(edit.unified_diff for edit in edits)


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with offset suffix.

    Consistent with :func:`foundry_x.observability.kpis._now_iso` —
    timezone-aware form keeps the line unambiguous across CI regions;
    ``datetime.fromisoformat`` (Python 3.11+) accepts the ``+00:00``
    suffix without modification.
    """
    return datetime.now(timezone.utc).isoformat()


def run_evolution_step(
    session_id: str,
    events: list[TraceEvent],
    harness_dir: Path,
    *,
    critic: Critic | None = None,
    evolver: Evolver | None = None,
    no_verify: bool = False,
) -> EvolutionResult:
    """Run one iteration of the evolution loop over a session's trace events.

    Pipeline (CONTEXT.md "The loop"):

        Digester.digest → Evolver.propose → Critic.evaluate

    The Critic is only invoked when the Digester returns a non-clean report
    *and* the Evolver returns at least one ProposedEdit. A clean report
    (no failure detected) short-circuits the loop and returns immediately
    with an empty ``proposed_edits`` list and ``verdict=None``.

    Parameters
    ----------
    session_id:
        Identifier of the session these events belong to.
    events:
        Ordered list of :class:`TraceEvent` objects for the session.
    harness_dir:
        Path to the live harness directory. The Critic works on a sandbox copy;
        this argument is only used to construct the :class:`Critic` instance
        when ``critic=None``.
    critic:
        Optional :class:`Critic` instance. When omitted a default instance is
        constructed from ``harness_dir``.
    evolver:
        Optional :class:`Evolver` instance. When omitted a default instance is
        constructed. The Evolver is only used when the failure report is not
        clean; a clean report short-circuits before any proposal work.
    no_verify:
        When ``True``, skip the Critic gate and return a synthetic
        ``CriticVerdict(verdict=None, notes="--no-verify: skipped")`` for the
        last proposed edit (issue #888). The audit trail still records a
        verdict event; downstream consumers treat ``None`` as a non-approval.

    Returns
    -------
    EvolutionResult
        A pydantic model containing the failure report, proposed edits (if any),
        and the critic verdict (if the full chain ran).
    """
    harness_version = resolve_harness_version(harness_dir)
    started_at = _now_iso()
    failure_report = Digester().digest(session_id, events)

    if failure_report.proposed_class == "clean":
        return EvolutionResult(
            session_id=session_id,
            failure_report=failure_report,
            failure_class=failure_report.proposed_class,
            proposed_edits=[],
            verdict=None,
            evolver_duration_ms=None,
            harness_version=harness_version,
            started_at=started_at,
            completed_at=_now_iso(),
        )

    if evolver is None:
        evolver = Evolver()

    evolver_duration_ms: float | None = None
    try:
        t0 = time.time()
        proposed_edits = evolver.propose(
            harness_dir=harness_dir,
            failure=failure_report,
            current_diff=None,
        )
        evolver_duration_ms = (time.time() - t0) * 1000
    except NotImplementedError:
        proposed_edits = []

    if not proposed_edits:
        return EvolutionResult(
            session_id=session_id,
            failure_report=failure_report,
            failure_class=failure_report.proposed_class,
            proposed_edits=[],
            verdict=None,
            evolver_duration_ms=evolver_duration_ms,
            harness_version=harness_version,
            started_at=started_at,
            completed_at=_now_iso(),
        )

    if critic is None and not no_verify:
        critic = Critic(harness_dir=harness_dir)

    verdict = None
    if no_verify:
        # Skip the Critic gate but preserve the audit trail (issue #888).
        # The synthetic verdict carries ``edit_index`` of the last edit so
        # downstream consumers can correlate it with the proposed edit.
        for idx, edit in enumerate(proposed_edits):
            verdict = CriticVerdict(
                verdict=None,
                passed_checks=[],
                failed_checks=[],
                notes="--no-verify: skipped",
                edit_index=idx,
                failure_class=failure_report.proposed_class,
            )
    else:
        for idx, edit in enumerate(proposed_edits):
            verdict = critic.evaluate(
                edit.unified_diff, edit_index=idx, failure_class=failure_report.proposed_class
            )

    return EvolutionResult(
        session_id=session_id,
        failure_report=failure_report,
        failure_class=failure_report.proposed_class,
        proposed_edits=proposed_edits,
        verdict=verdict,
        evolver_duration_ms=evolver_duration_ms,
        harness_version=harness_version,
        started_at=started_at,
        completed_at=_now_iso(),
    )


async def run_evolution_step_async(
    session_id: str,
    events: list[TraceEvent],
    harness_dir: Path,
    *,
    critic: Critic | None = None,
    evolver: Evolver | None = None,
    no_verify: bool = False,
) -> EvolutionResult:
    """Async variant of :func:`run_evolution_step`.

    Awaits ``evolver.propose_async()`` instead of calling ``evolver.propose()``.
    The Critic is still invoked synchronously because the subprocess call is
    inherently blocking (ADR-0010).

    When ``no_verify=True`` the Critic is skipped and a synthetic
    ``CriticVerdict(verdict=None, notes="--no-verify: skipped")`` is returned
    for the last proposed edit (issue #888).
    """
    harness_version = resolve_harness_version(harness_dir)
    started_at = _now_iso()
    failure_report = Digester().digest(session_id, events)

    if failure_report.proposed_class == "clean":
        return EvolutionResult(
            session_id=session_id,
            failure_report=failure_report,
            failure_class=failure_report.proposed_class,
            proposed_edits=[],
            verdict=None,
            evolver_duration_ms=None,
            harness_version=harness_version,
            started_at=started_at,
            completed_at=_now_iso(),
        )

    if evolver is None:
        evolver = Evolver()

    evolver_duration_ms: float | None = None
    try:
        t0 = time.time()
        proposed_edits = await evolver.propose_async(
            harness_dir=harness_dir,
            failure=failure_report,
            current_diff=None,
        )
        evolver_duration_ms = (time.time() - t0) * 1000
    except NotImplementedError:
        proposed_edits = []

    if not proposed_edits:
        return EvolutionResult(
            session_id=session_id,
            failure_report=failure_report,
            failure_class=failure_report.proposed_class,
            proposed_edits=[],
            verdict=None,
            evolver_duration_ms=evolver_duration_ms,
            harness_version=harness_version,
            started_at=started_at,
            completed_at=_now_iso(),
        )

    if critic is None and not no_verify:
        critic = Critic(harness_dir=harness_dir)

    verdict = None
    if no_verify:
        # Skip the Critic gate but preserve the audit trail (issue #888).
        for idx, edit in enumerate(proposed_edits):
            verdict = CriticVerdict(
                verdict=None,
                passed_checks=[],
                failed_checks=[],
                notes="--no-verify: skipped",
                edit_index=idx,
                failure_class=failure_report.proposed_class,
            )
    else:
        for idx, edit in enumerate(proposed_edits):
            verdict = critic.evaluate(edit.unified_diff, edit_index=idx)

    return EvolutionResult(
        session_id=session_id,
        failure_report=failure_report,
        failure_class=failure_report.proposed_class,
        proposed_edits=proposed_edits,
        verdict=verdict,
        evolver_duration_ms=evolver_duration_ms,
        harness_version=harness_version,
        started_at=started_at,
        completed_at=_now_iso(),
    )
