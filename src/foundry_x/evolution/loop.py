"""Single-step evolution loop chaining Digester → Evolver → Critic (issue #255).

``run_evolution_step`` is the pipeline described in CONTEXT.md "The loop"::

    Digester.digest → Evolver.propose → Critic.evaluate

Each stage is严格 ordered so the loop never calls a downstream component
unless the upstream stage emitted a non-clean signal.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from foundry_x.evolution.critic import Critic, CriticVerdict
from foundry_x.evolution.digester import Digester, FailureReport
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.trace.logger import TraceEvent


class EvolutionResult(BaseModel):
    """Structured result of a single evolution-step pipeline run (ADR-0006).

    Returned by :func:`run_evolution_step`. The ``verdict`` field is only
    present when the pipeline ran the full Digester → Evolver → Critic chain;
    it is absent when the report is clean (short-circuit) or when the
    Evolver returned no edits.
    """

    session_id: str
    failure_report: FailureReport
    proposed_edits: list[ProposedEdit] = Field(default_factory=list)
    verdict: CriticVerdict | None = None


def _edits_to_diff(edits: list[ProposedEdit]) -> str:
    """Concatenate a list of ProposedEdit unified diffs into one patch string."""
    if not edits:
        return ""
    return "\n".join(edit.unified_diff for edit in edits)


def run_evolution_step(
    session_id: str,
    events: list[TraceEvent],
    harness_dir: Path,
    *,
    critic: Critic | None = None,
    evolver: Evolver | None = None,
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

    Returns
    -------
    EvolutionResult
        A pydantic model containing the failure report, proposed edits (if any),
        and the critic verdict (if the full chain ran).
    """
    failure_report = Digester().digest(session_id, events)

    if failure_report.proposed_class == "clean":
        return EvolutionResult(
            session_id=session_id,
            failure_report=failure_report,
            proposed_edits=[],
            verdict=None,
        )

    if evolver is None:
        evolver = Evolver()

    try:
        proposed_edits = evolver.propose(
            harness_dir=harness_dir,
            failure=failure_report,
            current_diff=None,
        )
    except NotImplementedError:
        proposed_edits = []

    if not proposed_edits:
        return EvolutionResult(
            session_id=session_id,
            failure_report=failure_report,
            proposed_edits=[],
            verdict=None,
        )

    if critic is None:
        critic = Critic(harness_dir=harness_dir)

    verdict = None
    for idx, edit in enumerate(proposed_edits):
        verdict = critic.evaluate(edit.unified_diff, edit_index=idx)

    return EvolutionResult(
        session_id=session_id,
        failure_report=failure_report,
        proposed_edits=proposed_edits,
        verdict=verdict,
    )
