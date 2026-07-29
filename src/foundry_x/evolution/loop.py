"""Single-step evolution loop chaining Digester → Evolver → Critic (issue #255).

``run_evolution_step`` is the pipeline described in CONTEXT.md "The loop"::

    Digester.digest → Evolver.propose → Critic.evaluate

Each stage is严格 ordered so the loop never calls a downstream component
unless the upstream stage emitted a non-clean signal.
"""

from __future__ import annotations

import signal
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from foundry_x.evolution.critic import Critic, CriticVerdict
from foundry_x.evolution.digester import (
    BatchFailureReport,
    Digester,
    FailureReport,
    context_hash_bucket,
)
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.evolution.store import FailurePatternStore
from foundry_x.execution.runner import resolve_harness_version
from foundry_x.observability.regression_report import record_verdict
from foundry_x.trace.logger import TraceEvent, TraceLogger

EVOLVER_DURATION_KIND = "evolver_duration"


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


class BatchEvolutionResult(BaseModel):
    """Structured result of a batch evolution-step pipeline run (issue #1033).

    Returned by :func:`run_evolution_batch`. The ``results`` field contains
    individual :class:`EvolutionResult` objects for each failure class in the batch.
    The ``total_failures`` field reflects how many distinct failure classes were
    found and processed.
    """

    session_id: str
    batch_report: BatchFailureReport
    results: list[EvolutionResult] = Field(default_factory=list)
    total_failures: int = 0
    proposed_edits: list[ProposedEdit] = Field(default_factory=list)
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
    return datetime.now(UTC).isoformat()


def _emit_evolver_duration(
    trace_logger: TraceLogger | None,
    evolver: Evolver,
    session_id: str,
    failure_class: str,
    evolver_duration_ms: float,
    proposed_edits: list[ProposedEdit],
) -> None:
    """Emit an evolver_duration trace event if a logger is available.

    Uses the ``trace_logger`` argument when provided; otherwise falls back
    to ``evolver._trace_logger`` if the evolver has one attached.
    """
    logger = trace_logger if trace_logger is not None else evolver._trace_logger
    if logger is None:
        return
    logger.record(
        session_id,
        EVOLVER_DURATION_KIND,
        {
            "evolver_duration_ms": evolver_duration_ms,
            "failure_class": failure_class,
            "proposed_edits_count": len(proposed_edits),
        },
    )


def _record_and_annotate_pattern(
    failure_report: FailureReport,
    store: FailurePatternStore,
) -> None:
    """Record a failure into the pattern store and annotate the report.

    Mutates ``failure_report.seen_across_n_sessions`` in place so the
    Evolver can read the recurrence count from the report it already
    receives (ADR-0030, issue #1038).

    The timestamp used for the pattern row is taken from the first
    failed step (if available) or the current time as fallback.
    """
    bucket = context_hash_bucket(failure_report)
    timestamp = _now_iso()
    if failure_report.failed_steps:
        ts = failure_report.failed_steps[0].get("timestamp")
        if isinstance(ts, str) and ts:
            timestamp = ts
    store.record(
        proposed_class=failure_report.proposed_class,
        session_id=failure_report.session_id,
        timestamp=timestamp,
        context_hash=bucket,
    )
    pattern = store.find_pattern(
        proposed_class=failure_report.proposed_class,
        context_hash=bucket,
    )
    if pattern is not None:
        failure_report.seen_across_n_sessions = pattern.session_count


def run_evolution_step(
    session_id: str,
    events: list[TraceEvent],
    harness_dir: Path,
    *,
    critic: Critic | None = None,
    evolver: Evolver | None = None,
    no_verify: bool = False,
    trace_logger: TraceLogger | None = None,
    critic_tier: Literal["smoke", "full"] = "full",
    failure_pattern_store: FailurePatternStore | None = None,
) -> EvolutionResult:
    """Run one iteration of the evolution loop over a session's trace events.

    Pipeline (CONTEXT.md "The loop"):

        Digester.digest → Evolver.propose → Critic.evaluate

    The Critic is only invoked when the Digester returns a non-clean report
    *and* the Evolver returns at least one ProposedEdit. A clean report
    (no failure detected) short-circuits the loop and returns immediately
    with an empty ``proposed_edits`` list and ``verdict=None``.

    When a :class:`FailurePatternStore` is provided, non-clean failure
    reports are recorded into it and queried for cross-session recurrence
    before the Evolver runs (ADR-0030, issue #1038). The
    ``seen_across_n_sessions`` field on the annotated report lets the
    Evolver prefer structural (hook-based) fixes for recurring patterns.

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
    trace_logger:
        Optional :class:`TraceLogger` passed to the default
        :class:`Evolver` so template-fallback failures (e.g. unknown failure
        class) emit ``generation_attempt`` / ``generation_exhausted`` trace
        events instead of returning ``[]`` silently (issue #974). Ignored
        when ``evolver`` is explicitly provided.
    critic_tier:
        Which benchmark subset the Critic pytest gate runs (issue #1042).
        ``"full"`` (default) runs the entire ``@pytest.mark.benchmark``
        suite — unchanged historical behaviour. ``"smoke"`` runs only the
        smoke-tagged subset so an obvious rejection fast-fails without the
        full-suite cost, lowering ``kpi-cycle-time``.
    failure_pattern_store:
        Optional :class:`FailurePatternStore` for cross-session pattern
        accumulation (ADR-0030). When provided, non-clean reports are
        recorded and queried for recurrence before the Evolver runs.

    Returns
    -------
    EvolutionResult
        A pydantic model containing the failure report, proposed edits (if any),
        and the critic verdict (if the full chain ran).
    """
    harness_version = resolve_harness_version(harness_dir).version
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

    if failure_pattern_store is not None:
        _record_and_annotate_pattern(failure_report, failure_pattern_store)

    if evolver is None:
        evolver = Evolver(trace_logger=trace_logger, session_id=session_id)

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

    if evolver_duration_ms is not None:
        _emit_evolver_duration(
            trace_logger=trace_logger,
            evolver=evolver,
            session_id=session_id,
            failure_class=failure_report.proposed_class,
            evolver_duration_ms=evolver_duration_ms,
            proposed_edits=proposed_edits,
        )

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
                target_file=edit.target_file,
            )
    else:
        for idx, edit in enumerate(proposed_edits):
            verdict = critic.evaluate(
                edit.unified_diff,
                edit_index=idx,
                failure_class=failure_report.proposed_class,
                tier=critic_tier,
            )
            verdict.target_file = edit.target_file

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
    trace_logger: TraceLogger | None = None,
    critic_tier: Literal["smoke", "full"] = "full",
    failure_pattern_store: FailurePatternStore | None = None,
) -> EvolutionResult:
    """Async variant of :func:`run_evolution_step`.

    Awaits ``evolver.propose_async()`` instead of calling ``evolver.propose()``.
    The Critic is still invoked synchronously because the subprocess call is
    inherently blocking (ADR-0010).

    When ``no_verify=True`` the Critic is skipped and a synthetic
    ``CriticVerdict(verdict=None, notes="--no-verify: skipped")`` is returned
    for the last proposed edit (issue #888).

    ``trace_logger`` is passed to the default :class:`Evolver` so
    template-fallback failures emit trace events (issue #974).

    ``failure_pattern_store`` enables cross-session pattern accumulation
    (ADR-0030, issue #1038).
    """
    harness_version = resolve_harness_version(harness_dir).version
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

    if failure_pattern_store is not None:
        _record_and_annotate_pattern(failure_report, failure_pattern_store)

    if evolver is None:
        evolver = Evolver(trace_logger=trace_logger, session_id=session_id)

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

    if evolver_duration_ms is not None:
        _emit_evolver_duration(
            trace_logger=trace_logger,
            evolver=evolver,
            session_id=session_id,
            failure_class=failure_report.proposed_class,
            evolver_duration_ms=evolver_duration_ms,
            proposed_edits=proposed_edits,
        )

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
                target_file=edit.target_file,
            )
    else:
        for idx, edit in enumerate(proposed_edits):
            verdict = critic.evaluate(
                edit.unified_diff,
                edit_index=idx,
                failure_class=failure_report.proposed_class,
                tier=critic_tier,
            )
            verdict.target_file = edit.target_file

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


def run_evolution_batch(
    session_id: str,
    events: list[TraceEvent],
    harness_dir: Path,
    *,
    critic: Critic | None = None,
    evolver: Evolver | None = None,
    no_verify: bool = False,
    trace_logger: TraceLogger | None = None,
    critic_tier: Literal["smoke", "full"] = "full",
    failure_pattern_store: FailurePatternStore | None = None,
) -> BatchEvolutionResult:
    """Run batch evolution processing over a session's trace events (issue #1033).

    Unlike :func:`run_evolution_step` which processes only the first failure,
    this function processes all failures found in the session and proposes
    edits for each distinct failure class simultaneously.

    Pipeline::

        Digester.digest_batch → Evolver.propose_batch → Critic.evaluate (per failure)

    The Critic is invoked for each failure class that produces proposed edits.

    Parameters
    ----------
    session_id:
        Identifier of the session these events belong to.
    events:
        Ordered list of :class:`TraceEvent` objects for the session.
    harness_dir:
        Path to the live harness directory.
    critic:
        Optional :class:`Critic` instance.
    evolver:
        Optional :class:`Evolver` instance. When omitted a default instance is
        constructed.
    no_verify:
        When ``True``, skip the Critic gate.
    trace_logger:
        Optional :class:`TraceLogger` passed to the default :class:`Evolver`.
    critic_tier:
        Which benchmark subset the Critic pytest gate runs.
    failure_pattern_store:
        Optional :class:`FailurePatternStore` for cross-session pattern
        accumulation.

    Returns
    -------
    BatchEvolutionResult
        A pydantic model containing the batch report, individual results per
        failure class, and all proposed edits.
    """
    harness_version = resolve_harness_version(harness_dir).version
    started_at = _now_iso()
    batch_report = Digester().digest_batch(session_id, events)

    if evolver is None:
        evolver = Evolver(trace_logger=trace_logger, session_id=session_id)

    for failure_report in batch_report.failure_reports:
        if failure_report.proposed_class == "clean":
            continue
        if failure_pattern_store is not None:
            _record_and_annotate_pattern(failure_report, failure_pattern_store)

    evolver_duration_ms: float | None = None
    batch_edits: list[ProposedEdit] = []
    try:
        t0 = time.time()
        batch_edits = evolver.propose_batch(
            harness_dir=harness_dir,
            batch_report=batch_report,
            current_diff=None,
        )
        evolver_duration_ms = (time.time() - t0) * 1000
    except NotImplementedError:
        batch_edits = []

    target_to_edit: dict[str, ProposedEdit] = {}
    for edit in batch_edits:
        if edit.target_file not in target_to_edit:
            target_to_edit[edit.target_file] = edit

    use_batch_attribution = bool(batch_edits)

    results: list[EvolutionResult] = []
    all_edits: list[ProposedEdit] = []

    for failure_report in batch_report.failure_reports:
        if failure_report.proposed_class == "clean":
            continue

        proposed_edits = evolver.propose(
            harness_dir=harness_dir,
            failure=failure_report,
            current_diff=None,
        )

        if use_batch_attribution:
            failure_edits = []
            for edit in proposed_edits:
                if edit.target_file in target_to_edit and target_to_edit[edit.target_file] == edit:
                    failure_edits.append(edit)
        else:
            failure_edits = proposed_edits

        if evolver_duration_ms is not None:
            _emit_evolver_duration(
                trace_logger=trace_logger,
                evolver=evolver,
                session_id=session_id,
                failure_class=failure_report.proposed_class,
                evolver_duration_ms=evolver_duration_ms,
                proposed_edits=failure_edits,
            )

        if not failure_edits:
            results.append(
                EvolutionResult(
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
            )
            continue

        verdict = None
        if no_verify:
            for idx, edit in enumerate(failure_edits):
                verdict = CriticVerdict(
                    verdict=None,
                    passed_checks=[],
                    failed_checks=[],
                    notes="--no-verify: skipped",
                    edit_index=idx,
                    failure_class=failure_report.proposed_class,
                    target_file=edit.target_file,
                )
        else:
            if critic is None:
                critic = Critic(harness_dir=harness_dir)
            for idx, edit in enumerate(failure_edits):
                verdict = critic.evaluate(
                    edit.unified_diff,
                    edit_index=idx,
                    failure_class=failure_report.proposed_class,
                    tier=critic_tier,
                )
                verdict.target_file = edit.target_file

        results.append(
            EvolutionResult(
                session_id=session_id,
                failure_report=failure_report,
                failure_class=failure_report.proposed_class,
                proposed_edits=failure_edits,
                verdict=verdict,
                evolver_duration_ms=evolver_duration_ms,
                harness_version=harness_version,
                started_at=started_at,
                completed_at=_now_iso(),
            )
        )
        all_edits.extend(failure_edits)

    return BatchEvolutionResult(
        session_id=session_id,
        batch_report=batch_report,
        results=results,
        total_failures=batch_report.total_failures,
        proposed_edits=all_edits,
        harness_version=harness_version,
        started_at=started_at,
        completed_at=_now_iso(),
    )


async def run_evolution_batch_async(
    session_id: str,
    events: list[TraceEvent],
    harness_dir: Path,
    *,
    critic: Critic | None = None,
    evolver: Evolver | None = None,
    no_verify: bool = False,
    trace_logger: TraceLogger | None = None,
    critic_tier: Literal["smoke", "full"] = "full",
    failure_pattern_store: FailurePatternStore | None = None,
) -> BatchEvolutionResult:
    """Async variant of :func:`run_evolution_batch`.

    Awaits ``evolver.propose_batch_async()`` with the full batch report.
    """
    harness_version = resolve_harness_version(harness_dir).version
    started_at = _now_iso()
    batch_report = Digester().digest_batch(session_id, events)

    if evolver is None:
        evolver = Evolver(trace_logger=trace_logger, session_id=session_id)

    for failure_report in batch_report.failure_reports:
        if failure_report.proposed_class == "clean":
            continue
        if failure_pattern_store is not None:
            _record_and_annotate_pattern(failure_report, failure_pattern_store)

    evolver_duration_ms: float | None = None
    batch_edits: list[ProposedEdit] = []
    try:
        t0 = time.time()
        batch_edits = await evolver.propose_batch_async(
            harness_dir=harness_dir,
            batch_report=batch_report,
            current_diff=None,
        )
        evolver_duration_ms = (time.time() - t0) * 1000
    except NotImplementedError:
        batch_edits = []

    target_to_edit: dict[str, ProposedEdit] = {}
    for edit in batch_edits:
        if edit.target_file not in target_to_edit:
            target_to_edit[edit.target_file] = edit

    use_batch_attribution = bool(batch_edits)

    results: list[EvolutionResult] = []
    all_edits: list[ProposedEdit] = []

    for failure_report in batch_report.failure_reports:
        if failure_report.proposed_class == "clean":
            continue

        proposed_edits = evolver.propose(
            harness_dir=harness_dir,
            failure=failure_report,
            current_diff=None,
        )

        if use_batch_attribution:
            failure_edits = []
            for edit in proposed_edits:
                if edit.target_file in target_to_edit and target_to_edit[edit.target_file] == edit:
                    failure_edits.append(edit)
        else:
            failure_edits = proposed_edits

        if evolver_duration_ms is not None:
            _emit_evolver_duration(
                trace_logger=trace_logger,
                evolver=evolver,
                session_id=session_id,
                failure_class=failure_report.proposed_class,
                evolver_duration_ms=evolver_duration_ms,
                proposed_edits=failure_edits,
            )

        if not failure_edits:
            results.append(
                EvolutionResult(
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
            )
            continue

        verdict = None
        if no_verify:
            for idx, edit in enumerate(failure_edits):
                verdict = CriticVerdict(
                    verdict=None,
                    passed_checks=[],
                    failed_checks=[],
                    notes="--no-verify: skipped",
                    edit_index=idx,
                    failure_class=failure_report.proposed_class,
                    target_file=edit.target_file,
                )
        else:
            if critic is None:
                critic = Critic(harness_dir=harness_dir)
            for idx, edit in enumerate(failure_edits):
                verdict = critic.evaluate(
                    edit.unified_diff,
                    edit_index=idx,
                    failure_class=failure_report.proposed_class,
                    tier=critic_tier,
                )
                verdict.target_file = edit.target_file

        results.append(
            EvolutionResult(
                session_id=session_id,
                failure_report=failure_report,
                failure_class=failure_report.proposed_class,
                proposed_edits=failure_edits,
                verdict=verdict,
                evolver_duration_ms=evolver_duration_ms,
                harness_version=harness_version,
                started_at=started_at,
                completed_at=_now_iso(),
            )
        )
        all_edits.extend(failure_edits)

    return BatchEvolutionResult(
        session_id=session_id,
        batch_report=batch_report,
        results=results,
        total_failures=batch_report.total_failures,
        proposed_edits=all_edits,
        harness_version=harness_version,
        started_at=started_at,
        completed_at=_now_iso(),
    )


class DaemonResult(BaseModel):
    """Structured result of one daemon run (issue #1047).

    Returned by :func:`run_evolution_daemon` when the daemon exits (either
    via SIGTERM graceful shutdown or by reaching ``max_iterations``).
    """

    iterations: int = Field(description="Number of poll cycles completed")
    sessions_processed: int = Field(description="Total sessions evolved across all cycles")
    sessions_skipped: int = Field(description="Sessions that were already evolved or had no events")
    shutdown_reason: str = Field(description="Why the daemon stopped")
    started_at: str
    completed_at: str


class _ShutdownState:
    """Mutable flag shared between the signal handler and the daemon loop."""

    def __init__(self) -> None:
        self.requested = False

    def request(self) -> None:
        self.requested = True


def run_evolution_daemon(
    harness_dir: Path,
    trace_db: str,
    *,
    poll_interval_s: float = 60.0,
    no_verify: bool = False,
    verbose: bool = False,
    max_iterations: int | None = None,
    trace_logger: TraceLogger | None = None,
    shutdown_state: _ShutdownState | None = None,
) -> DaemonResult:
    """Run the evolution daemon continuously (issue #1047).

    Polls the trace store for sessions that have not yet been evolved,
    processes each through :func:`run_evolution_batch`, records the
    ``critic_verdict`` event for each failure class, and marks the session
    as evolved. Runs
    indefinitely until:

    * **SIGTERM** is received — the daemon finishes the current session
      and exits gracefully.
    * ``max_iterations`` is reached — useful for testing.

    Parameters
    ----------
    harness_dir:
        Path to the live harness directory.
    trace_db:
        Path to the trace SQLite database or JSONL file.
    poll_interval_s:
        Seconds to sleep between poll cycles when no unevolved sessions
        are found. Default 60.
    no_verify:
        Skip the Critic gate on each evolution step (issue #888).
    verbose:
        Print progress to stdout.
    max_iterations:
        Maximum number of poll cycles before the daemon returns. ``None``
        (default) means run indefinitely (until SIGTERM).
    trace_logger:
        Optional pre-constructed :class:`TraceLogger`. When omitted, one
        is created from ``trace_db``.
    shutdown_state:
        Optional pre-constructed shutdown flag. When omitted, a new one
        is created and a SIGTERM handler is installed. Tests can pass
        their own to avoid modifying process-global signal handlers.

    Returns
    -------
    DaemonResult
        Summary of the daemon run.
    """
    if not trace_db:
        raise ValueError("trace_db must not be empty")

    started_at = _now_iso()
    owns_logger = trace_logger is None
    if owns_logger:
        backend = "jsonl" if trace_db.endswith(".jsonl") else "sqlite"
        trace_logger = TraceLogger(trace_db, backend=backend)

    owns_shutdown = shutdown_state is None
    if owns_shutdown:
        shutdown_state = _ShutdownState()

    previous_handler: signal.Handlers | None = None
    if owns_shutdown:

        def _sigterm_handler(signum: int, frame: object) -> None:
            if verbose:
                sys.stderr.write(
                    "evolution-daemon: SIGTERM received, shutting down after current session...\n"
                )
            shutdown_state.requested = True

        previous_handler = signal.signal(signal.SIGTERM, _sigterm_handler)

    iterations = 0
    sessions_processed = 0
    sessions_skipped = 0

    try:
        while True:
            if shutdown_state.requested:
                break

            iterations += 1

            assert trace_logger is not None
            unevolved = trace_logger.list_unevolved_sessions()

            if not unevolved:
                if verbose:
                    print(
                        f"[daemon] cycle {iterations}: no unevolved sessions, "
                        f"sleeping {poll_interval_s}s..."
                    )
            else:
                if verbose:
                    print(
                        f"[daemon] cycle {iterations}: {len(unevolved)} "
                        f"unevolved session(s) to process"
                    )

            for session_id in unevolved:
                if shutdown_state.requested:
                    break

                assert trace_logger is not None
                events = trace_logger.load_session(session_id)
                if not events:
                    sessions_skipped += 1
                    trace_logger.mark_session_evolved(session_id)
                    continue

                if verbose:
                    print(f"[daemon] evolving session {session_id} ({len(events)} events)")

                try:
                    result = run_evolution_batch(
                        session_id,
                        events,
                        harness_dir,
                        no_verify=no_verify,
                        trace_logger=trace_logger,
                    )
                except Exception:  # noqa: BLE001 — per AGENTS.md, log & continue to next session
                    sys.stderr.write(
                        f"[daemon] error evolving session {session_id}: {sys.exc_info()[1]}\n"
                    )
                    trace_logger.mark_session_evolved(session_id)
                    sessions_skipped += 1
                    continue

                for step_result in result.results:
                    if step_result.verdict is not None:
                        record_verdict(trace_logger, session_id, step_result.verdict)

                trace_logger.mark_session_evolved(session_id)
                sessions_processed += 1

                if verbose:
                    n_approved = sum(
                        1 for r in result.results if r.verdict is not None and r.verdict.verdict
                    )
                    n_rejected = sum(
                        1 for r in result.results if r.verdict is not None and not r.verdict.verdict
                    )
                    n_clean = sum(1 for r in result.results if r.verdict is None)
                    n_classes = result.batch_report.total_failures
                    print(
                        f"[daemon] session {session_id} done "
                        f"({n_classes} failure class(es): "
                        f"{n_approved} approved, {n_rejected} rejected, {n_clean} clean)"
                    )

            if max_iterations is not None and iterations >= max_iterations:
                break

            if not shutdown_state.requested:
                _interruptible_sleep(poll_interval_s, shutdown_state)
    finally:
        if owns_shutdown and previous_handler is not None:
            signal.signal(signal.SIGTERM, previous_handler)
        if owns_logger and trace_logger is not None:
            trace_logger.close()

    if shutdown_state.requested:
        shutdown_reason = "sigterm"
    elif max_iterations is not None:
        shutdown_reason = "max_iterations"
    else:
        shutdown_reason = "unknown"

    return DaemonResult(
        iterations=iterations,
        sessions_processed=sessions_processed,
        sessions_skipped=sessions_skipped,
        shutdown_reason=shutdown_reason,
        started_at=started_at,
        completed_at=_now_iso(),
    )


def _interruptible_sleep(seconds: float, shutdown_state: _ShutdownState) -> None:
    """Sleep for *seconds* but wake early when shutdown is requested.

    Uses a :class:`threading.Event` so the sleep can be interrupted by a
    SIGTERM handler running in the main thread.
    """
    event = threading.Event()

    def _on_signal(signum: int, frame: object) -> None:
        shutdown_state.requested = True
        event.set()

    previous = signal.signal(signal.SIGTERM, _on_signal)
    try:
        event.wait(timeout=seconds)
    finally:
        signal.signal(signal.SIGTERM, previous)
        if event.is_set():
            shutdown_state.requested = True
