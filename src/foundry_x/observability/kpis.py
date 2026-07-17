"""Compute the three PRD success-metric KPIs from trace data.

The PRD (``docs/PRD.md`` §5) defines:

* **Cycle Time** — time from *Agent Failure* to *Harness Edit Proposal*.
* **Regression Rate** — number of previously-solved tasks that break after
  a harness edit.
* **Improvement Rate** — success rate on a standardized benchmark before
  vs. after harness evolution.

This module derives approximations of those metrics from the events already
recorded by :class:`~foundry_x.trace.logger.TraceLogger`:

* ``cycle_time_seconds`` — the operational proxy: mean wall-clock time from
  the first ``task_received`` event to the first ``critic_verdict`` event
  per session (the closest measurable proxy for the business-level "Agent
  Failure" → "Harness Edit Proposal" definition above).
* ``regression_rate`` — fraction of sessions with a ``critic_verdict`` in which
  a task previously seen in ``passed_checks`` later appears in ``failed_checks``
  (the persisted :class:`~foundry_x.observability.regression_report.VerdictRecord`
  shape).
* ``improvement_rate`` — fraction of ``critic_verdict`` events whose persisted
  payload has ``approved: true``.

When the source events are absent the function degrades gracefully,
returning ``None`` (cycle time) or ``0.0`` so the CLI can print ``N/A``.

Issue #120 adds an auxiliary per-session ``injection_blocked`` count derived
from the firewall events persisted by ``InjectionFirewallHook``. The
counts are surfaced only when at least one session has ≥1 block, so a
clean store does not grow the KPI output.

Issue #82: this module previously opened a raw ``sqlite3`` connection on
``logger.path`` and issued bespoke ``SELECT`` statements — see ADR-0003
("No raw SQL strings in business logic"). The store schema is now reached
exclusively through :class:`TraceLogger`'s ``list_sessions`` and
``iter_events`` methods, which own the row format and yield events one at
a time so a future streaming caller does not have to load everything.

Issue #183: an append-only JSONL history log (``--log-to`` /
``--from-history``) gives the regression signal a temporal axis —
operators can see cycle time drifting across harness edits without
manually diffing four JSON snapshots. The per-session
``injection_blocks`` map is intentionally excluded from history
entries; the trend table is a one-row-per-run summary, not a
per-session inventory.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel, ValidationError

from foundry_x.evolution.digester import INJECTION_BLOCKED_KIND
from foundry_x.observability.regression_report import VerdictRecord
from foundry_x.trace.logger import TraceEvent, TraceLogger

TASK_ABORTED_KIND = "task_aborted"
TOKEN_BUDGET_REASON = "token_budget"


CONTEXT_PRUNED_KIND = "context_pruned"


class KpiSummary(BaseModel):
    """Structured summary of the three PRD KPIs.

    Issue #120 adds ``injection_blocks``: a ``session_id -> count`` map
    of ``injection_blocked`` events per session, sourced from the firewall
    hook. Empty by default; populated only when the trace store has at
    least one ``injection_blocked`` event.

    Issue #271 adds ``token_totals``: a ``session_id -> int`` map of the
    cumulative ``total_tokens`` consumed per session, summed from the
    ``usage`` payloads the runner records on each ``model_response`` event
    (issue #191). Empty by default; populated only when at least one
    ``model_response`` event carries a ``usage`` dict, so a trace store
    with no token data (e.g. an endpoint that never reports usage) keeps
    the summary compact. Like ``injection_blocks`` this is an auxiliary
    operator signal, not one of the three PRD success-metric KPIs.

    Issue #604 adds ``evolver_duration_ms``: mean wall-clock milliseconds
    spent inside ``evolver.propose()`` per session, sourced from the
    ``evolver_duration_ms`` field of :class:`~foundry_x.evolution.loop.EvolutionResult`.
    ``None`` when no evolver phase was recorded for any session.

    Issue #585 adds ``hooks_disabled_count`` and ``hooks_disabled_rate``:
    the total count of ``hook_registry_error`` events and the fraction of
    sessions with at least one such event. Emitted when
    ``harness.hooks.get_registry()`` raises, disabling all hooks including
    the security-critical ``InjectionFirewallHook``.

    Issue #466 adds ``token_budget_abort_count``: the number of sessions
    that recorded at least one ``task_aborted(reason="token_budget")``
    event. Surfaced as an auxiliary operator signal alongside
    ``injection_blocks`` and ``token_totals``.

    Issue #551 adds ``token_budget_hit_rate``: the fraction of sessions
    that recorded at least one ``task_aborted(reason="token_budget")``
    event. This is a fourth tracked metric exposed via ``foundry-kpis``
    and the regression report, alongside the three PRD KPIs.

    Issue #580 adds ``streaming_quality``: a ``session_id ->
    StreamingQualityData`` map of per-session streaming quality metrics
    (avg TTFT, chunk count, avg chunk interval) derived from the timing
    fields on each ``model_response`` event. Empty by default; populated
    only when at least one ``model_response`` event carries timing data.

    Issue #626 adds ``context_pruned_count``: a ``session_id -> count`` map
    of ``context_pruned`` events per session, sourced from the pruning hook.
    Empty by default; populated only when at least one ``context_pruned``
    event has been recorded. Like ``injection_blocks`` and ``token_totals``
    this is an auxiliary operator signal.
    """

    cycle_time_seconds: float | None = None
    regression_rate: float = 0.0
    improvement_rate: float = 0.0
    injection_blocks: dict[str, int] = {}
    token_totals: dict[str, int] = {}
    evolver_duration_ms: float | None = None
    hooks_disabled_count: int = 0
    hooks_disabled_rate: float = 0.0
    token_budget_abort_count: int = 0
    token_budget_hit_rate: float = 0.0
    streaming_quality: dict[str, "StreamingQualityData"] = {}
    context_pruned_count: dict[str, int] = {}
    wall_clock_abort_count: int = 0
    failure_class_distribution: dict[str, int] = {}


class StreamingQualityData(BaseModel):
    """Streaming quality metrics for one session (issue #580).

    Aggregated from the timing fields on each ``model_response`` event:
    ``time_to_first_token_ms``, ``chunk_count``, and ``total_stream_ms``.
    """

    avg_ttft_ms: float | None = None
    total_chunks: int = 0
    avg_chunk_interval_ms: float | None = None


class KpiComparison(BaseModel):
    """Baseline-vs-candidate harness-version comparison (issue #100).

    ``deltas`` holds the raw ``candidate - baseline`` difference for each
    numeric KPI; the rendering layer interprets the sign per the PRD's
    "good direction" — improvement-rate up is good, regression-rate and
    cycle-time down are good. ``injection_blocks`` is intentionally
    excluded from the comparison because it is an auxiliary signal, not
    one of the three PRD success-metric KPIs.

    Issue #736 adds ``baseline_session_count`` and ``candidate_session_count``
    so that callers can distinguish "no change" (deltas near 0.0 with real
    sessions) from "no data" (deltas are 0.0 because one version has zero
    sessions in the trace store).
    """

    baseline: KpiSummary
    candidate: KpiSummary
    deltas: dict[str, float | None]
    baseline_session_count: int = 0
    candidate_session_count: int = 0


class KpiHistoryEntry(BaseModel):
    """One line in the append-only KPI history log (issue #183).

    Carries the three PRD-KPI fields from :class:`KpiSummary` plus a
    ``timestamp`` (ISO-8601, stamped at append time) and an optional
    ``harness_version`` (preserved when the operator filtered the
    run with ``--harness-version``). The per-session ``injection_blocks``
    map is intentionally absent — the history is a one-row-per-run
    summary, and per-session inventory is the trace store's job.

    Issue #585 adds ``hooks_disabled_count`` and ``hooks_disabled_rate``:
    these scalar fields are included in the history log (unlike the per-
    session maps) because they represent aggregate KPI signal, not per-
    session inventory.

    The serialized JSON line round-trips through :class:`KpiSummary`
    because pydantic's default ``extra='ignore'`` policy silently
    drops ``timestamp`` and ``harness_version`` on parse, leaving
    only the three numeric KPIs. That round-trip — minus the per-
    session map — is the on-disk contract the trend table relies on.
    """

    timestamp: str
    harness_version: str | None = None
    cycle_time_seconds: float | None = None
    regression_rate: float = 0.0
    improvement_rate: float = 0.0
    injection_blocks: dict[str, int] = {}
    hooks_disabled_count: int = 0
    hooks_disabled_rate: float = 0.0
    wall_clock_abort_count: int = 0
    failure_class_distribution: dict[str, int] = {}


def _failure_class_distribution(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Aggregate ``failure_class`` counts from persisted Critic verdicts (issue #705).

    Returns a ``failure_class -> count`` map across every ``critic_verdict``
    event matching *harness_version*. Verdicts without a ``failure_class``
    (e.g. from older stores or clean sessions that short-circuit before the
    Critic runs) are ignored so the map only includes sessions that ran the
    full pipeline.
    """
    distribution: dict[str, int] = {}
    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        record = VerdictRecord(**event.payload)
        if record.failure_class is not None:
            distribution[record.failure_class] = distribution.get(record.failure_class, 0) + 1
    return distribution


def compute_kpis(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> KpiSummary:
    """Compute KPIs from the trace store backing *logger*.

    Parameters
    ----------
    logger:
        A :class:`~foundry_x.trace.logger.TraceLogger`.
    harness_version:
        When provided, only sessions created with this harness version are
        considered.

    Issue #273 — the per-session helpers below each call
    :meth:`TraceLogger.query_events` exactly once per event kind. The
    previous shape issued ``list_sessions()`` and then ``iter_events(sid)``
    once per session per kind (S*K connect sites); the new shape is K
    streaming cursors total, with the ``harness_version`` filter pushed
    down to the store so a multi-session fixture does not need to be
    materialized in Python.
    """
    cycle_time = _cycle_time(logger, harness_version=harness_version)
    regression_rate, improvement_rate = _verdict_rates(logger, harness_version=harness_version)
    injection_blocks = _injection_blocks(logger, harness_version=harness_version)
    token_totals = _token_totals(logger, harness_version=harness_version)
    hooks_disabled_count, hooks_disabled_rate = _hook_registry_errors(
        logger, harness_version=harness_version
    )
    token_budget_abort_count = _token_budget_aborts(logger, harness_version=harness_version)
    token_budget_hit_rate = _token_budget_hit_rate(logger, harness_version=harness_version)
    streaming_quality = _streaming_quality(logger, harness_version=harness_version)
    context_pruned_count = _context_pruned(logger, harness_version=harness_version)
    wall_clock_abort_count = _wall_clock_abort_count(logger, harness_version=harness_version)
    failure_class_distribution = _failure_class_distribution(
        logger, harness_version=harness_version
    )

    return KpiSummary(
        cycle_time_seconds=cycle_time,
        regression_rate=regression_rate,
        improvement_rate=improvement_rate,
        injection_blocks=injection_blocks,
        token_totals=token_totals,
        hooks_disabled_count=hooks_disabled_count,
        hooks_disabled_rate=hooks_disabled_rate,
        token_budget_abort_count=token_budget_abort_count,
        token_budget_hit_rate=token_budget_hit_rate,
        streaming_quality=streaming_quality,
        context_pruned_count=context_pruned_count,
        wall_clock_abort_count=wall_clock_abort_count,
        failure_class_distribution=failure_class_distribution,
    )


def compare_kpis(
    logger: TraceLogger,
    baseline_version: str,
    candidate_version: str,
) -> KpiComparison:
    """Compute a baseline-vs-candidate comparison (issue #100).

    Each version is reduced to its own :class:`KpiSummary` via
    :func:`compute_kpis`, then the candidate-minus-baseline deltas are
    derived for the three PRD KPIs. The sign convention (which direction
    is "good") is applied at render time, not here, so the structured
    ``deltas`` stay sign-agnostic for JSON consumers.

    Issue #736: session counts are included so callers can distinguish
    "no change" (deltas near 0.0 with real sessions) from "no data"
    (deltas are 0.0 because one version has zero sessions).
    """
    baseline = compute_kpis(logger, harness_version=baseline_version)
    candidate = compute_kpis(logger, harness_version=candidate_version)
    baseline_session_count = len(logger.list_sessions(harness_version=baseline_version))
    candidate_session_count = len(logger.list_sessions(harness_version=candidate_version))
    return KpiComparison(
        baseline=baseline,
        candidate=candidate,
        deltas=_compute_deltas(baseline, candidate),
        baseline_session_count=baseline_session_count,
        candidate_session_count=candidate_session_count,
    )


def _compute_deltas(
    baseline: KpiSummary,
    candidate: KpiSummary,
) -> dict[str, float | None]:
    def _delta(b: float | None, c: float | None) -> float | None:
        if b is None or c is None:
            return None
        return c - b

    return {
        "cycle_time_seconds": _delta(baseline.cycle_time_seconds, candidate.cycle_time_seconds),
        "regression_rate": _delta(baseline.regression_rate, candidate.regression_rate),
        "improvement_rate": _delta(baseline.improvement_rate, candidate.improvement_rate),
        "token_budget_hit_rate": _delta(
            baseline.token_budget_hit_rate, candidate.token_budget_hit_rate
        ),
    }


def _cycle_time(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> float | None:
    """Mean wall-clock time from ``task_received`` to ``critic_verdict``.

    Issue #273 — previously looped every session id and called
    ``iter_events`` twice per session to find the first event of each
    kind. Now two :meth:`TraceLogger.query_events` cursors stream every
    qualifying event in timestamp order; ``setdefault`` keeps the first
    (earliest) event per session, which is exactly the prior
    first-event-of-kind semantics.
    """
    start_events: dict[str, TraceEvent] = {}
    for event in logger.query_events(kind="task_received", harness_version=harness_version):
        start_events.setdefault(event.session_id, event)
    end_events: dict[str, TraceEvent] = {}
    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        end_events.setdefault(event.session_id, event)

    deltas: list[float] = []
    for sid, start_event in start_events.items():
        end_event = end_events.get(sid)
        if end_event is None:
            continue
        try:
            t0 = datetime.fromisoformat(start_event.timestamp)
            t1 = datetime.fromisoformat(end_event.timestamp)
        except ValueError:
            continue
        delta = (t1 - t0).total_seconds()
        if delta > 0:
            deltas.append(delta)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _verdict_rates(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[float, float]:
    """Derive regression and improvement rates from persisted Critic verdicts.

    Verdicts are persisted as the :class:`VerdictRecord` shape
    (``approved`` / ``passed_checks`` / ``failed_checks`` / ``notes``), not the
    synthetic ``{"verdict", "regression"}`` payload the earlier implementation
    assumed (issue #98).

    Issue #273 — a single :meth:`TraceLogger.query_events` cursor walks
    every ``critic_verdict`` row across all matching sessions in
    timestamp order, so the ``prior_passed`` tracker sees verdicts in
    the same order the previous per-session nested loop produced.

    * *improvement_rate* = approved verdicts / total verdicts.
    * *regression_rate* = sessions with >=1 regressed task / sessions with a
      verdict, where a task regresses when it appears in ``failed_checks`` after
      having appeared in ``passed_checks`` in an earlier verdict.
    """

    total_verdicts = 0
    approved = 0
    prior_passed: dict[str, str] = {}
    sessions_with_verdicts: set[str] = set()
    regression_sessions: set[str] = set()

    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        total_verdicts += 1
        sessions_with_verdicts.add(event.session_id)
        record = VerdictRecord(**event.payload)
        if record.verdict:
            approved += 1
        for task in record.failed_checks:
            if task in prior_passed:
                regression_sessions.add(event.session_id)
        for task in record.passed_checks:
            prior_passed[task] = event.session_id

    improvement_rate = approved / total_verdicts if total_verdicts else 0.0
    regression_rate = (
        len(regression_sessions) / len(sessions_with_verdicts) if sessions_with_verdicts else 0.0
    )
    return regression_rate, improvement_rate


def _injection_blocks(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Per-session count of ``injection_blocked`` events (issue #120).

    Returns a ``session_id -> count`` map including only sessions with at
    least one block. Sessions without blocks are omitted so the rendering
    path can decide whether to add an extra section based on the map being
    non-empty (per the issue's "show … when at least one is present").

    Issue #273 — one :meth:`TraceLogger.query_events` cursor replaces
    the previous per-session ``iter_events`` loop; the kind filter is
    pushed down so only ``injection_blocked`` rows cross the boundary.
    """
    blocks: dict[str, int] = {}
    for event in logger.query_events(
        kind=INJECTION_BLOCKED_KIND,
        harness_version=harness_version,
    ):
        blocks[event.session_id] = blocks.get(event.session_id, 0) + 1
    return blocks


def _token_totals(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Per-session cumulative token totals (issue #271).

    Sums ``usage.total_tokens`` across every ``model_response`` event the
    runner records (issue #191). The runner itself keeps a running
    ``tokens_used`` counter (issue #197); summing the per-response
    ``total_tokens`` reproduces that cumulative figure without depending on
    the ``tokens_used`` key being present, so events written before that
    field landed still contribute.

    A ``model_response`` whose ``usage`` is missing or ``None`` (an
    OpenAI-compatible endpoint that omits accounting) contributes zero and
    does **not** seed the session into the map — only sessions with at
    least one event carrying a ``usage`` dict appear, mirroring the
    ``_injection_blocks`` "show only when present" contract.

    Like the other per-session helpers this uses one
    :meth:`TraceLogger.query_events` cursor (issue #273) with the kind and
    ``harness_version`` filters pushed down, so a multi-session store is a
    single ordered scan rather than S round-trips.
    """
    totals: dict[str, int] = {}
    for event in logger.query_events(
        kind="model_response",
        harness_version=harness_version,
    ):
        usage = event.payload.get("token_usage")
        if not isinstance(usage, dict):
            continue
        step_total = usage.get("total_tokens", 0)
        # ``bool`` is a subclass of ``int``; guard against truthy flags.
        if isinstance(step_total, bool) or not isinstance(step_total, int):
            continue
        totals[event.session_id] = totals.get(event.session_id, 0) + step_total
    return totals


def _hook_registry_errors(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[int, float]:
    """Total count and session-fraction of ``hook_registry_error`` events (issue #585).

    Returns ``(total_count, disabled_rate)`` where ``disabled_rate`` is the
    fraction of sessions with a ``task_received`` event that also had at
    least one ``hook_registry_error``. A registry error means every hook —
    including the security-critical ``InjectionFirewallHook`` — is silently
    disabled for the entire session, so any presence is noteworthy.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    sessions_with_errors: set[str] = set()
    total_count = 0
    for event in logger.query_events(
        kind="hook_registry_error",
        harness_version=harness_version,
    ):
        total_count += 1
        sessions_with_errors.add(event.session_id)

    if not sessions_with_errors:
        return 0, 0.0

    # Use sessions with task_received as the denominator (active work sessions).
    sessions_with_task: set[str] = set()
    for event in logger.query_events(kind="task_received", harness_version=harness_version):
        sessions_with_task.add(event.session_id)

    rate = len(sessions_with_errors) / len(sessions_with_task) if sessions_with_task else 0.0
    return total_count, rate


def _token_budget_aborts(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count sessions that hit ``task_aborted(reason="token_budget")`` (issue #466).

    Unlike ``_injection_blocks`` which returns a per-session map, this
    function returns a single integer: the number of sessions that
    recorded at least one ``task_aborted`` event with
    ``reason="token_budget"``. Sessions are counted once regardless of
    how many times the abort fires within them.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with
    the kind and ``harness_version`` filters pushed down.
    """
    sessions_with_abort: set[str] = set()
    for event in logger.query_events(
        kind=TASK_ABORTED_KIND,
        harness_version=harness_version,
    ):
        if event.payload.get("reason") == TOKEN_BUDGET_REASON:
            sessions_with_abort.add(event.session_id)
    return len(sessions_with_abort)


def _token_budget_hit_rate(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> float:
    """Fraction of sessions with at least one ``task_aborted(reason="token_budget")`` event.

    Issue #551 — the token budget hit rate is a fourth tracked metric
    exposed via ``foundry-kpis`` alongside the three PRD KPIs. It signals
    whether the harness is driving tasks that repeatedly hit the token
    budget, which would indicate the context-pruning hook is not aggressive
    enough, or that the model-context window is being misspent.

    A session contributes to the numerator if it has at least one
    ``task_aborted`` event whose ``payload["reason"] == "token_budget"``.
    The denominator is the total number of sessions that have a
    ``task_received`` event (matching the harness version filter), which
    is the natural population boundary for the KPI.
    """
    sessions_with_abort: set[str] = set()
    all_sessions: set[str] = set()

    for event in logger.query_events(kind="task_received", harness_version=harness_version):
        all_sessions.add(event.session_id)

    for event in logger.query_events(kind="task_aborted", harness_version=harness_version):
        if event.payload.get("reason") == "token_budget":
            sessions_with_abort.add(event.session_id)

    if not all_sessions:
        return 0.0
    return len(sessions_with_abort) / len(all_sessions)


def _streaming_quality(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, StreamingQualityData]:
    """Per-session streaming quality metrics (issue #580).

    Aggregates ``time_to_first_token_ms``, ``chunk_count``, and ``total_stream_ms``
    from every ``model_response`` event the runner records. Computes per-session
    average TTFT and average chunk interval (total_stream_ms / chunk_count).

    Sessions with no ``model_response`` events, or whose events lack timing
    data, are omitted from the returned map (mirrors the "show only when present"
    contract of ``_injection_blocks`` and ``_token_totals``).
    """
    session_ttfts: dict[str, list[int]] = {}
    session_chunks: dict[str, list[int]] = {}
    session_stream_ms: dict[str, list[int]] = {}

    for event in logger.query_events(
        kind="model_response",
        harness_version=harness_version,
    ):
        sid = event.session_id
        ttft = event.payload.get("time_to_first_token_ms")
        chunk_count = event.payload.get("chunk_count")
        total_stream_ms = event.payload.get("total_stream_ms")

        if isinstance(ttft, int):
            session_ttfts.setdefault(sid, []).append(ttft)
        if isinstance(chunk_count, int):
            session_chunks.setdefault(sid, []).append(chunk_count)
        if isinstance(total_stream_ms, int):
            session_stream_ms.setdefault(sid, []).append(total_stream_ms)

    result: dict[str, StreamingQualityData] = {}
    for sid in session_ttfts:
        ttfts = session_ttfts.get(sid, [])
        chunks = session_chunks.get(sid, [])
        stream_ms = session_stream_ms.get(sid, [])

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        total_chunks = sum(chunks) if chunks else 0
        total_ms = sum(stream_ms) if stream_ms else 0
        avg_interval = total_ms / total_chunks if total_chunks > 0 else None

        result[sid] = StreamingQualityData(
            avg_ttft_ms=avg_ttft,
            total_chunks=total_chunks,
            avg_chunk_interval_ms=avg_interval,
        )
    return result


def _context_pruned(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Per-session count of ``context_pruned`` events (issue #626).

    Returns a ``session_id -> count`` map including only sessions with at
    least one prune. Sessions without pruning are omitted so the rendering
    path can decide whether to add an extra section based on the map being
    non-empty (mirroring the ``_injection_blocks`` "show only when present"
    contract).

    Uses one :meth:`TraceLogger.query_events` cursor with the kind and
    ``harness_version`` filters pushed down.
    """
    counts: dict[str, int] = {}
    for event in logger.query_events(
        kind=CONTEXT_PRUNED_KIND,
        harness_version=harness_version,
    ):
        counts[event.session_id] = counts.get(event.session_id, 0) + 1
    return counts


def _wall_clock_abort_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count sessions aborted by the FOUNDRY_TASK_TIMEOUT wall-clock cap (issue #711).

    Counts every ``task_aborted`` event whose ``reason`` is ``"wall_clock"``,
    fired by :func:`foundry_x.execution.runner.run_with_limits` when
    ``asyncio.wait_for`` raises :class:`asyncio.TimeoutError`. Each session
    contributes at most one such event (the runner records it once per abort).
    """
    count = 0
    for event in logger.query_events(
        kind="task_aborted",
        harness_version=harness_version,
    ):
        if event.payload.get("reason") == "wall_clock":
            count += 1
    return count


def _format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _format_delta(
    baseline: float | None,
    candidate: float | None,
    higher_is_better: bool,
) -> str:
    """Render a candidate-minus-baseline delta with a PRD sign convention.

    Per issue #100 an *improvement-rate* increase is marked ``positive``
    (good) while a *regression-rate* or *cycle-time* increase is marked
    ``negative`` (bad). ``higher_is_better`` selects which polarity the
    PRD treats as favorable for the given KPI. A near-zero change is
    ``neutral``; an unmeasurable side (``None``) yields ``N/A``.
    """
    if baseline is None or candidate is None:
        return "N/A"
    delta = candidate - baseline
    if abs(delta) < 1e-9:
        mark = "neutral"
    elif (delta > 0) is higher_is_better:
        mark = "positive"
    else:
        mark = "negative"
    return f"{delta:+.2f} ({mark})"


def _render_markdown(summary: KpiSummary) -> str:
    lines = [
        "| KPI | Value |",
        "| --- | --- |",
        f"| Cycle Time (seconds) | {_format_value(summary.cycle_time_seconds)} |",
        f"| Regression Rate | {_format_value(summary.regression_rate)} |",
        f"| Improvement Rate | {_format_value(summary.improvement_rate)} |",
        f"| Hooks Disabled Count | {summary.hooks_disabled_count} |",
        f"| Hooks Disabled Rate | {_format_value(summary.hooks_disabled_rate)} |",
        f"| Token Budget Hit Rate | {_format_value(summary.token_budget_hit_rate)} |",
    ]
    # Issue #120: surface per-session ``injection_blocked`` counts only when
    # at least one session has ≥1 block; a clean trace store stays compact.
    if summary.injection_blocks:
        total = sum(summary.injection_blocks.values())
        lines.append("")
        lines.append(
            f"Injection Blocked: {total} block(s) across "
            f"{len(summary.injection_blocks)} session(s)."
        )
        lines.append("")
        lines.append("| Session | injection_blocked |")
        lines.append("| --- | --- |")
        for sid, count in sorted(summary.injection_blocks.items()):
            lines.append(f"| {sid} | {count} |")
    # Issue #271: surface per-session token consumption only when at least
    # one ``model_response`` carried a ``usage`` payload; a trace store with
    # no token accounting (budget never plumbed, or an endpoint that omits
    # usage) keeps the summary compact.
    if summary.token_totals:
        grand_total = sum(summary.token_totals.values())
        lines.append("")
        lines.append(
            f"Token Usage: {grand_total} token(s) across {len(summary.token_totals)} session(s)."
        )
        lines.append("")
        lines.append("| Session | Tokens |")
        lines.append("| --- | --- |")
        for sid, count in sorted(summary.token_totals.items()):
            lines.append(f"| {sid} | {count} |")
    if summary.token_budget_abort_count > 0:
        lines.append("")
        lines.append(
            f"Token Budget Aborts: {summary.token_budget_abort_count} session(s) "
            "hit the token budget limit."
        )
    # Issue #580: surface per-session streaming quality (avg TTFT) only when
    # at least one ``model_response`` carried timing data.
    if summary.streaming_quality:
        lines.append("")
        lines.append("Streaming Quality (avg TTFT):")
        lines.append("")
        lines.append("| Session | avg TTFT (ms) | total chunks | avg chunk interval (ms) |")
        lines.append("| --- | --- | --- | --- |")
        for sid, sq in sorted(summary.streaming_quality.items()):
            avg_ttft = _format_value(sq.avg_ttft_ms)
            avg_interval = _format_value(sq.avg_chunk_interval_ms)
            lines.append(f"| {sid} | {avg_ttft} | {sq.total_chunks} | {avg_interval} |")
    # Issue #626: surface per-session ``context_pruned`` counts only when
    # at least one session has ≥1 prune; a clean trace store stays compact.
    if summary.context_pruned_count:
        total = sum(summary.context_pruned_count.values())
        lines.append("")
        lines.append(
            f"Context Pruned: {total} prune(s) across "
            f"{len(summary.context_pruned_count)} session(s)."
        )
        lines.append("")
        lines.append("| Session | context_pruned |")
        lines.append("| --- | --- |")
        for sid, count in sorted(summary.context_pruned_count.items()):
            lines.append(f"| {sid} | {count} |")
    # Issue #711: surface wall-clock abort count as an auxiliary operator
    # signal. Zero means the timeout cap is not firing (expected for healthy
    # runs); non-zero means a session was aborted by FOUNDRY_TASK_TIMEOUT.
    if summary.wall_clock_abort_count > 0:
        lines.append("")
        lines.append(
            f"Wall-Clock Aborts: {summary.wall_clock_abort_count} session(s) "
            "were aborted by FOUNDRY_TASK_TIMEOUT."
        )
    if summary.failure_class_distribution:
        total = sum(summary.failure_class_distribution.values())
        lines.append("")
        lines.append(
            f"Failure Class Distribution: {total} verdict(s) across "
            f"{len(summary.failure_class_distribution)} class(es)."
        )
        lines.append("")
        lines.append("| Failure Class | Count |")
        lines.append("| --- | --- |")
        for cls, count in sorted(summary.failure_class_distribution.items()):
            lines.append(f"| {cls} | {count} |")
    return "\n".join(lines)


def _resolve_format(args_format: str | None, out: str | None) -> str:
    """Return ``"markdown"`` or ``"json"``.

    The explicit ``--format`` flag always wins. When unset, the format is
    inferred from the ``--out`` file extension (``.json`` → JSON);
    otherwise Markdown is returned. Issue #101 keeps the decision local to
    the CLI layer so the pydantic model remains the single source of truth.
    """
    if args_format is not None:
        return args_format
    if out is not None and Path(out).suffix.lower() == ".json":
        return "json"
    return "markdown"


def _render_json(summary: KpiSummary) -> str:
    """Serialize a KPI summary as a stable JSON snapshot (issue #101)."""
    return summary.model_dump_json(indent=2)


def _render_comparison_markdown(baseline: KpiSummary, candidate: KpiSummary) -> str:
    """Render baseline / candidate / delta columns for the three PRD KPIs.

    Issue #100 requires the comparison to surface a delta column whose
    sign convention follows the PRD: improvement-rate up is good,
    regression-rate and cycle-time up are bad.
    """
    lines = [
        "| KPI | Baseline | Candidate | Delta |",
        "| --- | --- | --- | --- |",
        "| Cycle Time (seconds) | "
        f"{_format_value(baseline.cycle_time_seconds)} | "
        f"{_format_value(candidate.cycle_time_seconds)} | "
        f"{_format_delta(baseline.cycle_time_seconds, candidate.cycle_time_seconds, higher_is_better=False)} |",
        "| Regression Rate | "
        f"{_format_value(baseline.regression_rate)} | "
        f"{_format_value(candidate.regression_rate)} | "
        f"{_format_delta(baseline.regression_rate, candidate.regression_rate, higher_is_better=False)} |",
        "| Improvement Rate | "
        f"{_format_value(baseline.improvement_rate)} | "
        f"{_format_value(candidate.improvement_rate)} | "
        f"{_format_delta(baseline.improvement_rate, candidate.improvement_rate, higher_is_better=True)} |",
        "| Token Budget Hit Rate | "
        f"{_format_value(baseline.token_budget_hit_rate)} | "
        f"{_format_value(candidate.token_budget_hit_rate)} | "
        f"{_format_delta(baseline.token_budget_hit_rate, candidate.token_budget_hit_rate, higher_is_better=False)} |",
    ]
    return "\n".join(lines)


def _render_comparison_json(comparison: KpiComparison) -> str:
    """Serialize a baseline-vs-candidate comparison as JSON (issue #100)."""
    return comparison.model_dump_json(indent=2)


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with offset suffix.

    Issue #183 uses this to stamp each appended history row. The
    timezone-aware form keeps the line unambiguous when CI runs
    across multiple regions; ``datetime.fromisoformat`` (Python 3.11+)
    accepts the ``+00:00`` suffix without modification.
    """
    return datetime.now(timezone.utc).isoformat()


def append_kpi_history(
    path: Path,
    summary: KpiSummary,
    harness_version: str | None = None,
) -> None:
    """Append one KPI snapshot to the append-only JSONL history log (issue #183).

    Each run produces exactly one line. The three PRD-KPI fields are
    emitted via :meth:`KpiSummary.model_dump` with ``injection_blocks``,
    ``token_totals``, ``streaming_quality``, and ``wall_clock_abort_count``
    excluded (the "minus per-session maps" half of the round-trip contract).
    ``hooks_disabled_count``, ``hooks_disabled_rate``,
    ``token_budget_abort_count``, and ``token_budget_hit_rate`` are scalar
    fields and are included. Then ``timestamp`` and the optional
    ``harness_version`` are added. Parent directories are created on
    demand so the operator does not have to ``mkdir`` before the first run.
    ``failure_class_distribution`` is included so the trend table can show
    per-class deltas (issue #705).

    The file is opened in append mode and a single ``\\n``-terminated
    line is written per call, so concurrent appends from independent
    ``foundry-kpis`` invocations interleave cleanly at line
    boundaries rather than corrupting the JSON payload of the
    previous line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(
        mode="json",
        exclude={"injection_blocks", "token_totals", "streaming_quality", "wall_clock_abort_count"},
    )
    payload["timestamp"] = _now_iso()
    if harness_version is not None:
        payload["harness_version"] = harness_version
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def read_kpi_history(path: Path) -> list[KpiHistoryEntry]:
    """Read every line of the JSONL history log (issue #183).

    Returns entries in file order — which, for an append-only log,
    is chronological order. Blank lines are tolerated; lines that
    fail pydantic validation are skipped so a single malformed entry
    (e.g. written by a future schema-bumped version of the CLI)
    does not blank the trend table. A missing file yields an empty
    list so the caller can render the placeholder table without a
    precondition check.
    """
    if not path.exists():
        return []
    entries: list[KpiHistoryEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entries.append(KpiHistoryEntry.model_validate_json(stripped))
            except ValidationError:
                continue
    return entries


def _sparkline(values: list[float | None]) -> str:
    """Render a minimal ASCII sparkline for a sequence of values.

    Uses Unicode block characters to approximate a bar-chart feel:
    ``▁▂▃▄▅▆▇█`` (U+2581–U+2588), where each character represents one
    data point scaled linearly across the min/max range.
    ``None`` values render as ``·`` and are excluded from the scale.
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return "N/A"
    lo, hi = min(valid), max(valid)
    span = hi - lo
    blocks = "▁▂▃▄▅▆▇█"

    def _char(v: float | None) -> str:
        if v is None:
            return "·"
        if span == 0:
            idx = len(blocks) - 1
        else:
            idx = min(int((v - lo) / span * (len(blocks) - 1)), len(blocks) - 1)
        return blocks[idx]

    return "".join(_char(v) for v in values)


def render_history_markdown(
    entries: Sequence[KpiHistoryEntry],
    *,
    trend: bool = False,
) -> str:
    """Render a Markdown trend table from KPI history entries (issue #183).

    The table preserves file order, which is the same as append order
    for a JSONL log. Each row carries the timestamp plus the three
    PRD KPIs formatted with two decimals; ``None`` cycle times render
    as ``N/A`` (same convention as :func:`_render_markdown`).

    When *trend* is ``True`` (issue #622), three ASCII sparkline
    columns are appended to the table — one per numeric KPI — giving
    operators a quick visual read of direction without opening a chart.

    An empty history renders a single placeholder line so CI summary
    cells that template-embed the table are never completely blank.

    Plotting (matplotlib, ASCII sparklines) is explicitly out of
    scope per the issue; a pure table is the contract.

    Issue #705: a Failure Class Distribution section is appended when
    at least one entry carries a non-empty ``failure_class_distribution``.
    """
    if not entries:
        return "_No KPI history entries yet._"

    cycle_times = [e.cycle_time_seconds for e in entries]
    regression_rates = [e.regression_rate for e in entries]
    improvement_rates = [e.improvement_rate for e in entries]

    header = "| Timestamp | Cycle Time (s) | Regression Rate | Improvement Rate |"
    if trend:
        header += " Cycle Time | Reg. Rate | Impr. Rate |"
    lines = [header, "| --- | --- | --- | --- |" + (" --- | --- | --- |" if trend else "")]

    sparkline_cycle = _sparkline(cycle_times) if trend else None
    sparkline_reg = _sparkline(regression_rates) if trend else None
    sparkline_imp = _sparkline(improvement_rates) if trend else None

    for idx, entry in enumerate(entries):
        row = (
            f"| {entry.timestamp} | "
            f"{_format_value(entry.cycle_time_seconds)} | "
            f"{_format_value(entry.regression_rate)} | "
            f"{_format_value(entry.improvement_rate)} |"
        )
        if trend:
            sc = sparkline_cycle[idx] if sparkline_cycle else " "
            sr = sparkline_reg[idx] if sparkline_reg else " "
            si = sparkline_imp[idx] if sparkline_imp else " "
            row += f" {sc} | {sr} | {si} |"
        lines.append(row)
    if any(entry.failure_class_distribution for entry in entries):
        lines.append("")
        lines.append("### Failure Class Distribution")
        lines.append("")
        lines.append(
            "| Failure Class | " + " | ".join(f"{e.timestamp[:10]}" for e in entries) + " |"
        )
        lines.append("| --- | " + " | ".join("---" for _ in entries) + " |")
        all_classes = sorted({cls for entry in entries for cls in entry.failure_class_distribution})
        for cls in all_classes:
            row = [f"| {cls} |"]
            for entry in entries:
                count = entry.failure_class_distribution.get(cls, 0)
                row.append(f" {count} |")
            lines.append("".join(row))
    return "\n".join(lines)


def export_prometheus(
    entries: Sequence[KpiHistoryEntry],
    *,
    metric_name_prefix: str = "foundryx",
) -> str:
    """Render Prometheus-format metrics from KPI history entries (issue #565).

    Emits one ``foundryx_kpi_entry`` sample per history entry per KPI
    with a ``kpi`` label identifying the metric. The ``harness_version``
    label is set to the entry's value or ``"unknown"`` if absent.
    This makes it straightforward to scrape and ingest into Grafana
    without a custom exporter.

    The metric is gauge-typed so the most recent value is always the
    current KPI state; the scrape timestamp becomes the ``timestamp``
    field in Prometheus (seconds since epoch).
    """
    if not entries:
        return f"# No KPI history entries — {metric_name_prefix}_kpi_entry is empty.\n"

    lines: list[str] = [
        f"# HELP {metric_name_prefix}_kpi_entry FoundryX KPI from history (issue #565)",
        f"# TYPE {metric_name_prefix}_kpi_entry gauge",
    ]
    for entry in entries:
        ts = entry.timestamp
        harness = entry.harness_version or "unknown"
        labels = f'harness_version="{harness}",kpi="cycle_time_seconds"'
        value = f"{entry.cycle_time_seconds:.6f}" if entry.cycle_time_seconds is not None else "NaN"
        lines.append(f"{metric_name_prefix}_kpi_entry{{{labels}}} {value} {ts}")

        labels = f'harness_version="{harness}",kpi="regression_rate"'
        lines.append(f"{metric_name_prefix}_kpi_entry{{{labels}}} {entry.regression_rate:.6f} {ts}")

        labels = f'harness_version="{harness}",kpi="improvement_rate"'
        lines.append(
            f"{metric_name_prefix}_kpi_entry{{{labels}}} {entry.improvement_rate:.6f} {ts}"
        )

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foundry-kpis",
        description="Compute and display the three PRD success-metric KPIs.",
    )
    parser.add_argument(
        "--db",
        default="./logs/traces.db",
        help="Path to the trace SQLite database (default: ./logs/traces.db).",
    )
    parser.add_argument(
        "--harness-version",
        default=None,
        help="Only consider sessions with this harness version.",
    )
    parser.add_argument(
        "--baseline-harness-version",
        default=None,
        help=(
            "Baseline harness version for a baseline-vs-candidate comparison"
            " (issue #100). Must be paired with --candidate-harness-version."
        ),
    )
    parser.add_argument(
        "--candidate-harness-version",
        default=None,
        help=(
            "Candidate harness version for a baseline-vs-candidate comparison"
            " (issue #100). Must be paired with --baseline-harness-version."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default=None,
        help=(
            "Output format. Default: 'markdown'. When --out ends in '.json',"
            " 'json' is selected automatically."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write output to this path instead of stdout.",
    )
    parser.add_argument(
        "--log-to",
        default=None,
        help=(
            "Append the single-summary KPI snapshot to this JSONL history"
            " log (issue #183). The per-session injection_blocks map is"
            " excluded; only the three PRD KPIs plus a timestamp and"
            " optional harness version are persisted. Comparison runs"
            " do not log — the history is per single-summary run."
        ),
    )
    parser.add_argument(
        "--from-history",
        default=None,
        help=(
            "Render a Markdown trend table from this JSONL history log"
            " (issue #183) and exit without reading the trace store."
            " The file is read in append order; missing or empty files"
            " render a placeholder table."
        ),
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help=(
            "Exit non-zero when the computed regression_rate exceeds this"
            " threshold (issue #565). Applies only to live KPI computation"
            " (not --from-history). Example: --alert-threshold 0.1"
            " causes a non-zero exit when regression_rate > 0.1."
        ),
    )
    parser.add_argument(
        "--cycle-time-alert-threshold",
        type=float,
        default=None,
        dest="cycle_time_alert_threshold",
        help=(
            "Exit non-zero when cycle_time_seconds exceeds this value (issue #621)."
            " The exit message names the triggering KPI and value."
        ),
    )
    parser.add_argument(
        "--export-prometheus",
        action="store_true",
        default=False,
        help=(
            "Emit Prometheus-format metrics to stdout (issue #565)."
            " When used with --from-history, reads the JSONL history log"
            " and exports one sample per entry per KPI. The output is"
            " compatible with Prometheus scraping and Grafana ingestion."
        ),
    )
    parser.add_argument(
        "--trend",
        action="store_true",
        default=False,
        help=(
            "Append ASCII sparkline columns to the Markdown trend table"
            " (issue #565). Requires --from-history. Each KPI gets a"
            " Unicode-block sparkline showing the full history at a glance."
        ),
    )
    args = parser.parse_args(argv)

    baseline_version = args.baseline_harness_version
    candidate_version = args.candidate_harness_version
    if (baseline_version is None) != (candidate_version is None):
        parser.error(
            "--baseline-harness-version and --candidate-harness-version must be supplied together"
        )

    if args.trend and args.from_history is None:
        parser.error(
            "--trend requires --from-history: sparklines are rendered from the KPI history log"
        )

    if args.from_history is not None:
        # Issue #183: trend rendering is a pure read of the JSONL log;
        # it does not require a trace store, so we short-circuit before
        # opening the SQLite database. ``--out`` still works as a sink.
        entries = read_kpi_history(Path(args.from_history))
        if args.export_prometheus:
            output = export_prometheus(entries)
        else:
            output = render_history_markdown(entries, trend=args.trend)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0

    fmt = _resolve_format(args.format, args.out)
    logger = TraceLogger(args.db)

    if baseline_version is not None and candidate_version is not None:
        comparison = compare_kpis(logger, baseline_version, candidate_version)
        if fmt == "json":
            output = _render_comparison_json(comparison)
        else:
            output = _render_comparison_markdown(comparison.baseline, comparison.candidate)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0
    else:
        summary = compute_kpis(logger, harness_version=args.harness_version)
        if args.log_to is not None:
            append_kpi_history(
                Path(args.log_to),
                summary,
                harness_version=args.harness_version,
            )
        if (
            args.cycle_time_alert_threshold is not None
            and summary.cycle_time_seconds is not None
            and summary.cycle_time_seconds > args.cycle_time_alert_threshold
        ):
            print(
                f"ALERT: cycle_time_seconds ({summary.cycle_time_seconds:.2f}) exceeds "
                f"threshold ({args.cycle_time_alert_threshold:.2f})",
                file=sys.stderr,
            )
            return 1
        output = _render_json(summary) if fmt == "json" else _render_markdown(summary)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)

    # Issue #565: alert threshold — regression rate above threshold triggers CI gate.
    if args.alert_threshold is not None and summary.regression_rate > args.alert_threshold:
        sys.stderr.write(
            f"[ALERT] regression_rate {summary.regression_rate:.4f}"
            f" exceeds threshold {args.alert_threshold:.4f}\n"
        )
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
