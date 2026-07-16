"""Compute the three PRD success-metric KPIs from trace data.

The PRD (``docs/PRD.md`` §5) defines:

* **Cycle Time** — time from *Agent Failure* to *Harness Edit Proposal*.
* **Regression Rate** — number of previously-solved tasks that break after
  a harness edit.
* **Improvement Rate** — success rate on a standardized benchmark before
  vs. after harness evolution.

This module derives approximations of those metrics from the events already
recorded by :class:`~foundry_x.trace.logger.TraceLogger`:

* ``cycle_time_seconds`` — mean wall-clock time from the first
  ``task_received`` event to the first ``critic_verdict`` event per session.
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

    Issue #585 adds ``hooks_disabled_count`` and ``hooks_disabled_rate``:
    the total count of ``hook_registry_error`` events and the fraction of
    sessions with at least one such event. Emitted when
    ``harness.hooks.get_registry()`` raises, disabling all hooks including
    the security-critical ``InjectionFirewallHook``.
    """

    cycle_time_seconds: float | None = None
    regression_rate: float = 0.0
    improvement_rate: float = 0.0
    injection_blocks: dict[str, int] = {}
    token_totals: dict[str, int] = {}
    hooks_disabled_count: int = 0
    hooks_disabled_rate: float = 0.0


class KpiComparison(BaseModel):
    """Baseline-vs-candidate harness-version comparison (issue #100).

    ``deltas`` holds the raw ``candidate - baseline`` difference for each
    numeric KPI; the rendering layer interprets the sign per the PRD's
    "good direction" — improvement-rate up is good, regression-rate and
    cycle-time down are good. ``injection_blocks`` is intentionally
    excluded from the comparison because it is an auxiliary signal, not
    one of the three PRD success-metric KPIs.
    """

    baseline: KpiSummary
    candidate: KpiSummary
    deltas: dict[str, float | None]


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

    return KpiSummary(
        cycle_time_seconds=cycle_time,
        regression_rate=regression_rate,
        improvement_rate=improvement_rate,
        injection_blocks=injection_blocks,
        token_totals=token_totals,
        hooks_disabled_count=hooks_disabled_count,
        hooks_disabled_rate=hooks_disabled_rate,
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
    """
    baseline = compute_kpis(logger, harness_version=baseline_version)
    candidate = compute_kpis(logger, harness_version=candidate_version)
    return KpiComparison(
        baseline=baseline,
        candidate=candidate,
        deltas=_compute_deltas(baseline, candidate),
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
        usage = event.payload.get("usage")
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
    emitted via :meth:`KpiSummary.model_dump` with ``injection_blocks``
    and ``token_totals`` excluded (the "minus per-session maps" half of
    the round-trip contract). ``hooks_disabled_count`` and
    ``hooks_disabled_rate`` are scalar fields and are included. Then
    ``timestamp`` and the optional ``harness_version`` are added.
    Parent directories are created on demand so the operator does not
    have to ``mkdir`` before the first run.

    The file is opened in append mode and a single ``\\n``-terminated
    line is written per call, so concurrent appends from independent
    ``foundry-kpis`` invocations interleave cleanly at line
    boundaries rather than corrupting the JSON payload of the
    previous line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(mode="json", exclude={"injection_blocks", "token_totals"})
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


def render_history_markdown(entries: Sequence[KpiHistoryEntry]) -> str:
    """Render a Markdown trend table from KPI history entries (issue #183).

    The table preserves file order, which is the same as append order
    for a JSONL log. Each row carries the timestamp plus the three
    PRD KPIs formatted with two decimals; ``None`` cycle times render
    as ``N/A`` (same convention as :func:`_render_markdown`).

    An empty history renders a single placeholder line so CI summary
    cells that template-embed the table are never completely blank.
    Plotting (matplotlib, ASCII sparklines) is explicitly out of
    scope per the issue; a pure table is the contract.
    """
    if not entries:
        return "_No KPI history entries yet._"
    lines = [
        "| Timestamp | Cycle Time (s) | Regression Rate | Improvement Rate |",
        "| --- | --- | --- | --- |",
    ]
    for entry in entries:
        lines.append(
            f"| {entry.timestamp} | "
            f"{_format_value(entry.cycle_time_seconds)} | "
            f"{_format_value(entry.regression_rate)} | "
            f"{_format_value(entry.improvement_rate)} |"
        )
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
    args = parser.parse_args(argv)

    baseline_version = args.baseline_harness_version
    candidate_version = args.candidate_harness_version
    if (baseline_version is None) != (candidate_version is None):
        parser.error(
            "--baseline-harness-version and --candidate-harness-version must be supplied together"
        )

    if args.from_history is not None:
        # Issue #183: trend rendering is a pure read of the JSONL log;
        # it does not require a trace store, so we short-circuit before
        # opening the SQLite database. ``--out`` still works as a sink.
        entries = read_kpi_history(Path(args.from_history))
        output = render_history_markdown(entries)
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
    else:
        summary = compute_kpis(logger, harness_version=args.harness_version)
        # Issue #183: append-only history log. Comparison runs are
        # intentionally not logged (the history is per single-summary
        # run; a comparison is a one-off baseline-vs-candidate diff).
        if args.log_to is not None:
            append_kpi_history(
                Path(args.log_to),
                summary,
                harness_version=args.harness_version,
            )
        output = _render_json(summary) if fmt == "json" else _render_markdown(summary)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
