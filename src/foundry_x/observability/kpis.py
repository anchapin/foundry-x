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
* ``regression_rate`` — fraction of sessions whose ``critic_verdict`` payload
  carries ``regression: true``.
* ``improvement_rate`` — fraction of ``critic_verdict`` events whose payload
  verdict is ``"approved"``.

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
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from typing import Sequence

from pydantic import BaseModel

from foundry_x.evolution.digester import INJECTION_BLOCKED_KIND
from foundry_x.trace.logger import TraceEvent, TraceLogger


class KpiSummary(BaseModel):
    """Structured summary of the three PRD KPIs.

    Issue #120 adds ``injection_blocks``: a ``session_id -> count`` map
    of ``injection_blocked`` events per session, sourced from the firewall
    hook. Empty by default; populated only when the trace store has at
    least one ``injection_blocked`` event.
    """

    cycle_time_seconds: float | None = None
    regression_rate: float = 0.0
    improvement_rate: float = 0.0
    injection_blocks: dict[str, int] = {}


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
    """
    # Issue #82: ``list_sessions`` and ``iter_events`` are the only paths
    # to the underlying store; raw SQL is centralized on the logger.
    sessions = logger.list_sessions(harness_version=harness_version)
    if not sessions:
        return KpiSummary()

    session_ids = [s.session_id for s in sessions]
    cycle_time = _cycle_time(logger, session_ids)
    regression_rate, improvement_rate = _verdict_rates(logger, session_ids)
    injection_blocks = _injection_blocks(logger, session_ids)

    return KpiSummary(
        cycle_time_seconds=cycle_time,
        regression_rate=regression_rate,
        improvement_rate=improvement_rate,
        injection_blocks=injection_blocks,
    )


def _first_event_of_kind(logger: TraceLogger, session_id: str, kind: str) -> TraceEvent | None:
    """Return the first event of *kind* in *session_id* via iter_events.

    ``iter_events`` orders by timestamp and yields one row at a time, so
    we stop at the first match without scanning the rest of the session.
    """
    for event in logger.iter_events(session_id, kind=kind):
        return event
    return None


def _cycle_time(logger: TraceLogger, session_ids: Sequence[str]) -> float | None:
    deltas: list[float] = []
    for sid in session_ids:
        start_event = _first_event_of_kind(logger, sid, "task_received")
        end_event = _first_event_of_kind(logger, sid, "critic_verdict")
        if start_event is None or end_event is None:
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


def _verdict_rates(logger: TraceLogger, session_ids: Sequence[str]) -> tuple[float, float]:
    total_verdicts = 0
    approved = 0
    regression_sessions = 0
    sessions_with_verdicts = 0

    for sid in session_ids:
        session_has_regression = False
        session_had_verdict = False
        for event in logger.iter_events(sid, kind="critic_verdict"):
            session_had_verdict = True
            total_verdicts += 1
            payload = event.payload
            if payload.get("verdict") == "approved":
                approved += 1
            if payload.get("regression"):
                session_has_regression = True
        if session_had_verdict:
            sessions_with_verdicts += 1
            if session_has_regression:
                regression_sessions += 1

    improvement_rate = approved / total_verdicts if total_verdicts else 0.0
    regression_rate = (
        regression_sessions / sessions_with_verdicts if sessions_with_verdicts else 0.0
    )
    return regression_rate, improvement_rate


def _injection_blocks(logger: TraceLogger, session_ids: Sequence[str]) -> dict[str, int]:
    """Per-session count of ``injection_blocked`` events (issue #120).

    Returns a ``session_id -> count`` map including only sessions with at
    least one block. Sessions without blocks are omitted so the rendering
    path can decide whether to add an extra section based on the map being
    non-empty (per the issue's "show … when at least one is present").
    """
    blocks: dict[str, int] = {}
    for sid in session_ids:
        # ``iter_events(kind=...)`` pushes the kind filter down to the
        # underlying store, so we only ever see injection_blocked rows.
        count = sum(1 for _ in logger.iter_events(sid, kind=INJECTION_BLOCKED_KIND))
        if count:
            blocks[sid] = count
    return blocks


def _format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _render_markdown(summary: KpiSummary) -> str:
    lines = [
        "| KPI | Value |",
        "| --- | --- |",
        f"| Cycle Time (seconds) | {_format_value(summary.cycle_time_seconds)} |",
        f"| Regression Rate | {_format_value(summary.regression_rate)} |",
        f"| Improvement Rate | {_format_value(summary.improvement_rate)} |",
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
    return "\n".join(lines)


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
    args = parser.parse_args(argv)

    logger = TraceLogger(args.db)
    summary = compute_kpis(logger, harness_version=args.harness_version)
    print(_render_markdown(summary))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
