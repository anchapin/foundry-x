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
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from typing import Sequence

from pydantic import BaseModel

from foundry_x.trace.logger import TraceLogger


class KpiSummary(BaseModel):
    """Structured summary of the three PRD KPIs."""

    cycle_time_seconds: float | None = None
    regression_rate: float = 0.0
    improvement_rate: float = 0.0


def compute_kpis(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> KpiSummary:
    """Compute KPIs from the trace store backing *logger*.

    Parameters
    ----------
    logger:
        A :class:`~foundry_x.trace.logger.TraceLogger` whose ``.path``
        points at a SQLite trace database.
    harness_version:
        When provided, only sessions created with this harness version are
        considered.
    """
    conn = sqlite3.connect(logger.path)
    try:
        session_ids = _session_ids(conn, harness_version)
        if not session_ids:
            return KpiSummary()

        cycle_time = _cycle_time(conn, session_ids)
        regression_rate, improvement_rate = _verdict_rates(conn, session_ids)
    finally:
        conn.close()

    return KpiSummary(
        cycle_time_seconds=cycle_time,
        regression_rate=regression_rate,
        improvement_rate=improvement_rate,
    )


def _session_ids(conn: sqlite3.Connection, harness_version: str | None) -> list[str]:
    if harness_version is not None:
        rows = conn.execute(
            "SELECT session_id FROM sessions WHERE harness_version = ?",
            (harness_version,),
        ).fetchall()
    else:
        rows = conn.execute("SELECT session_id FROM sessions").fetchall()
    return [row[0] for row in rows]


def _cycle_time(conn: sqlite3.Connection, session_ids: list[str]) -> float | None:
    deltas: list[float] = []
    for sid in session_ids:
        start = conn.execute(
            "SELECT timestamp FROM events "
            "WHERE session_id = ? AND kind = 'task_received' "
            "ORDER BY timestamp LIMIT 1",
            (sid,),
        ).fetchone()
        end = conn.execute(
            "SELECT timestamp FROM events "
            "WHERE session_id = ? AND kind = 'critic_verdict' "
            "ORDER BY timestamp LIMIT 1",
            (sid,),
        ).fetchone()
        if start and end:
            t0 = datetime.fromisoformat(start[0])
            t1 = datetime.fromisoformat(end[0])
            delta = (t1 - t0).total_seconds()
            if delta > 0:
                deltas.append(delta)
    if not deltas:
        return None
    return sum(deltas) / len(deltas)


def _verdict_rates(conn: sqlite3.Connection, session_ids: list[str]) -> tuple[float, float]:
    total_verdicts = 0
    approved = 0
    regression_sessions = 0
    sessions_with_verdicts = 0

    for sid in session_ids:
        rows = conn.execute(
            "SELECT payload FROM events " "WHERE session_id = ? AND kind = 'critic_verdict'",
            (sid,),
        ).fetchall()
        if not rows:
            continue
        sessions_with_verdicts += 1
        session_has_regression = False
        for (payload_str,) in rows:
            payload = json.loads(payload_str)
            total_verdicts += 1
            if payload.get("verdict") == "approved":
                approved += 1
            if payload.get("regression"):
                session_has_regression = True
        if session_has_regression:
            regression_sessions += 1

    improvement_rate = approved / total_verdicts if total_verdicts else 0.0
    regression_rate = (
        regression_sessions / sessions_with_verdicts if sessions_with_verdicts else 0.0
    )
    return regression_rate, improvement_rate


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
