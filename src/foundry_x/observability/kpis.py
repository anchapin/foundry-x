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
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Sequence

from pydantic import BaseModel

from foundry_x.evolution.digester import INJECTION_BLOCKED_KIND
from foundry_x.observability.regression_report import VerdictRecord
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
    """Derive regression and improvement rates from persisted Critic verdicts.

    Verdicts are persisted as the :class:`VerdictRecord` shape
    (``approved`` / ``passed_checks`` / ``failed_checks`` / ``notes``), not the
    synthetic ``{"verdict", "regression"}`` payload the earlier implementation
    assumed (issue #98). Uses ``logger.iter_events()`` per ADR-0003 (no raw SQL).

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

    for sid in session_ids:
        for event in logger.iter_events(sid, kind="critic_verdict"):
            total_verdicts += 1
            sessions_with_verdicts.add(sid)
            record = VerdictRecord(**event.payload)
            if record.approved:
                approved += 1
            for task in record.failed_checks:
                if task in prior_passed:
                    regression_sessions.add(sid)
            for task in record.passed_checks:
                prior_passed[task] = sid

    improvement_rate = approved / total_verdicts if total_verdicts else 0.0
    regression_rate = (
        len(regression_sessions) / len(sessions_with_verdicts) if sessions_with_verdicts else 0.0
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
    args = parser.parse_args(argv)

    fmt = _resolve_format(args.format, args.out)
    logger = TraceLogger(args.db)
    summary = compute_kpis(logger, harness_version=args.harness_version)
    output = _render_json(summary) if fmt == "json" else _render_markdown(summary)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
