from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass

from pydantic import BaseModel, Field

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.trace.logger import TraceLogger

VERDICT_KIND = "critic_verdict"


class VerdictRecord(BaseModel):
    """Structured payload persisted for every Critic verdict (ADR-0006 boundary model)."""

    approved: bool = False
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    notes: str = ""


@dataclass
class _Regression:
    task: str
    was_passing_session: str
    now_failing_session: str


@dataclass
class _NewPass:
    task: str
    was_failing_session: str
    now_passing_session: str


def record_verdict(logger: TraceLogger, session_id: str, verdict: CriticVerdict) -> None:
    """Persist a CriticVerdict as a ``critic_verdict`` trace event."""
    record = VerdictRecord(
        approved=verdict.approved,
        passed_checks=list(verdict.passed_checks),
        failed_checks=list(verdict.failed_checks),
        notes=verdict.notes,
    )
    logger.record(session_id=session_id, kind=VERDICT_KIND, payload=record.model_dump())


def _load_verdict_events(
    logger: TraceLogger,
    since: str | None,
) -> list[tuple[str, str, VerdictRecord]]:
    query = (
        "SELECT session_id, timestamp, payload FROM events "
        "WHERE kind = ?"
    )
    params: list[object] = [VERDICT_KIND]
    if since is not None:
        query += " AND timestamp >= ?"
        params.append(since)
    query += " ORDER BY timestamp ASC, rowid ASC"
    events: list[tuple[str, str, VerdictRecord]] = []
    with sqlite3.connect(logger.path) as conn:
        rows = conn.execute(query, tuple(params)).fetchall()
    for session_id, timestamp, payload in rows:
        events.append((session_id, timestamp, VerdictRecord(**json.loads(payload))))
    return events


def _compute(
    events: list[tuple[str, str, VerdictRecord]],
) -> tuple[list[_Regression], list[_NewPass]]:
    prior_passed: dict[str, str] = {}
    prior_failed: dict[str, str] = {}
    regressions: list[_Regression] = []
    new_passes: list[_NewPass] = []
    for session_id, _timestamp, verdict in events:
        for task in verdict.failed_checks:
            if task in prior_passed:
                regressions.append(
                    _Regression(
                        task=task,
                        was_passing_session=prior_passed[task],
                        now_failing_session=session_id,
                    )
                )
        for task in verdict.passed_checks:
            if task in prior_failed:
                new_passes.append(
                    _NewPass(
                        task=task,
                        was_failing_session=prior_failed[task],
                        now_passing_session=session_id,
                    )
                )
        for task in verdict.passed_checks:
            prior_passed[task] = session_id
        for task in verdict.failed_checks:
            prior_failed[task] = session_id
    return regressions, new_passes


def generate_regression_report(logger: TraceLogger, since: str | None = None) -> str:
    """Produce a Markdown regression report over all persisted Critic verdicts."""
    events = _load_verdict_events(logger, since)
    total = len(events)
    approvals = sum(1 for _sid, _ts, v in events if v.approved)
    rejections = total - approvals
    regressions, new_passes = _compute(events)
    return _render(total, approvals, rejections, regressions, new_passes)


def _render(
    total: int,
    approvals: int,
    rejections: int,
    regressions: list[_Regression],
    new_passes: list[_NewPass],
) -> str:
    lines: list[str] = [
        "# Critic Regression Report",
        "",
        "## Regression Summary",
        "",
        f"- Total verdicts: {total}",
        f"- Approvals: {approvals}",
        f"- Rejections: {rejections}",
        "",
        "## Regressed Tasks",
        "",
    ]
    if regressions:
        lines.append("| Task | Was passing (session) | Now failing (session) |")
        lines.append("| --- | --- | --- |")
        for reg in regressions:
            lines.append(
                f"| {reg.task} | {reg.was_passing_session} | {reg.now_failing_session} |"
            )
    else:
        lines.append("_None._")
    lines += ["", "## New Passes", ""]
    if new_passes:
        lines.append("| Task | Was failing (session) | Now passing (session) |")
        lines.append("| --- | --- | --- |")
        for pas in new_passes:
            lines.append(
                f"| {pas.task} | {pas.was_failing_session} | {pas.now_passing_session} |"
            )
    else:
        lines.append("_None._")
    lines.append("")
    return "\n".join(lines)
