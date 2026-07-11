"""Cross-session outcome roll-up table (issue #184).

The regression report shows task-level deltas; KPIs show aggregate
rates; the session card shows one session. ``session_summary``
provides the missing view: one row per session, in newest-first order,
showing the recorded outcome (``status`` / ``reason`` / ``steps``)
alongside the session's start time and wall-clock duration so an
Operator can spot trends like "all v2 sessions failed at max_steps"
without opening every individual session card.

This module is deliberately read-only: it does **not** invoke the
``Digester`` or ``Critic`` (those are agent-evolution concerns) and
it routes every read through :class:`TraceLogger` per ADR-0003 (no
raw SQL in business logic).
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

from pydantic import BaseModel

from foundry_x.trace.logger import TraceEvent, TraceLogger

OUTCOME_KIND = "outcome"

# The kind string persisted on terminal ``outcome`` events by
# ``src/foundry_x/execution/runner.py`` at lines 644-648. Each event's
# payload carries ``status``, ``reason`` and ``steps`` (issue #184
# acceptance: the three columns surface those three fields).

# Column widths for the fixed-width text table. Chosen so a UUIDv4
# (36 chars), an ISO-8601 timestamp in UTC (~25 chars), a short
# duration, the common ``outcome.status`` / ``outcome.reason``
# strings, and a small integer all fit on a single 120-column line.
_SESSION_ID_WIDTH = 36
_TIMESTAMP_WIDTH = 25
_DURATION_WIDTH = 10
_OUTCOME_FIELD_WIDTH = 16
_STEPS_WIDTH = 5

_PLACEHOLDER = "_"


class SessionSummaryRow(BaseModel):
    """One row of the cross-session outcome roll-up table (issue #184).

    Mirrors the column order in the issue's acceptance criteria. The
    three outcome-derived fields are ``None`` when the underlying
    session has no recorded ``outcome`` event; the renderer maps each
    ``None`` to the ``_`` placeholder so that an Operator reading
    across a table can never confuse "no outcome event recorded" with
    a literal empty string.
    """

    session_id: str
    started_at: str
    duration_seconds: float | None
    outcome_status: str | None
    outcome_reason: str | None
    steps: int | None


def _truncate(value: str, width: int) -> str:
    """Return *value* clipped to *width* with an ellipsis suffix.

    Used to keep the fixed-width table from being pushed wider by an
    unusually long ``outcome.reason``. An ellipsis suffix makes the
    truncation visible to a human reading the table.
    """
    if width <= 1:
        return ""
    if len(value) <= width:
        return value
    return value[: width - 1] + "\u2026"


def _format_duration(seconds: float | None) -> str:
    """Render a wall-clock duration as ``"<ms>"``, ``"s"``, or ``"m:ss"``."""
    if seconds is None or seconds < 0:
        return _PLACEHOLDER
    if seconds < 1:
        return f"{seconds * 1000:g}ms"
    if seconds < 60:
        return f"{seconds:.2f}s"
    minutes, rest = divmod(seconds, 60)
    return f"{int(minutes)}m{int(rest):02d}s"


def _compute_duration(started_at: str, ended_at: str | None) -> float | None:
    """Return ``ended_at - started_at`` in seconds, or ``None`` on parse failure.

    A ``None`` return value is indistinguishable from "session never
    ended" from this module's perspective; both render as an
    underscore placeholder in the table.
    """
    if ended_at is None:
        return None
    try:
        start = datetime.fromisoformat(started_at)
        end = datetime.fromisoformat(ended_at)
    except ValueError:
        return None
    return (end - start).total_seconds()


def _latest_outcome(events: Iterable[TraceEvent]) -> TraceEvent | None:
    """Return the most recent ``outcome`` event in *events*.

    ``outcome`` is a terminal kind — only one should ever be recorded
    per session by ``runner.py`` — but we walk the full stream and
    keep the latest timestamp so a buggy double-recording still picks
    the most recent value rather than the first.
    """
    latest: TraceEvent | None = None
    for event in events:
        latest = event
    return latest


def build_session_summary(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> list[SessionSummaryRow]:
    """Return one :class:`SessionSummaryRow` per session in *logger*.

    Rows are sorted newest-first by ``started_at`` (ISO-8601 lexical
    compare matches chronological order for consistent timezones).
    Sessions without a recorded ``outcome`` event produce a row whose
    ``outcome_status``, ``outcome_reason`` and ``steps`` fields are
    ``None``; :func:`render_session_summary` turns those into the
    ``_`` placeholder.
    """
    sessions = logger.list_sessions(harness_version=harness_version)
    rows: list[SessionSummaryRow] = []
    for session in sessions:
        outcome = _latest_outcome(logger.iter_events(session.session_id, kind=OUTCOME_KIND))
        payload = outcome.payload if outcome is not None else {}
        raw_steps = payload.get("steps")
        steps_value: int | None = raw_steps if isinstance(raw_steps, int) else None
        rows.append(
            SessionSummaryRow(
                session_id=session.session_id,
                started_at=session.started_at,
                duration_seconds=_compute_duration(session.started_at, session.ended_at),
                outcome_status=_string_or_none(payload.get("status")),
                outcome_reason=_string_or_none(payload.get("reason")),
                steps=steps_value,
            )
        )
    rows.sort(key=lambda row: row.started_at, reverse=True)
    return rows


def _string_or_none(value: object) -> str | None:
    """Coerce *value* to ``str`` if it is one, otherwise return ``None``.

    Defensive against payloads that store the outcome fields as, say,
    an int or a list — neither should appear in practice but the
    renderer's ``None``→``_`` contract depends on getting ``None``
    back rather than a stringified version of something else.
    """
    return value if isinstance(value, str) else None


def render_session_summary(
    rows: Sequence[SessionSummaryRow],
    limit: int | None = None,
) -> str:
    """Render *rows* as the cross-session roll-up table (issue #184).

    Columns: ``session_id  started_at  duration  outcome.status
    outcome.reason  steps``. Sessions without a recorded outcome event
    emit an underscore ``_`` in the three outcome-derived columns.

    *limit* truncates the rendered set to the first ``limit`` rows
    *after* newest-first sorting so a small ``--limit N`` always shows
    the most recent N sessions, matching the issue's command-line
    promise.
    """
    if not rows:
        return "no sessions"

    ordered = list(rows)
    if limit is not None:
        ordered = ordered[:limit]

    header = "  ".join(
        [
            "session_id".ljust(_SESSION_ID_WIDTH),
            "started_at".ljust(_TIMESTAMP_WIDTH),
            "duration".ljust(_DURATION_WIDTH),
            "outcome.status".ljust(_OUTCOME_FIELD_WIDTH),
            "outcome.reason".ljust(_OUTCOME_FIELD_WIDTH),
            "steps".rjust(_STEPS_WIDTH),
        ]
    )
    lines = [header]
    for row in ordered:
        status = row.outcome_status
        reason = row.outcome_reason
        status_cell = _truncate(_PLACEHOLDER if status is None else status, _OUTCOME_FIELD_WIDTH)
        reason_cell = _truncate(_PLACEHOLDER if reason is None else reason, _OUTCOME_FIELD_WIDTH)
        if row.steps is None:
            steps_cell = _PLACEHOLDER.rjust(_STEPS_WIDTH)
        else:
            steps_cell = str(row.steps).rjust(_STEPS_WIDTH)
        duration_cell = _format_duration(row.duration_seconds).ljust(_DURATION_WIDTH)
        lines.append(
            "  ".join(
                [
                    row.session_id.ljust(_SESSION_ID_WIDTH),
                    row.started_at.ljust(_TIMESTAMP_WIDTH),
                    duration_cell,
                    status_cell.ljust(_OUTCOME_FIELD_WIDTH),
                    reason_cell.ljust(_OUTCOME_FIELD_WIDTH),
                    steps_cell,
                ]
            )
        )
    return "\n".join(lines)
