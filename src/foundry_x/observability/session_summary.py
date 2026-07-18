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

from foundry_x.observability.regression_report import VERDICT_KIND, VerdictRecord

OUTCOME_KIND = "outcome"
TASK_ABORTED_KIND = "task_aborted"
TOKEN_BUDGET_REASON = "token_budget"
CRITIC_VERDICT_KIND = VERDICT_KIND

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


CONTEXT_PRUNED_KIND = "context_pruned"


class SessionSummaryRow(BaseModel):
    """One row of the cross-session outcome roll-up table (issue #184).

    Mirrors the column order in the issue's acceptance criteria. The
    three outcome-derived fields are ``None`` when the underlying
    session has no recorded ``outcome`` event; the renderer maps each
    ``None`` to the ``_`` placeholder so that an Operator reading
    across a table can never confuse "no outcome event recorded" with
    a literal empty string.

    Issue #466 adds ``token_budget_hit``: ``True`` when the session
    recorded at least one ``task_aborted(reason="token_budget")``
    event, ``False`` when the session has outcome data but no token
    budget abort, and ``None`` when the session has no outcome event
    at all (the underscore placeholder is rendered in that case).

    Issue #626 adds ``context_pruned``: the number of ``context_pruned``
    events recorded for this session by the pruning hook. ``None`` when
    no pruning occurred for this session.

    Issue #737 adds ``failure_class``: the failure class from the
    session's ``critic_verdict`` event, if any.
    """

    session_id: str
    started_at: str
    duration_seconds: float | None
    outcome_status: str | None
    outcome_reason: str | None
    steps: int | None
    token_budget_hit: bool | None = None
    context_pruned: int | None = None
    failure_class: str | None = None


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


def _count_context_pruned_events(
    logger: TraceLogger,
    session_id: str,
) -> int:
    """Count ``context_pruned`` events for *session_id* (issue #626)."""
    count = 0
    for _ in logger.iter_events(session_id, kind=CONTEXT_PRUNED_KIND):
        count += 1
    return count


def _get_session_failure_class(logger: TraceLogger, session_id: str) -> str | None:
    """Return the failure class from the session's ``critic_verdict`` event (issue #737).

    Returns the ``failure_class`` field from the most recent ``critic_verdict``
    event for *session_id*, or ``None`` if no verdict or no failure_class
    is recorded.
    """
    for event in logger.iter_events(session_id, kind=CRITIC_VERDICT_KIND):
        record = VerdictRecord(**event.payload)
        if record.failure_class is not None:
            return record.failure_class
    return None


class SessionSummaryReport(BaseModel):
    """Report containing session summary rows and failure class distribution (issue #737).

    The ``failure_class_distribution`` field holds an aggregate count of verdicts
    grouped by failure class across all sessions. The ``rows`` field contains the
    per-session summary data.
    """

    failure_class_distribution: dict[str, int] = {}
    rows: list[SessionSummaryRow] = []


def _failure_class_distribution(
    rows: Sequence[SessionSummaryRow],
) -> dict[str, int]:
    """Compute ``failure_class_distribution`` from session summary rows (issue #737).

    Aggregates the ``failure_class`` values from *rows* into a
    ``failure_class -> count`` map.
    """
    distribution: dict[str, int] = {}
    for row in rows:
        if row.failure_class is not None:
            distribution[row.failure_class] = distribution.get(row.failure_class, 0) + 1
    return distribution


def build_session_summary(
    logger: TraceLogger,
    harness_version: str | None = None,
    since: str | None = None,
) -> list[SessionSummaryRow]:
    """Return one :class:`SessionSummaryRow` per session in *logger*.

    Rows are sorted newest-first by ``started_at`` (ISO-8601 lexical
    compare matches chronological order for consistent timezones).
    Sessions without a recorded ``outcome`` event produce a row whose
    ``outcome_status``, ``outcome_reason`` and ``steps`` fields are
    ``None``; :func:`render_session_summary` turns those into the
    ``_`` placeholder.

    Issue #466: ``token_budget_hit`` is ``True`` when at least one
    ``task_aborted(reason="token_budget")`` event was recorded for
    the session, ``False`` when an outcome exists but no token budget
    abort, and ``None`` when the session has no outcome event at all.

    *since* filters to sessions whose ``started_at`` is at or after the
    given ISO-8601 timestamp.
    """
    sessions = logger.list_sessions(harness_version=harness_version)
    if since is not None:
        sessions = [s for s in sessions if s.started_at >= since]
    rows: list[SessionSummaryRow] = []
    for session in sessions:
        outcome = _latest_outcome(logger.iter_events(session.session_id, kind=OUTCOME_KIND))
        payload = outcome.payload if outcome is not None else {}
        raw_steps = payload.get("steps")
        steps_value: int | None = raw_steps if isinstance(raw_steps, int) else None
        token_budget_hit = _has_token_budget_abort(logger, session.session_id)
        context_pruned_count = _count_context_pruned_events(logger, session.session_id)
        context_pruned_value: int | None = (
            context_pruned_count if context_pruned_count > 0 else None
        )
        failure_class = _get_session_failure_class(logger, session.session_id)
        rows.append(
            SessionSummaryRow(
                session_id=session.session_id,
                started_at=session.started_at,
                duration_seconds=_compute_duration(session.started_at, session.ended_at),
                outcome_status=_string_or_none(payload.get("status")),
                outcome_reason=_string_or_none(payload.get("reason")),
                steps=steps_value,
                token_budget_hit=token_budget_hit,
                context_pruned=context_pruned_value,
                failure_class=failure_class,
            )
        )
    rows.sort(key=lambda row: row.started_at, reverse=True)
    return rows


def _has_token_budget_abort(logger: TraceLogger, session_id: str) -> bool | None:
    """Return whether session has a ``task_aborted(reason="token_budget")`` event.

    Returns ``True`` when at least one such event exists, ``False``
    when the session has outcome data but no token budget abort, and
    ``None`` when the session has no ``outcome`` event at all (the
    caller uses this to decide whether to render ``_`` or a boolean).
    """
    has_outcome = False
    for event in logger.iter_events(session_id, kind=OUTCOME_KIND):
        has_outcome = True
        break
    if not has_outcome:
        return None
    for event in logger.iter_events(session_id, kind=TASK_ABORTED_KIND):
        if event.payload.get("reason") == TOKEN_BUDGET_REASON:
            return True
    return False


def _string_or_none(value: object) -> str | None:
    """Coerce *value* to ``str`` if it is one, otherwise return ``None``.

    Defensive against payloads that store the outcome fields as, say,
    an int or a list — neither should appear in practice but the
    renderer's ``None``→``_`` contract depends on getting ``None``
    back rather than a stringified version of something else.
    """
    return value if isinstance(value, str) else None


_TOKEN_BUDGET_HIT_WIDTH = 5


def render_session_summary(
    rows: Sequence[SessionSummaryRow],
    limit: int | None = None,
    failure_class_distribution: dict[str, int] | None = None,
) -> str:
    """Render *rows* as the cross-session roll-up table (issue #184).

    Columns: ``session_id  started_at  duration  outcome.status
    outcome.reason  steps  token_budget_hit``. Sessions without a
    recorded outcome event emit an underscore ``_`` in the three
    outcome-derived columns and ``_`` in ``token_budget_hit``.

    *limit* truncates the rendered set to the first ``limit`` rows
    *after* newest-first sorting so a small ``--limit N`` always shows
    the most recent N sessions, matching the issue's command-line
    promise.

    *failure_class_distribution* (issue #737), when provided, is
    rendered as a breakdown table after the main table.
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
            "budget_hit".rjust(_TOKEN_BUDGET_HIT_WIDTH),
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
        token_budget_cell = (
            _PLACEHOLDER.rjust(_TOKEN_BUDGET_HIT_WIDTH)
            if row.token_budget_hit is None
            else (
                "true".rjust(_TOKEN_BUDGET_HIT_WIDTH)
                if row.token_budget_hit
                else "false".rjust(_TOKEN_BUDGET_HIT_WIDTH)
            )
        )
        lines.append(
            "  ".join(
                [
                    row.session_id.ljust(_SESSION_ID_WIDTH),
                    row.started_at.ljust(_TIMESTAMP_WIDTH),
                    duration_cell,
                    status_cell.ljust(_OUTCOME_FIELD_WIDTH),
                    reason_cell.ljust(_OUTCOME_FIELD_WIDTH),
                    steps_cell,
                    token_budget_cell,
                ]
            )
        )

    if failure_class_distribution:
        total = sum(failure_class_distribution.values())
        lines.append("")
        lines.append(
            f"Failure Class Distribution: {total} verdict(s) across "
            f"{len(failure_class_distribution)} class(es)."
        )
        lines.append("")
        lines.append("| Failure Class | Count |")
        lines.append("| --- | --- |")
        for cls, count in sorted(failure_class_distribution.items()):
            lines.append(f"| {cls} | {count} |")

    return "\n".join(lines)
