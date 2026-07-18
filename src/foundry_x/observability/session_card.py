"""Per-session triage card renderer (issue #180).

The :func:`format_session_card` function renders a one-screen summary of a
trace session so a reviewer handed a ``session_id`` can triage a run in
under 60 seconds. It is the counterpart to :func:`foundry_x.observability.timeline.format_timeline`:
where ``format_timeline`` prints every event in order, ``format_session_card``
prints a fixed roll-up of harness version, model, duration, outcome,
event counts, and the first failure-bearing event.

The card is intentionally pure — it accepts a :class:`TraceSession` and
an ordered sequence of :class:`TraceEvent` objects and returns a single
string. The CLI layer (issue #180 §"fx-trace session-card <sid>") is
responsible for opening the trace store, deciding which events to load,
and turning exit codes into the 0 / non-zero contract the issue
requires. Keeping the formatter pure lets the unit tests pin the
golden-string output (issue #180 acceptance: "synthetic session yields
the expected lines").

Issue #871 — ``model_retry`` does not contain a generic failure token, so
it is explicitly included in the failure-kind predicate. This surfaces the
per-session retry count in ``errors_by_kind`` as an API reliability signal.
The session-aggregate KPI lives in :mod:`foundry_x.observability.kpis`.

Issue #872 — the runner emits ``tool_argument_parse_error`` events when
the model produces malformed tool-call arguments (see
``src/foundry_x/execution/runner.py:1684``). The kind's ``"error"``
substring matches ``_ERROR_KIND_RE`` below, so it is automatically
counted by the per-kind ``errors_by_kind`` bucket without any
card-specific code path. The dedicated session-aggregate KPI lives in
:mod:`foundry_x.observability.kpis` (``KpiSummary.tool_argument_parse_error_count``).

Issue #869 — the runner emits ``task_aborted(reason="event_limit")`` when
the per-session event cap is exceeded (see
``src/foundry_x/execution/runner.py:1523``). The kind's ``"abort"``
substring matches ``_ERROR_KIND_RE`` below, so it is automatically
counted by the per-kind ``errors_by_kind`` bucket as ``task_aborted=N``
without any card-specific code path. The dedicated session-aggregate KPI
lives in :mod:`foundry_x.observability.kpis`
(``KpiSummary.event_limit_abort_count``).
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import Any, Sequence

from foundry_x.evolution.digester import FAILURE_KINDS, INJECTION_BLOCKED_KIND
from foundry_x.observability.kpis import MODEL_RETRY_KIND
from foundry_x.trace.logger import TraceEvent, TraceSession

# A failure-bearing ``kind`` either matches this regex (so future kinds
# like ``abort_loop`` or ``commit_failed`` light up without code edits) or
# appears in the digester's curated :data:`FAILURE_KINDS` set. The
# ``outcome`` kind is intentionally *not* a failure — it is a terminal
# marker whose ``status`` field already carries success/failure, and is
# surfaced separately in its own card line.
_ERROR_KIND_RE = re.compile(r"error|fail|abort", re.IGNORECASE)

# Stable column width so the rendered card is line-for-line
# reproducible; the golden test relies on it.
_LABEL_WIDTH = 16
_VALUE_INDENT = "  "

# Sentinel strings (issue #180: "sessions with no `outcome` event degrade
# gracefully — read '_no outcome_'"). Keeping them in module-level
# constants makes the contract visible in one place.
_NO_OUTCOME = "_no outcome_"
_NONE = "_none_"
_UNKNOWN = "_unknown_"

# Tool-call events whose payload lacks a string ``name`` get bucketed
# under this placeholder so the per-name count is always well-defined.
_UNTITLED_TOOL = "<unnamed>"

# Truncation length for the inline first-error message.
_ERROR_SNIPPET_LIMIT = 120

# First-N chars of an event_id surfaced on the first_error line.
_EVENT_ID_PREFIX = 8

# Payload keys (most-specific first) tried when building the inline
# first-error snippet. Matches the digester's :data:`FAILURE_PAYLOAD_KEYS`
# vocabulary plus a few common synonyms so tool events with a plain
# ``message`` still surface a useful hint.
_ERROR_PAYLOAD_KEYS: tuple[str, ...] = (
    "error",
    "message",
    "exception",
    "traceback",
)


def _parse_ts(value: str | None) -> datetime | None:
    """Return the parsed timestamp or ``None`` on any failure.

    Tolerant of missing or malformed values: a session card never wants
    to abort because ``started_at`` was hand-edited into garbage.
    """
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None


def _format_duration(start: str, end: str | None) -> str:
    """Return ``end - start`` formatted via ``str(timedelta)`` or ``_unknown_``."""
    s = _parse_ts(start)
    e = _parse_ts(end)
    if s is None or e is None:
        return _UNKNOWN
    return str(e - s)


def _is_error_kind(kind: str) -> bool:
    return (
        kind in FAILURE_KINDS
        or kind in {INJECTION_BLOCKED_KIND, MODEL_RETRY_KIND}
        or bool(_ERROR_KIND_RE.search(kind))
    )


def _tool_call_name(payload: dict[str, Any]) -> str:
    name = payload.get("name")
    if isinstance(name, str) and name:
        return name
    return _UNTITLED_TOOL


def _first_error_event(events: Sequence[TraceEvent]) -> TraceEvent | None:
    for event in events:
        if _is_error_kind(event.kind):
            return event
    return None


def _summarize_error_payload(payload: dict[str, Any]) -> str:
    """Return a single-line inline snippet from an error event payload."""
    for key in _ERROR_PAYLOAD_KEYS:
        value = payload.get(key)
        if not isinstance(value, str) or not value:
            continue
        text = value.replace("\n", " ").strip()
        if text:
            if len(text) > _ERROR_SNIPPET_LIMIT:
                return text[: _ERROR_SNIPPET_LIMIT - 1] + "\u2026"
            return text
    return "-"


def _outcome_payload(events: Sequence[TraceEvent]) -> dict[str, Any] | None:
    """Return the payload of the most-recent ``outcome`` event or ``None``.

    If the runner emitted more than one ``outcome`` event (e.g. when the
    loop recovered and re-terminated), the latest one wins — that is the
    terminal state of the session.
    """
    latest: dict[str, Any] | None = None
    for event in events:
        if event.kind == "outcome":
            latest = dict(event.payload)
    return latest


def _format_line(label: str, value: str) -> str:
    """Render one ``label / value`` row at the fixed column width."""
    return f"{label:<{_LABEL_WIDTH}}{_VALUE_INDENT}{value}".rstrip()


def format_session_card(
    session: TraceSession,
    events: Sequence[TraceEvent],
) -> str:
    """Render a one-screen per-session triage card.

    Produces the lines required by issue #180 §Acceptance criteria:

    * ``harness_version`` — from the :class:`TraceSession`.
    * ``model_id`` — ``-`` if absent.
    * ``started_at`` / ``ended_at`` — ISO-8601 as recorded, ``-`` if absent.
    * ``duration`` — ``str(ended_at - started_at)`` or ``_unknown_``.
    * ``outcome`` — ``status=… reason=… steps=…`` derived from the most
      recent ``outcome`` event. If no such event exists the line reads
      ``_no outcome_`` (degrades gracefully for in-flight sessions).
    * ``event_count`` — number of events in ``events``.
    * ``tool_calls`` — ``name=N`` pairs sorted by name, ``_none_`` if empty.
    * ``errors_by_kind`` — ``kind=N`` pairs sorted by kind, ``_none_`` if empty.
    * ``first_error`` — ``<id_prefix> <kind>: <snippet>`` for the first
      failure-bearing event in timestamp order; ``_none_`` when none.

    The output is a single deterministic string with one ``label: value``
    pair per line and a fixed label column width; the golden test in
    ``tests/test_session_card.py`` pins this output line-for-line.
    """
    tool_calls_by_name: Counter[str] = Counter()
    errors_by_kind: Counter[str] = Counter()
    for event in events:
        if event.kind == "tool_call":
            tool_calls_by_name[_tool_call_name(event.payload)] += 1
        if _is_error_kind(event.kind):
            errors_by_kind[event.kind] += 1

    outcome = _outcome_payload(events)
    first_error = _first_error_event(events)
    duration = _format_duration(session.started_at, session.ended_at)

    outcome_value = (
        f"status={outcome.get('status', '-')} "
        f"reason={outcome.get('reason', '-')} "
        f"steps={outcome.get('steps', '-')}"
        if outcome is not None
        else _NO_OUTCOME
    )

    tool_calls_value = (
        ", ".join(f"{name}={count}" for name, count in sorted(tool_calls_by_name.items()))
        if tool_calls_by_name
        else _NONE
    )

    errors_value = (
        ", ".join(f"{kind}={count}" for kind, count in sorted(errors_by_kind.items()))
        if errors_by_kind
        else _NONE
    )

    if first_error is not None:
        # Trim trailing word delimiters at the prefix boundary so a
        # synthetic 12-char id like ``evt-bash-err01`` displays as
        # ``evt-bash`` rather than ``evt-bash-``. Real UUIDs never
        # contain ``-`` past position 8, so the strip is a no-op in
        # production but keeps the card clean in tests.
        prefix = first_error.event_id[:_EVENT_ID_PREFIX].rstrip("-_")
        snippet = _summarize_error_payload(first_error.payload)
        first_error_value = f"{prefix} {first_error.kind}: {snippet}"
    else:
        first_error_value = _NONE

    lines = [
        _format_line("session_id", session.session_id),
        _format_line("harness_version", session.harness_version),
        _format_line("model_id", session.model_id or "-"),
        _format_line("started_at", session.started_at),
        _format_line("ended_at", session.ended_at or "-"),
        _format_line("duration", duration),
        _format_line("outcome", outcome_value),
        _format_line("event_count", str(len(events))),
        _format_line("tool_calls", tool_calls_value),
        _format_line("errors_by_kind", errors_value),
        _format_line("first_error", first_error_value),
    ]
    return "\n".join(lines)
