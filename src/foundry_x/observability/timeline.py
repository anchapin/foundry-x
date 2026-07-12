from __future__ import annotations

import json
import re
import sys
from datetime import datetime
from typing import Any, Sequence

from pydantic import BaseModel

from foundry_x.trace.logger import TraceEvent

# Kinds whose name suggests a failure get an error marker prefix.
_ERROR_PATTERN = re.compile(r"error|fail|abort", re.IGNORECASE)

# Layout constants ----------------------------------------------------------
_KIND_COLUMN = 16
_SUMMARY_LIMIT = 60
_STEP_NUM_WIDTH = 2
_OFFSET_WIDTH = 6


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _format_offset(delta_seconds: float) -> str:
    sign = "+" if delta_seconds >= 0 else ""
    return f"{sign}{delta_seconds:.1f}s"


def _error_marker() -> str:
    try:
        if sys.stdout.isatty():
            return "\u2717"
    except (AttributeError, ValueError):
        pass
    return "!"


def _format_latency(payload: dict[str, Any]) -> str:
    duration_ms = payload.get("duration_ms")
    if isinstance(duration_ms, bool) or not isinstance(duration_ms, (int, float)):
        return ""
    return f"{duration_ms:g}ms"


def _with_latency(summary: str, payload: dict[str, Any]) -> str:
    latency = _format_latency(payload)
    if not latency:
        return summary
    if not summary:
        return latency
    return f"{summary} ({latency})"


def _with_token_total(summary: str, payload: dict[str, Any]) -> str:
    """Annotate a ``model_response`` line with its cumulative token count.

    Issue #271 — the runner records a running ``tokens_used`` counter on
    every ``model_response`` event (issue #197). Showing that running
    total inline lets an operator watch a session burn through its
    ``FOUNDRY_TOKEN_BUDGET`` without leaving the timeline. A missing or
    non-integer ``tokens_used`` (endpoint that omits accounting, or an
    event written before the field landed) yields no annotation.
    """
    tokens = payload.get("tokens_used")
    if isinstance(tokens, bool) or not isinstance(tokens, int):
        return summary
    annotation = f"tokens:{tokens}"
    if not summary:
        return annotation
    return f"{summary} [{annotation}]"


def _extract_summary(payload: dict[str, Any]) -> str:
    """Return a one-line human summary for an event payload.

    Priority: ``name`` (tool identifier) then free-text fields
    (``prompt``, ``text``, ``message``, ``error``) truncated to
    ``_SUMMARY_LIMIT`` chars, then ``status``/``result``.
    """
    name = payload.get("name")
    if isinstance(name, str) and name:
        return _with_latency(name, payload)
    for key in ("prompt", "text", "message", "error"):
        value = payload.get(key)
        if value is None:
            continue
        text = str(value).replace("\n", " ").strip()
        if text:
            return _with_latency(text[:_SUMMARY_LIMIT], payload)
    for key in ("status", "result"):
        value = payload.get(key)
        if value is not None:
            return _with_latency(str(value), payload)
    return _with_latency("", payload)


class TimelineRecord(BaseModel):
    """One structured timeline event for JSON consumers (issue #270).

    Mirrors the columns of :func:`format_timeline` so programmatic
    consumers (the Evolver, CI tooling) do not have to reverse-parse
    the formatted text. ``step`` is 1-indexed; ``offset_seconds`` is
    the wall-clock delta from the first event in the sequence;
    ``summary`` is the same one-line extraction the text renderer uses
    (including latency and, for ``model_response``, the cumulative
    token-total annotation from issue #271); ``is_error`` is ``True``
    when the event ``kind`` matches the error/fail/abort pattern that
    the text renderer flags with a leading marker.
    """

    step: int
    offset_seconds: float
    kind: str
    summary: str
    is_error: bool


def build_timeline_records(events: Sequence[TraceEvent]) -> list[TimelineRecord]:
    """Return one :class:`TimelineRecord` per event (issue #270).

    The text renderer (:func:`format_timeline`) and the JSON renderer
    (:func:`render_timeline_json`) share this builder so their
    ``summary`` and ``is_error`` columns never drift apart. An empty
    event sequence yields an empty list.
    """
    if not events:
        return []
    base = _parse_timestamp(events[0].timestamp)
    records: list[TimelineRecord] = []
    for index, event in enumerate(events, start=1):
        delta = (_parse_timestamp(event.timestamp) - base).total_seconds()
        summary = _extract_summary(event.payload)
        # Issue #271: keep the JSON summary faithful to the text line so a
        # consumer reading either surface sees the same token annotation.
        if event.kind == "model_response":
            summary = _with_token_total(summary, event.payload)
        records.append(
            TimelineRecord(
                step=index,
                offset_seconds=delta,
                kind=event.kind,
                summary=summary,
                is_error=bool(_ERROR_PATTERN.search(event.kind)),
            )
        )
    return records


def render_timeline_json(events: Sequence[TraceEvent]) -> str:
    """Render trace ``events`` as a JSON array (issue #270).

    Each element is a :class:`TimelineRecord` carrying ``step``,
    ``offset_seconds``, ``kind``, ``summary`` and ``is_error``. An
    empty event sequence renders as ``[]`` so JSON consumers never
    have to special-case a missing array.
    """
    return json.dumps(
        [record.model_dump() for record in build_timeline_records(events)],
        indent=2,
    )


def format_timeline(
    events: Sequence[TraceEvent],
    highlight_errors: bool = True,
) -> str:
    """Render trace ``events`` as a human-readable timeline.

    Each event produces one line with an incrementing step number, the
    relative offset from the first event's timestamp (e.g. ``+0.3s``),
    the event ``kind`` left-justified to a fixed column, and a one-line
    summary extracted from the payload.

    When *highlight_errors* is ``True`` (default), events whose ``kind``
    contains ``error``, ``fail``, or ``abort`` are prefixed with an
    error marker — ``\u2717`` when stdout is a TTY, the plain ASCII
    ``!`` otherwise so output stays greppable in pipes and logs.

    The per-event observation (offset, summary, error flag) is produced
    by :func:`build_timeline_records`, which :func:`render_timeline_json`
    also consumes, so the text and JSON surfaces stay faithful mirrors
    of each other.
    """
    records = build_timeline_records(events)
    if not records:
        return ""

    marker = _error_marker() if highlight_errors else ""
    lines: list[str] = []

    for record in records:
        offset = _format_offset(record.offset_seconds)

        prefix = "  "
        if highlight_errors and record.is_error:
            prefix = f"{marker} "

        kind = record.kind.ljust(_KIND_COLUMN)
        step = f"#{record.step}".ljust(_STEP_NUM_WIDTH + 1)

        lines.append(f"{prefix}{step} {offset:>{_OFFSET_WIDTH}}  {kind} {record.summary}".rstrip())

    return "\n".join(lines)
