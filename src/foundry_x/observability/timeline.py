from __future__ import annotations

import re
import sys
from datetime import datetime
from typing import Any, Sequence

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
    """
    if not events:
        return ""

    base = _parse_timestamp(events[0].timestamp)
    marker = _error_marker() if highlight_errors else ""
    lines: list[str] = []

    for index, event in enumerate(events, start=1):
        delta = (_parse_timestamp(event.timestamp) - base).total_seconds()
        offset = _format_offset(delta)

        prefix = "  "
        if highlight_errors and _ERROR_PATTERN.search(event.kind):
            prefix = f"{marker} "

        kind = event.kind.ljust(_KIND_COLUMN)
        summary = _extract_summary(event.payload)
        step = f"#{index}".ljust(_STEP_NUM_WIDTH + 1)

        lines.append(f"{prefix}{step} {offset:>{_OFFSET_WIDTH}}  {kind} {summary}".rstrip())

    return "\n".join(lines)
