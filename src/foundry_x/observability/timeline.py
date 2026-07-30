from __future__ import annotations

import json
import re
import sys
from collections.abc import Sequence
from datetime import datetime
from typing import Any

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
    (``prompt``, ``text``, ``message``, ``error_type``, ``error``) truncated to
    ``_SUMMARY_LIMIT`` chars, then ``status``/``result``.
    """
    name = payload.get("name")
    if isinstance(name, str) and name:
        return _with_latency(name, payload)
    for key in ("prompt", "text", "message", "error_type", "error"):
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


def _kind_color(kind: str) -> str:
    if _ERROR_PATTERN.search(kind):
        return "#ef4444"
    if kind.startswith("model_"):
        return "#3b82f6"
    if kind.startswith("tool_"):
        return "#22c55e"
    if kind in ("trace_event", "span"):
        return "#a855f7"
    return "#6b7280"


def _kind_row(kind: str) -> str:
    if kind.startswith("model_"):
        return "model"
    if kind.startswith("tool_"):
        return "tool"
    if kind in ("trace_event", "span"):
        return "trace"
    return "other"


def render_timeline_svg(events: Sequence[TraceEvent]) -> str:
    records = build_timeline_records(events)
    if not records:
        return ""

    total_duration = max((r.offset_seconds for r in records), default=1)
    if total_duration == 0:
        total_duration = 1
    padding = 40
    row_height = 24
    row_spacing = 8
    label_width = 100
    chart_width = 800
    chart_height = padding * 2 + len(records) * (row_height + row_spacing)

    rows: list[str] = []
    for r in records:
        row = _kind_row(r.kind)
        if row not in rows:
            rows.append(row)
    row_to_y: dict[str, float] = {
        row: padding + i * (row_height + row_spacing) for i, row in enumerate(rows)
    }

    bars: list[str] = []
    for record in records:
        y = row_to_y[_kind_row(record.kind)]
        color = _kind_color(record.kind)
        x = (record.offset_seconds / total_duration) * chart_width + label_width
        width = (
            max(2, (record.offset_seconds / total_duration) * chart_width * 0.1)
            if record.offset_seconds > 0
            else 2
        )
        error_attrs = ' stroke="#ef4444" stroke-width="2"' if record.is_error else ""
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" height="{row_height}"'
            f' fill="{color}" rx="3"{error_attrs}/>'
        )
        if record.summary:
            bars.append(
                f'<text x="{x + width + 4:.1f}" y="{y + row_height - 5:.1f}"'
                f' font-size="10" fill="#374151">{_escape_svg(record.summary[:40])}</text>'
            )

    legend_items = [
        (
            '<rect x="0" y="0" width="12" height="12" fill="#3b82f6" rx="2"/><text x="16" y="11" font-size="11" fill="#374151">model_*</text>',
            0,
        ),
        (
            '<rect x="0" y="0" width="12" height="12" fill="#22c55e" rx="2"/><text x="16" y="11" font-size="11" fill="#374151">tool_*</text>',
            1,
        ),
        (
            '<rect x="0" y="0" width="12" height="12" fill="#a855f7" rx="2"/><text x="16" y="11" font-size="11" fill="#374151">trace/span</text>',
            2,
        ),
        (
            '<rect x="0" y="0" width="12" height="12" fill="#6b7280" rx="2"/><text x="16" y="11" font-size="11" fill="#374151">other</text>',
            3,
        ),
        (
            '<rect x="0" y="0" width="12" height="12" fill="#ef4444" rx="2"/><text x="16" y="11" font-size="11" fill="#374151">error</text>',
            4,
        ),
    ]
    legend_x_start = label_width + 20
    legend_spacing = 90
    legend_bars = []
    for item, idx in legend_items:
        lx = legend_x_start + idx * legend_spacing
        legend_bars.append(f'<g transform="translate({lx}, 10)">{item}</g>')

    svg_parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {label_width + chart_width + 40} {chart_height}">',
        "<style>text{font-family:ui-monospace,monospace}</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<rect x="{label_width}" y="{padding}" width="{chart_width}" height="{len(rows) * (row_height + row_spacing) - row_spacing}" fill="#f9fafb" rx="4"/>',
    ]
    svg_parts.extend(bars)
    svg_parts.extend(legend_bars)
    svg_parts.append(
        f'<text x="{label_width + chart_width // 2}" y="{chart_height - 6}" font-size="11" fill="#9ca3af" text-anchor="middle">0s — {total_duration:.1f}s</text>'
    )
    svg_parts.append("</svg>")
    svg = "".join(svg_parts)
    return svg


def _escape_svg(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


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
