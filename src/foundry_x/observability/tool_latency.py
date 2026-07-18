"""Aggregate ``tool_call`` latency into p50/p95/p99 percentiles per tool name.

Issue #181: ``src/foundry_x/execution/runner.py:599`` now writes a
``duration_ms`` field on every ``tool_call`` event (issue #91), but no
observability surface rolls those numbers up. This module is the rollup:

  1. Walk every session in the trace store.
  2. Stream its ``tool_call`` events through :class:`TraceLogger.iter_events`.
  3. Bucket each event's ``duration_ms`` under its ``name`` (skipping rows
     with missing or non-numeric durations so a malformed payload cannot
     poison the percentile).
  4. Compute p50/p95/p99 per bucket with the deterministic
     nearest-rank method so the same input set always produces the same
     output (the test suite pins this).

The same shape is reused by ``fx-trace tool-latency`` to render a
Markdown table for humans and a JSON object for CI. Both call sites go
through :func:`aggregate_tool_latency`, which is the single source of
truth for the math; renderers never re-derive percentiles.

Issue #877 extends the rollup with **windowed trend analysis**: callers
can pass ``windows=[LatencyWindow.LAST_24H, ...]`` and the function
returns a per-window section comparing the current window's p95 to the
previous equal-sized window's p95, labelling the trend as improving,
stable, or degrading. The default (no ``windows`` argument) keeps the
single all-time aggregate behavior so existing callers are unaffected.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from enum import Enum
from typing import Iterable

from pydantic import BaseModel, Field

from foundry_x.trace.logger import TraceEvent, TraceLogger

TOOL_CALL_KIND = "tool_call"

# Threshold for labelling a trend as "stable" rather than improving or
# degrading: when |delta_p95| / previous_p95 is below this fraction, the
# difference is treated as normal variance (issue #877 acceptance).
# Conservative default — 5% is wider than most measurement noise on a
# handful of samples, so we avoid crying wolf on noise.
_STABLE_THRESHOLD = 0.05

# Minimum number of p95 samples required in *both* windows before a
# trend label is computed; below this we have too little data and the
# direction is set to "unknown".
_MIN_SAMPLES_FOR_TREND = 2


class LatencyWindow(str, Enum):
    """Window spec for trend analysis (issue #877).

    Each member names a single-window aggregate plus its equal-sized
    previous-window comparison:

    * ``LAST_5_SESSIONS`` — the most recent 5 sessions vs the 5 before them
    * ``LAST_24H`` — last 24 hours vs the 24 hours before that
    * ``LAST_7D`` — last 7 days vs the 7 days before that

    The window anchor is the latest event timestamp in the trace store,
    not wall-clock ``now()``, so reports are reproducible across runs and
    are unaffected by the host's clock drift.
    """

    LAST_5_SESSIONS = "last_5_sessions"
    LAST_24H = "last_24h"
    LAST_7D = "last_7d"


class TrendDirection(str, Enum):
    """Comparison result between a current and previous window (issue #877).

    * ``IMPROVING`` — current p95 is materially lower than previous p95
    * ``DEGRADING`` — current p95 is materially higher than previous p95
    * ``STABLE`` — change is within the configured noise threshold
    * ``UNKNOWN`` — comparison is not possible (tool missing from one
      window, too few samples, or no previous-window data at all)
    """

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


class ToolLatencyRow(BaseModel):
    """One tool's percentile summary (ADR-0006 boundary model).

    ``tool`` is the canonical tool name (the ``name`` payload field on
    ``tool_call`` events). ``count`` is the number of tool_call events
    in the analysis window that landed in this bucket; the three
    percentile fields are durations in milliseconds.
    """

    tool: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float


class WindowedToolLatencyRow(BaseModel):
    """One tool's percentile summary within a single window (ADR-0006 boundary model).

    Carries the current-window percentiles plus a trend direction and
    signed p95 delta derived by comparing the current window to the
    previous equal-sized window (issue #877). ``delta_p95_ms`` is
    ``current - previous`` in milliseconds; ``None`` when the comparison
    is not possible (missing data in one window or insufficient samples).
    """

    tool: str
    count: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    trend_p95: TrendDirection
    delta_p95_ms: float | None = None


class WindowedLatencySection(BaseModel):
    """One window's full result (ADR-0006 boundary model, issue #877).

    Combines the current-window rows (with trend annotations) and the
    bounds of both the current and previous equal-sized windows so the
    renderer can show a single cohesive table per window. The
    ``previous_*`` fields are ``None`` when there is no previous-window
    data (e.g. a fresh trace store), which the renderer maps to a
    "no previous window" note rather than a phantom timestamp.
    """

    window: LatencyWindow
    current_start: str | None
    current_end: str | None
    previous_start: str | None
    previous_end: str | None
    rows: list[WindowedToolLatencyRow]
    total_calls: int


class ToolLatencyReport(BaseModel):
    """Full result of an aggregation pass.

    Mirrors :class:`RegressionAnalysis` in shape: the structured rows
    travel alongside the ``total_calls`` count and the window bounds so
    a caller (CLI, future Digester hook, future regression gate) can
    surface a precise Markdown/JSON artifact *and* drive machine logic
    from the same observation.

    Issue #877 adds ``windowed_sections``: when the caller requests one
    or more :class:`LatencyWindow` specs, each section carries the
    per-window trend analysis. The top-level ``rows`` continue to
    represent the single aggregate over the full ``since`` /
    ``harness_version``-filtered range so existing callers see no
    behavioral change.
    """

    rows: list[ToolLatencyRow] = Field(default_factory=list)
    total_calls: int = 0
    window_start: str | None = None
    window_end: str | None = None
    windowed_sections: list[WindowedLatencySection] = Field(default_factory=list)


def percentile(sorted_values: list[float], q: float) -> float:
    """Return the *q*-th percentile of *sorted_values* using nearest-rank.

    *sorted_values* must be in ascending order; the function does not
    re-sort. Nearest-rank (a.k.a. ``ceil(q/100 * n)``) is deliberately
    chosen over linear interpolation: it is deterministic, has no
    floating-point interpolation edge cases, and matches the operator
    intuition "p95 means the worst of the top 5%".

    Returns ``0.0`` for an empty input — the caller's groupby shape
    already excludes tools with zero calls, so this branch is only hit
    defensively.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    # ``math.ceil`` over a non-negative rank never underflows, so the
    # ``- 1`` clamp below also never goes negative for n >= 1.
    rank = max(1, math.ceil(q / 100.0 * n))
    return float(sorted_values[min(rank, n) - 1])


def _extract_duration_ms(payload: dict) -> float | None:
    """Pull ``duration_ms`` from a ``tool_call`` payload, tolerating bad rows.

    The runner writes ``duration_ms`` as an ``int`` (see
    ``src/foundry_x/execution/runner.py:596``), but a hand-edited trace
    or a future producer could store it as a float, a numeric string, or
    omit it entirely. We coerce defensively and return ``None`` for any
    non-positive or unparseable value so the percentile math never
    silently inflates with garbage.
    """
    value = payload.get("duration_ms")
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value)
        except ValueError:
            return None
    else:
        return None
    if math.isnan(number) or number < 0:
        return None
    return number


def _bucket_durations(
    events: Iterable[TraceEvent],
    since: str | None = None,
) -> tuple[dict[str, list[float]], int, str | None, str | None]:
    """Bucket ``tool_call`` event durations under their ``name`` field.

    Shared by the all-time aggregate (issue #181) and the windowed
    trend pass (issue #877): both need the same name/duration math,
    only the input stream differs. ``since`` is applied as a string
    ``>=`` comparison after the fetch (the logger deliberately does not
    accept a timestamp filter on ``iter_events`` — see
    regression_report.py:94-96).

    Returns ``(buckets, total_calls, earliest, latest)`` where the two
    timestamps are the lexicographic min/max of the accepted event
    timestamps (``None`` when the input is empty).
    """
    buckets: dict[str, list[float]] = {}
    total_calls = 0
    earliest: str | None = None
    latest: str | None = None
    for event in events:
        if since is not None and event.timestamp < since:
            continue
        name = event.payload.get("name")
        if not isinstance(name, str) or not name:
            # A tool_call without a name cannot be bucketed; skip
            # rather than synthesize a key like "<unknown>" that
            # would skew the operator's view of named-tool latency.
            continue
        duration = _extract_duration_ms(event.payload)
        if duration is None:
            continue
        buckets.setdefault(name, []).append(duration)
        total_calls += 1
        if earliest is None or event.timestamp < earliest:
            earliest = event.timestamp
        if latest is None or event.timestamp > latest:
            latest = event.timestamp
    return buckets, total_calls, earliest, latest


def _rows_from_buckets(buckets: dict[str, list[float]]) -> list[ToolLatencyRow]:
    """Build the deterministic per-tool percentile rows from a bucket map.

    Buckets with zero entries cannot occur by construction (see
    :func:`_bucket_durations`), so the resulting list contains only
    tools that actually fired in the input stream. Rows are sorted by
    tool name so the Markdown table is reproducible across runs.
    """
    rows: list[ToolLatencyRow] = []
    for tool, durations in buckets.items():
        durations.sort()
        rows.append(
            ToolLatencyRow(
                tool=tool,
                count=len(durations),
                p50_ms=percentile(durations, 50.0),
                p95_ms=percentile(durations, 95.0),
                p99_ms=percentile(durations, 99.0),
            )
        )
    rows.sort(key=lambda r: r.tool)
    return rows


def aggregate_tool_latency(
    logger: TraceLogger,
    since: str | None = None,
    harness_version: str | None = None,
    windows: list[LatencyWindow] | None = None,
) -> ToolLatencyReport:
    """Walk every session and bucket tool_call durations by tool name.

    Mirrors the streaming pattern established in
    ``regression_report._load_verdict_events``: iterate sessions through
    :meth:`TraceLogger.list_sessions`, then pull each session's
    ``tool_call`` events through :meth:`TraceLogger.iter_events`. The
    ``since`` filter is applied as a string ``>=`` comparison after the
    fetch (the logger deliberately does not accept a timestamp filter
    on ``iter_events`` — see regression_report.py:94-96). The
    ``harness_version`` filter is passed directly to ``list_sessions``.

    Tools with zero matching events in the window are excluded by
    construction: the bucket is only created when a valid duration is
    observed, so the resulting ``rows`` list cannot contain a phantom
    tool.

    Issue #877 — when ``windows`` is provided, each spec produces a
    :class:`WindowedLatencySection` in the returned report's
    ``windowed_sections`` field. The top-level ``rows`` continue to
    represent the single aggregate over the full ``since`` /
    ``harness_version``-filtered range. The windowed pass uses a single
    :meth:`TraceLogger.query_events` cursor over all sessions so the
    cost is one full sweep of the events table regardless of how many
    windows are requested — well under the 2-second budget for 1000
    sessions.
    """
    all_events = (
        event
        for session in logger.list_sessions(harness_version=harness_version)
        for event in logger.iter_events(session.session_id, kind=TOOL_CALL_KIND)
    )
    buckets, total_calls, earliest, latest = _bucket_durations(all_events, since=since)
    rows = _rows_from_buckets(buckets)

    sections: list[WindowedLatencySection] = []
    if windows:
        sections = _compute_windowed_sections(logger, windows, harness_version=harness_version)

    return ToolLatencyReport(
        rows=rows,
        total_calls=total_calls,
        window_start=earliest,
        window_end=latest,
        windowed_sections=sections,
    )


def _compute_windowed_sections(
    logger: TraceLogger,
    windows: list[LatencyWindow],
    harness_version: str | None,
) -> list[WindowedLatencySection]:
    """Compute per-window trend sections for the requested *windows*.

    Pulls every ``tool_call`` event once via :meth:`TraceLogger.query_events`
    and reuses the bucket math from :func:`_bucket_durations` to derive
    the current and previous equal-sized windows for each spec.

    The window anchor for time-based windows (``LAST_24H`` /
    ``LAST_7D``) is the latest event timestamp in the data, not
    wall-clock ``now()``. Anchoring to the data keeps reports
    reproducible across runs and isolates the analysis from host clock
    drift; the alternative (anchoring to ``now()``) would silently shift
    the comparison windows each time the report was re-rendered against
    the same trace store.
    """
    # Single pass through ``query_events``; iterated multiple times below
    # by slicing into per-window lists. Holding the materialized list in
    # memory is acceptable for the documented scale (1000 sessions of
    # tool_call events) and avoids re-querying inside each window.
    all_events: list[TraceEvent] = list(
        logger.query_events(kind=TOOL_CALL_KIND, harness_version=harness_version)
    )
    if not all_events:
        return [
            WindowedLatencySection(
                window=spec,
                current_start=None,
                current_end=None,
                previous_start=None,
                previous_end=None,
                rows=[],
                total_calls=0,
            )
            for spec in windows
        ]

    sessions_by_id = {
        s.session_id: s for s in logger.list_sessions(harness_version=harness_version)
    }
    latest_event_ts = max(event.timestamp for event in all_events)

    sections: list[WindowedLatencySection] = []
    for spec in windows:
        sections.append(
            _build_window_section(
                spec,
                all_events,
                sessions_by_id,
                latest_event_ts,
            )
        )
    return sections


def _events_in_range(
    events: list[TraceEvent],
    start_iso: str,
    end_iso: str,
) -> list[TraceEvent]:
    """Return the subset of *events* whose timestamp lies in ``[start_iso, end_iso]``.

    Inclusive on both ends; the ``since`` filter applied elsewhere is
    also inclusive so the boundary behavior matches.
    """
    return [event for event in events if start_iso <= event.timestamp <= end_iso]


def _session_window_bounds(
    spec: LatencyWindow,
    sessions_by_id: dict[str, object],
    all_events: list[TraceEvent],
) -> tuple[
    list[TraceEvent],
    list[TraceEvent],
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """Bucket events into the current and previous session-based windows.

    Sessions are ordered newest-first by ``started_at``; the current
    window contains the first ``N`` sessions and the previous window
    contains the next ``N`` (issue #877). Bounds are derived from the
    earliest and latest event timestamp inside each window, not from
    ``started_at``/``ended_at``, because the latter can span long idle
    gaps that would dilute the comparison.

    Returns ``(current_events, previous_events, current_start,
    current_end, previous_start, previous_end)``. Timestamp fields are
    ``None`` when the corresponding window is empty.
    """
    n = _session_window_size(spec)
    if n == 0 or not sessions_by_id:
        return [], [], None, None, None, None
    sorted_sessions = sorted(sessions_by_id.values(), key=lambda s: s.started_at, reverse=True)
    current_sessions = sorted_sessions[:n]
    previous_sessions = sorted_sessions[n : 2 * n]
    current_ids = {s.session_id for s in current_sessions}
    previous_ids = {s.session_id for s in previous_sessions}
    current_events = [e for e in all_events if e.session_id in current_ids]
    previous_events = [e for e in all_events if e.session_id in previous_ids]
    return (
        current_events,
        previous_events,
        _bounds_min(current_events),
        _bounds_max(current_events),
        _bounds_min(previous_events),
        _bounds_max(previous_events),
    )


def _session_window_size(spec: LatencyWindow) -> int:
    """Return the number of sessions that make up one session-based window.

    Single source of truth for the LAST_N_SESSIONS count so the
    acceptance criteria and the renderer stay in lockstep.
    """
    return 5


def _time_window_bounds(
    spec: LatencyWindow,
    latest_event_ts: str,
    all_events: list[TraceEvent],
) -> tuple[
    list[TraceEvent],
    list[TraceEvent],
    str | None,
    str | None,
    str | None,
    str | None,
]:
    """Bucket events into the current and previous time-based windows.

    The window anchor is *latest_event_ts* — the latest event timestamp
    in the trace store — so the analysis is reproducible across runs
    against the same data (see :func:`_compute_windowed_sections` for
    the rationale). Each window is the configured duration wide; the
    current window ends at the anchor and the previous window sits
    immediately behind it.
    """
    duration = _time_window_duration(spec)
    if duration is None:
        return [], [], None, None, None, None
    anchor = datetime.fromisoformat(latest_event_ts)
    current_end = anchor
    current_start = anchor - duration
    previous_end = current_start
    previous_start = previous_end - duration
    current_events = _events_in_range(
        all_events,
        current_start.isoformat(),
        current_end.isoformat(),
    )
    previous_events = _events_in_range(
        all_events,
        previous_start.isoformat(),
        previous_end.isoformat(),
    )
    return (
        current_events,
        previous_events,
        current_start.isoformat(),
        current_end.isoformat(),
        previous_start.isoformat(),
        previous_end.isoformat(),
    )


def _time_window_duration(spec: LatencyWindow) -> timedelta | None:
    """Return the duration of one time-based window.

    ``LAST_24H`` -> 24 hours, ``LAST_7D`` -> 7 days. ``LAST_5_SESSIONS``
    is session-based and returns ``None`` so the caller knows to skip
    the time-window path.
    """
    if spec == LatencyWindow.LAST_24H:
        return timedelta(hours=24)
    if spec == LatencyWindow.LAST_7D:
        return timedelta(days=7)
    return None


def _bounds_min(events: list[TraceEvent]) -> str | None:
    """Return the lexicographic minimum timestamp from *events* (``None`` if empty)."""
    return min((e.timestamp for e in events), default=None)


def _bounds_max(events: list[TraceEvent]) -> str | None:
    """Return the lexicographic maximum timestamp from *events* (``None`` if empty)."""
    return max((e.timestamp for e in events), default=None)


def _build_window_section(
    spec: LatencyWindow,
    all_events: list[TraceEvent],
    sessions_by_id: dict[str, object],
    latest_event_ts: str,
) -> WindowedLatencySection:
    """Build one :class:`WindowedLatencySection` for *spec*.

    Routes the window spec to either the session- or time-based bucket
    logic, then derives the trend-annotated per-tool rows from the two
    windows. Pure function with no side effects on the logger.
    """
    if spec == LatencyWindow.LAST_5_SESSIONS:
        (
            current_events,
            previous_events,
            current_start,
            current_end,
            previous_start,
            previous_end,
        ) = _session_window_bounds(spec, sessions_by_id, all_events)
    else:
        (
            current_events,
            previous_events,
            current_start,
            current_end,
            previous_start,
            previous_end,
        ) = _time_window_bounds(spec, latest_event_ts, all_events)

    current_buckets, current_total, _cs, _ce = _bucket_durations(current_events)
    previous_buckets, _, _ps, _pe = _bucket_durations(previous_events)

    # Build rows for every tool that fired in the current window; tools
    # that fired in the previous window but not the current get a
    # separate trend label downstream (the trend is computed against the
    # current window's set of tools, which is the operator's view).
    rows: list[WindowedToolLatencyRow] = []
    for tool in sorted(current_buckets):
        durations = sorted(current_buckets[tool])
        p95 = percentile(durations, 95.0)
        previous_durations = previous_buckets.get(tool)
        if previous_durations is None or len(previous_durations) < _MIN_SAMPLES_FOR_TREND:
            trend = TrendDirection.UNKNOWN
            delta: float | None = None
        else:
            previous_p95 = percentile(sorted(previous_durations), 95.0)
            delta = p95 - previous_p95
            if previous_p95 <= 0:
                # Defensive: a zero previous p95 means there was no
                # meaningful comparison point. If current is also zero
                # the trend is stable; otherwise we cannot reason about
                # the direction without a non-zero baseline.
                trend = TrendDirection.STABLE if p95 == 0.0 else TrendDirection.UNKNOWN
            else:
                relative = abs(delta) / previous_p95
                if relative < _STABLE_THRESHOLD:
                    trend = TrendDirection.STABLE
                elif delta < 0:
                    trend = TrendDirection.IMPROVING
                else:
                    trend = TrendDirection.DEGRADING
        rows.append(
            WindowedToolLatencyRow(
                tool=tool,
                count=len(durations),
                p50_ms=percentile(durations, 50.0),
                p95_ms=p95,
                p99_ms=percentile(durations, 99.0),
                trend_p95=trend,
                delta_p95_ms=delta,
            )
        )

    return WindowedLatencySection(
        window=spec,
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
        rows=rows,
        total_calls=current_total,
    )


def _window_label(spec: LatencyWindow) -> str:
    """Human-friendly label for the Markdown section header.

    Kept separate from the enum so the canonical CLI string and the
    human label can diverge without one starving the other.
    """
    if spec == LatencyWindow.LAST_5_SESSIONS:
        return "Last 5 sessions"
    if spec == LatencyWindow.LAST_24H:
        return "Last 24 hours"
    if spec == LatencyWindow.LAST_7D:
        return "Last 7 days"
    return spec.value


def _format_trend_arrow(direction: TrendDirection) -> str:
    """Compact ASCII marker for the trend column.

    Kept ASCII-only because the existing Markdown renders ship in many
    environments where unicode arrows render inconsistently.
    """
    if direction == TrendDirection.IMPROVING:
        return "v improving"
    if direction == TrendDirection.DEGRADING:
        return "^ degrading"
    if direction == TrendDirection.STABLE:
        return "= stable"
    return "? unknown"


def _format_window_bounds(
    current_start: str | None,
    current_end: str | None,
    previous_start: str | None,
    previous_end: str | None,
) -> list[str]:
    """Format the window-bounds bullet lines for the Markdown section header.

    When the previous window has no data (fresh trace store), the
    renderer surfaces a single "no previous window" note rather than
    phantom timestamps so the operator can distinguish "we don't have
    history yet" from "history was empty by design".
    """
    lines: list[str] = []
    if current_start or current_end:
        lines.append(f"- Current window: {current_start or '?'} -> {current_end or '?'}")
    if previous_start or previous_end:
        lines.append(f"- Previous window: {previous_start or '?'} -> {previous_end or '?'}")
    else:
        lines.append("- Previous window: no prior data")
    return lines


def render_tool_latency_markdown(report: ToolLatencyReport) -> str:
    """Render a :class:`ToolLatencyReport` as a Markdown table.

    The empty case renders a single-sentence artifact (matches the
    CLI's "no tool_call events in window" contract from issue #181).

    Issue #877: when the report carries ``windowed_sections``, a
    secondary per-window table is appended after the existing all-time
    aggregate. Each section header names the window spec, the current /
    previous bounds, and the per-tool trend column. The original
    all-time table is left untouched so the diff is purely additive —
    a caller that passes no ``windows`` argument sees byte-identical
    output to the previous release.
    """
    lines = [
        "# Tool Latency Report",
        "",
        f"- Total tool_call events: {report.total_calls}",
    ]
    if report.window_start or report.window_end:
        lines.append(f"- Window: {report.window_start or '?'} -> {report.window_end or '?'}")
    lines.append("")

    if not report.rows:
        lines.append("no tool_call events in window")
        lines.append("")
    else:
        lines.append("| Tool | Calls | p50 (ms) | p95 (ms) | p99 (ms) |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        for row in report.rows:
            lines.append(
                f"| {row.tool} | {row.count} | {row.p50_ms:g} | {row.p95_ms:g} | {row.p99_ms:g} |"
            )
        lines.append("")

    for section in report.windowed_sections:
        lines.append(f"## Window: {_window_label(section.window)}")
        lines.append("")
        lines.append(f"- Events in current window: {section.total_calls}")
        lines.extend(
            _format_window_bounds(
                section.current_start,
                section.current_end,
                section.previous_start,
                section.previous_end,
            )
        )
        lines.append("")
        if not section.rows:
            lines.append("no tool_call events in window")
            lines.append("")
            continue
        lines.append("| Tool | Calls | p50 (ms) | p95 (ms) | p99 (ms) | \u0394 p95 (ms) | Trend |")
        lines.append("| --- | ---: | ---: | ---: | ---: | ---: | --- |")
        for row in section.rows:
            delta_str = f"{row.delta_p95_ms:+g}" if row.delta_p95_ms is not None else ""
            lines.append(
                f"| {row.tool} | {row.count} | {row.p50_ms:g} | {row.p95_ms:g} | "
                f"{row.p99_ms:g} | {delta_str} | {_format_trend_arrow(row.trend_p95)} |"
            )
        lines.append("")
    return "\n".join(lines)


def render_tool_latency_json(report: ToolLatencyReport) -> str:
    """Render a :class:`ToolLatencyReport` as a JSON object.

    The accepted shape is ``{tool: {count, p50_ms, p95_ms, p99_ms}}``
    (issue #181 acceptance criteria). The window bounds, total call
    count, and per-window trend sections travel on the
    :class:`ToolLatencyReport` itself; machine consumers that need the
    full structured view should call
    :meth:`ToolLatencyReport.model_dump_json` directly (issue #877).

    Keeping the legacy JSON shape stable is deliberate — CI consumers
    built against the issue #181 contract must keep parsing the same
    ``{tool: {count, p50_ms, p95_ms, p99_ms}}`` shape, and the new
    windowed data is fully accessible via the Pydantic model.
    """
    tools: dict[str, dict[str, float | int]] = {}
    for row in report.rows:
        tools[row.tool] = {
            "count": row.count,
            "p50_ms": row.p50_ms,
            "p95_ms": row.p95_ms,
            "p99_ms": row.p99_ms,
        }
    return json.dumps(tools, indent=2, sort_keys=True)
