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
"""

from __future__ import annotations

import json
import math

from pydantic import BaseModel, Field

from foundry_x.trace.logger import TraceLogger

TOOL_CALL_KIND = "tool_call"


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


class ToolLatencyReport(BaseModel):
    """Full result of an aggregation pass.

    Mirrors :class:`RegressionAnalysis` in shape: the structured rows
    travel alongside the ``total_calls`` count and the window bounds so
    a caller (CLI, future Digester hook, future regression gate) can
    surface a precise Markdown/JSON artifact *and* drive machine logic
    from the same observation.
    """

    rows: list[ToolLatencyRow] = Field(default_factory=list)
    total_calls: int = 0
    window_start: str | None = None
    window_end: str | None = None


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


def aggregate_tool_latency(
    logger: TraceLogger,
    since: str | None = None,
) -> ToolLatencyReport:
    """Walk every session and bucket tool_call durations by tool name.

    Mirrors the streaming pattern established in
    ``regression_report._load_verdict_events``: iterate sessions through
    :meth:`TraceLogger.list_sessions`, then pull each session's
    ``tool_call`` events through :meth:`TraceLogger.iter_events`. The
    ``since`` filter is applied as a string ``>=`` comparison after the
    fetch (the logger deliberately does not accept a timestamp filter
    on ``iter_events`` — see regression_report.py:94-96).

    Tools with zero matching events in the window are excluded by
    construction: the bucket is only created when a valid duration is
    observed, so the resulting ``rows`` list cannot contain a phantom
    tool.
    """
    buckets: dict[str, list[float]] = {}
    earliest: str | None = None
    latest: str | None = None
    total_calls = 0

    for session in logger.list_sessions():
        for event in logger.iter_events(session.session_id, kind=TOOL_CALL_KIND):
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

    # Stable ordering: by tool name, so the Markdown table is reproducible
    # across runs (the underlying trace store has no guaranteed event
    # order across sessions).
    rows.sort(key=lambda r: r.tool)

    return ToolLatencyReport(
        rows=rows,
        total_calls=total_calls,
        window_start=earliest,
        window_end=latest,
    )


def render_tool_latency_markdown(report: ToolLatencyReport) -> str:
    """Render a :class:`ToolLatencyReport` as a Markdown table.

    The empty case renders a single-sentence artifact (matches the
    CLI's "no tool_call events in window" contract from issue #181).
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
        return "\n".join(lines)

    lines.append("| Tool | Calls | p50 (ms) | p95 (ms) | p99 (ms) |")
    lines.append("| --- | ---: | ---: | ---: | ---: |")
    for row in report.rows:
        lines.append(
            f"| {row.tool} | {row.count} | {row.p50_ms:g} | {row.p95_ms:g} | {row.p99_ms:g} |"
        )
    lines.append("")
    return "\n".join(lines)


def render_tool_latency_json(report: ToolLatencyReport) -> str:
    """Render a :class:`ToolLatencyReport` as a JSON object.

    The accepted shape is ``{tool: {count, p50_ms, p95_ms, p99_ms}}``
    (issue #181 acceptance criteria). The window bounds and total call
    count travel on the :class:`ToolLatencyReport` itself for Markdown
    callers; CI consumers that need the window can read the artifact
    alongside this dict.
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
