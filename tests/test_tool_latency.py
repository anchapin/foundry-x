"""Tests for the tool-call latency percentile aggregator (issue #181).

The acceptance criteria require:

  * ``tests/test_tool_latency.py`` asserts deterministic percentile
    math on a planted set.
  * The CLI subcommand prints a Markdown table, emits a ``{tool: ...}``
    JSON shape on ``--format json``, and exits 0 with the empty-store
    message when no ``tool_call`` events exist.

The tests below pin each of those properties plus a handful of edge
cases the field math needs to handle (single sample, missing
``duration_ms``, unnamed tool, ``since`` filter).

Issue #877 adds windowed trend analysis: ``aggregate_tool_latency``
accepts a ``windows`` argument, the renderer emits a per-window section
with a trend column, and the CLI surfaces the same through
``--window``. The trend tests below plant deterministic timestamp
spreads so the comparison windows are unambiguous.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from foundry_x.observability.cli import main as cli_main
from foundry_x.observability.tool_latency import (
    LatencyWindow,
    ToolLatencyReport,
    ToolLatencyRow,
    TrendDirection,
    WindowedLatencySection,
    WindowedToolLatencyRow,
    aggregate_tool_latency,
    percentile,
    render_tool_latency_json,
    render_tool_latency_markdown,
)
from foundry_x.trace.logger import TraceLogger

_BASE = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def _ts(offset_seconds: float) -> str:
    return (_BASE + timedelta(seconds=offset_seconds)).isoformat()


def _seed_tool_calls(
    db_path,
    events: list[tuple[str, str, dict]],
    harness_version: str = "0.1.0",
) -> None:
    """Plant a single session with the supplied ``tool_call`` events.

    *events* is a list of ``(offset_seconds, tool_name, payload)``
    tuples. The logger stamps each event with wall-clock time at the
    moment of ``record()`` (see ``TraceLogger._now``), so the supplied
    *offset_seconds* is informational only — the actual stored
    timestamp depends on real time. Tests that need to drive a
    ``--since`` cutoff should fetch a real timestamp via
    :func:`_event_timestamps` rather than reconstructing one from the
    offsets.
    """
    logger = TraceLogger(db_path)
    with logger.session(harness_version=harness_version) as sid:
        for _offset, _name, payload in events:
            logger.record(sid, "tool_call", payload)


def _event_timestamps(db_path) -> list[str]:
    """Return the stored timestamps for every ``tool_call`` row, in order."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT timestamp FROM events WHERE kind = 'tool_call' ORDER BY timestamp"
        ).fetchall()
    return [row[0] for row in rows]


def _split_timestamp(earlier: str, later: str) -> str:
    """Return an ISO-8601 timestamp that sorts strictly between *earlier* and *later*.

    Used by tests that need a ``--since`` cutoff which is guaranteed to
    drop exactly the older event in a planted pair, without depending
    on the wall-clock time at which the trace was written.
    """
    lo = datetime.fromisoformat(earlier)
    hi = datetime.fromisoformat(later)
    delta = (hi - lo) / 2
    return (lo + delta).isoformat()


# ---------------------------------------------------------------------------
# Percentile math
# ---------------------------------------------------------------------------


def test_percentile_deterministic_planted_set():
    """Pinned values on a 1..100 ramp with nearest-rank semantics."""
    values = list(range(1, 101))
    # p50 = 50 (rank 50 of 100), p95 = 95, p99 = 99.
    assert percentile(values, 50.0) == 50.0
    assert percentile(values, 95.0) == 95.0
    assert percentile(values, 99.0) == 99.0


def test_percentile_single_value_collapses_to_that_value():
    """n=1 must yield the only sample for every q (rank clamps to 1)."""
    assert percentile([42.0], 50.0) == 42.0
    assert percentile([42.0], 95.0) == 42.0
    assert percentile([42.0], 99.0) == 42.0


def test_percentile_nearest_rank_chooses_ceiling_not_average():
    """n=2 with q=50 picks the lower sample (nearest-rank, not interpolation)."""
    assert percentile([10.0, 20.0], 50.0) == 10.0


def test_percentile_odd_count_uses_ceiling():
    """n=99 ramp: rank=ceil(99*0.5)=50 → index 49 → value 50."""
    values = list(range(1, 100))
    assert percentile(values, 50.0) == 50.0


def test_percentile_empty_returns_zero():
    """Defensive branch — caller filters empty buckets already."""
    assert percentile([], 50.0) == 0.0


# ---------------------------------------------------------------------------
# Aggregate function
# ---------------------------------------------------------------------------


def _planted_call_set():
    """Two tools, planted with deterministic percentile targets.

    read_file: [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
      → p50 = 50 (rank 5/10), p95 = 100 (rank 10/10), p99 = 100
    write_file: [5, 5, 5, 5, 5, 500]
      → p50 = 5 (rank 3/6), p95 = 500 (rank 6/6), p99 = 500

    Hand-checked against nearest-rank: ``ceil(q/100 * n)`` then index.
    """
    reads = list(range(10, 101, 10))
    writes = [5, 5, 5, 5, 5, 500]
    return reads, writes


def test_aggregate_buckets_per_tool_and_computes_percentiles(tmp_path):
    db = tmp_path / "traces.db"
    reads, writes = _planted_call_set()
    events: list[tuple[str, str, dict]] = []
    for i, ms in enumerate(reads):
        events.append((i * 0.01, "read_file", {"name": "read_file", "duration_ms": ms}))
    for i, ms in enumerate(writes):
        events.append((1.0 + i * 0.01, "write_file", {"name": "write_file", "duration_ms": ms}))
    _seed_tool_calls(db, events)

    report = aggregate_tool_latency(TraceLogger(db))

    by_tool = {row.tool: row for row in report.rows}
    assert set(by_tool) == {"read_file", "write_file"}
    assert by_tool["read_file"].count == len(reads)
    assert by_tool["read_file"].p50_ms == 50.0
    assert by_tool["read_file"].p95_ms == 100.0
    assert by_tool["read_file"].p99_ms == 100.0
    assert by_tool["write_file"].count == len(writes)
    assert by_tool["write_file"].p50_ms == 5.0
    assert by_tool["write_file"].p95_ms == 500.0
    assert by_tool["write_file"].p99_ms == 500.0
    assert report.total_calls == len(reads) + len(writes)


def test_aggregate_excludes_tools_with_zero_calls_in_window(tmp_path):
    """A tool_call event with missing/unparseable duration_ms must not appear."""
    db = tmp_path / "traces.db"
    _seed_tool_calls(
        db,
        [
            (0.0, "read_file", {"name": "read_file", "duration_ms": 12}),
            (0.1, "missing_duration", {"name": "missing_duration"}),
            (0.2, "negative_duration", {"name": "negative_duration", "duration_ms": -5}),
            (0.3, "string_duration", {"name": "string_duration", "duration_ms": "not a number"}),
            (0.4, "no_name", {"duration_ms": 50}),
        ],
    )

    report = aggregate_tool_latency(TraceLogger(db))

    tools = {row.tool for row in report.rows}
    assert tools == {"read_file"}
    assert report.total_calls == 1


def test_aggregate_respects_since_filter(tmp_path):
    db = tmp_path / "traces.db"
    _seed_tool_calls(
        db,
        [
            (0.0, "read_file", {"name": "read_file", "duration_ms": 10}),
            (5.0, "read_file", {"name": "read_file", "duration_ms": 20}),
            (10.0, "read_file", {"name": "read_file", "duration_ms": 30}),
        ],
    )

    timestamps = _event_timestamps(db)
    assert len(timestamps) == 3
    # Cutoff sits between the first and the second stored timestamp so
    # exactly one event is dropped. Using the real stored timestamps is
    # necessary because the logger stamps events with wall-clock time at
    # record-time rather than honoring the supplied offsets.
    cutoff = _split_timestamp(timestamps[0], timestamps[1])

    report = aggregate_tool_latency(TraceLogger(db), since=cutoff)

    assert len(report.rows) == 1
    assert report.rows[0].count == 2
    # Nearest-rank on [20, 30]: p50 rank 1/2 → index 0 → 20.
    assert report.rows[0].p50_ms == 20.0
    assert report.rows[0].p95_ms == 30.0
    assert report.rows[0].p99_ms == 30.0


def test_aggregate_empty_store_returns_empty_report(tmp_path):
    db = tmp_path / "traces.db"
    TraceLogger(db)  # create schema, no events

    report = aggregate_tool_latency(TraceLogger(db))

    assert report == ToolLatencyReport(rows=[], total_calls=0, window_start=None, window_end=None)


def test_aggregate_aggregates_across_sessions(tmp_path):
    """Two sessions, same tool name — bucket merges across the session boundary."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as sid_a:
        logger.record(sid_a, "tool_call", {"name": "read_file", "duration_ms": 10})
        logger.record(sid_a, "tool_call", {"name": "read_file", "duration_ms": 30})
    with logger.session(harness_version="0.2.0") as sid_b:
        logger.record(sid_b, "tool_call", {"name": "read_file", "duration_ms": 20})

    report = aggregate_tool_latency(TraceLogger(db))

    assert len(report.rows) == 1
    row = report.rows[0]
    assert row.tool == "read_file"
    assert row.count == 3
    # Nearest-rank on [10, 20, 30]: p50 rank 2/3 = index 1 → 20.
    assert row.p50_ms == 20.0
    assert row.p99_ms == 30.0


def test_aggregate_respects_harness_version_filter(tmp_path):
    """Only sessions matching --harness-version are included in the aggregation."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1.0.0") as sid_a:
        logger.record(sid_a, "tool_call", {"name": "read_file", "duration_ms": 10})
        logger.record(sid_a, "tool_call", {"name": "read_file", "duration_ms": 30})
    with logger.session(harness_version="v2.0.0") as sid_b:
        logger.record(sid_b, "tool_call", {"name": "read_file", "duration_ms": 20})

    report_v1 = aggregate_tool_latency(TraceLogger(db), harness_version="v1.0.0")
    assert len(report_v1.rows) == 1
    assert report_v1.rows[0].count == 2
    # Nearest-rank on [10, 30]: p50 rank ceil(50/100*2)=1 → index 0 → 10.
    assert report_v1.rows[0].p50_ms == 10.0

    report_v2 = aggregate_tool_latency(TraceLogger(db), harness_version="v2.0.0")
    assert len(report_v2.rows) == 1
    assert report_v2.rows[0].count == 1
    assert report_v2.rows[0].p50_ms == 20.0

    report_all = aggregate_tool_latency(TraceLogger(db))
    assert report_all.rows[0].count == 3


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------


def test_render_markdown_contains_table_headers_and_rows():
    report = ToolLatencyReport(
        rows=[
            ToolLatencyRow(tool="read_file", count=10, p50_ms=50.0, p95_ms=100.0, p99_ms=100.0),
            ToolLatencyRow(tool="write_file", count=6, p50_ms=5.0, p95_ms=500.0, p99_ms=500.0),
        ],
        total_calls=16,
        window_start=_ts(0.0),
        window_end=_ts(1.0),
    )

    rendered = render_tool_latency_markdown(report)

    assert "# Tool Latency Report" in rendered
    assert "| Tool | Calls | p50 (ms) | p95 (ms) | p99 (ms) |" in rendered
    assert "| read_file | 10 | 50 | 100 | 100 |" in rendered
    assert "| write_file | 6 | 5 | 500 | 500 |" in rendered
    assert "Total tool_call events: 16" in rendered


def test_render_markdown_empty_store_message():
    report = ToolLatencyReport()
    rendered = render_tool_latency_markdown(report)
    assert "no tool_call events in window" in rendered
    assert "| Tool |" not in rendered


def test_render_json_matches_issue_spec():
    """Issue #181 acceptance: ``{tool: {count, p50_ms, p95_ms, p99_ms}}``."""
    report = ToolLatencyReport(
        rows=[
            ToolLatencyRow(tool="read_file", count=10, p50_ms=50.0, p95_ms=100.0, p99_ms=100.0),
        ],
        total_calls=10,
        window_start=_ts(0.0),
        window_end=_ts(1.0),
    )

    payload = json.loads(render_tool_latency_json(report))

    assert set(payload) == {"read_file"}
    assert payload["read_file"] == {
        "count": 10,
        "p50_ms": 50.0,
        "p95_ms": 100.0,
        "p99_ms": 100.0,
    }


def test_render_json_empty_store_is_empty_object():
    payload = json.loads(render_tool_latency_json(ToolLatencyReport()))
    assert payload == {}


# ---------------------------------------------------------------------------
# CLI surface (issue #181 acceptance)
# ---------------------------------------------------------------------------


def test_cli_tool_latency_markdown_prints_table(tmp_path, capsys):
    db = tmp_path / "traces.db"
    reads, writes = _planted_call_set()
    events: list[tuple[str, str, dict]] = []
    for i, ms in enumerate(reads):
        events.append((i * 0.01, "read_file", {"name": "read_file", "duration_ms": ms}))
    for i, ms in enumerate(writes):
        events.append((1.0 + i * 0.01, "write_file", {"name": "write_file", "duration_ms": ms}))
    _seed_tool_calls(db, events)

    rc = cli_main(["tool-latency", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# Tool Latency Report" in out
    assert "| read_file |" in out
    assert "| write_file |" in out
    # Planted p50 values must show up.
    assert " 50 " in out
    assert " 500 " in out


def test_cli_tool_latency_json_emits_spec_shape(tmp_path, capsys):
    db = tmp_path / "traces.db"
    _seed_tool_calls(
        db,
        [
            (0.0, "read_file", {"name": "read_file", "duration_ms": 10}),
            (0.1, "read_file", {"name": "read_file", "duration_ms": 20}),
        ],
    )

    rc = cli_main(["tool-latency", "--db", str(db), "--format", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert "read_file" in payload
    assert set(payload["read_file"]) == {"count", "p50_ms", "p95_ms", "p99_ms"}
    assert payload["read_file"]["count"] == 2


def test_cli_tool_latency_empty_store_exits_zero_with_message(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = cli_main(["tool-latency", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "no tool_call events in window" in out


def test_cli_tool_latency_out_writes_file(tmp_path):
    db = tmp_path / "traces.db"
    _seed_tool_calls(
        db,
        [
            (0.0, "read_file", {"name": "read_file", "duration_ms": 10}),
            (0.1, "read_file", {"name": "read_file", "duration_ms": 20}),
        ],
    )
    out_file = tmp_path / "latency.md"

    rc = cli_main(["tool-latency", "--db", str(db), "--out", str(out_file)])

    assert rc == 0
    written = out_file.read_text(encoding="utf-8")
    assert "# Tool Latency Report" in written
    assert "read_file" in written


def test_cli_tool_latency_since_filter_drops_older_events(tmp_path, capsys):
    db = tmp_path / "traces.db"
    _seed_tool_calls(
        db,
        [
            (0.0, "read_file", {"name": "read_file", "duration_ms": 10}),
            (5.0, "read_file", {"name": "read_file", "duration_ms": 20}),
            (10.0, "read_file", {"name": "read_file", "duration_ms": 30}),
        ],
    )

    timestamps = _event_timestamps(db)
    cutoff = _split_timestamp(timestamps[0], timestamps[1])
    rc = cli_main(["tool-latency", "--db", str(db), "--since", cutoff])

    assert rc == 0
    out = capsys.readouterr().out
    # Only the two later rows survive → count column shows 2.
    assert "| read_file | 2 |" in out


def test_cli_tool_latency_harness_version_filter(tmp_path, capsys):
    """--harness-version only includes sessions with that version."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1.0.0") as sid_v1:
        logger.record(sid_v1, "tool_call", {"name": "read_file", "duration_ms": 10})
        logger.record(sid_v1, "tool_call", {"name": "read_file", "duration_ms": 30})
    with logger.session(harness_version="v2.0.0") as sid_v2:
        logger.record(sid_v2, "tool_call", {"name": "read_file", "duration_ms": 20})

    rc = cli_main(["tool-latency", "--db", str(db), "--harness-version", "v1.0.0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| read_file | 2 |" in out

    rc = cli_main(["tool-latency", "--db", str(db), "--harness-version", "v2.0.0"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "| read_file | 1 |" in out


# ---------------------------------------------------------------------------
# Windowed trend analysis (issue #877)
# ---------------------------------------------------------------------------


def _plant_windowed_sessions(
    db_path,
    session_durations_ms: list[list[int]],
    tool_name: str = "read_file",
    harness_version: str = "0.1.0",
) -> list[str]:
    """Plant *N* sessions each carrying a list of ``duration_ms`` values.

    Returns the session IDs in insertion order (which is also the
    chronological order the logger writes). Tests then drive the
    windowed aggregation against this deterministic ordering by feeding
    the rows through ``list_sessions`` rather than the wall-clock
    timestamps, because the latter is unaffected by the planted offsets
    (see :func:`_seed_tool_calls` docstring).
    """
    logger = TraceLogger(db_path)
    session_ids: list[str] = []
    for durations in session_durations_ms:
        with logger.session(harness_version=harness_version) as sid:
            for ms in durations:
                logger.record(sid, "tool_call", {"name": tool_name, "duration_ms": ms})
            session_ids.append(sid)
    return session_ids


def _plant_windowed_with_custom_timestamps(
    db_path,
    per_event: list[tuple[int, int, int]],
    harness_version: str = "0.1.0",
) -> None:
    """Plant ``tool_call`` events with explicit timestamp offsets.

    *per_event* is a list of ``(session_started_offset_seconds,
    duration_ms, event_timestamp_offset_seconds)`` tuples. Sessions are
    grouped by ``session_started_offset`` (each unique offset becomes
    one session) so the session-based window logic sees the right
    partition. Sessions and events are inserted with timestamps derived
    from a fixed base (``2026-01-01T00:00:00Z``) so the time-window
    math can be driven against deterministic positions on the timeline
    without sleeping.

    The logger's public surface stamps events with wall-clock time at
    ``record()`` (see :func:`_seed_tool_calls` docstring), so direct
    ``INSERT`` is the only way to plant events at known timestamps. The
    schema is created by instantiating :class:`TraceLogger` once before
    the inserts run.
    """
    conn = sqlite3.connect(db_path)
    base = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    session_id_for_offset: dict[int, str] = {}
    for event_idx, (session_offset, duration_ms, event_offset) in enumerate(per_event):
        session_id = session_id_for_offset.setdefault(
            session_offset,
            f"00000000-0000-0000-0000-{len(session_id_for_offset):012d}",
        )
        started_at = (base + timedelta(seconds=session_offset)).isoformat()
        event_ts = (base + timedelta(seconds=event_offset)).isoformat()
        conn.execute(
            "INSERT OR REPLACE INTO sessions "
            "(session_id, started_at, harness_version, model_id, metadata, ended_at) "
            "VALUES (?, ?, ?, NULL, ?, NULL)",
            (session_id, started_at, harness_version, "{}"),
        )
        conn.execute(
            "INSERT INTO events (event_id, session_id, timestamp, kind, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                f"evt-{event_idx}",
                session_id,
                event_ts,
                "tool_call",
                json.dumps({"name": "read_file", "duration_ms": duration_ms}),
            ),
        )
    conn.commit()
    conn.close()


def test_windowed_last_5_sessions_trend_degrading(tmp_path):
    """Last 5 sessions got slower than the previous 5 → DEGRADING."""
    db = tmp_path / "traces.db"
    # 10 sessions: first 5 fast (10ms each), last 5 slow (200ms each).
    fast = [[10] for _ in range(5)]
    slow = [[200] for _ in range(5)]
    _plant_windowed_sessions(db, fast + slow)

    report = aggregate_tool_latency(TraceLogger(db), windows=[LatencyWindow.LAST_5_SESSIONS])

    assert len(report.windowed_sections) == 1
    section = report.windowed_sections[0]
    assert section.window is LatencyWindow.LAST_5_SESSIONS
    assert section.total_calls == 5
    assert len(section.rows) == 1
    row = section.rows[0]
    assert row.tool == "read_file"
    assert row.p95_ms == 200.0
    assert row.delta_p95_ms == 190.0  # 200 - 10
    assert row.trend_p95 is TrendDirection.DEGRADING


def test_windowed_last_5_sessions_trend_improving(tmp_path):
    """Last 5 sessions got faster than the previous 5 → IMPROVING."""
    db = tmp_path / "traces.db"
    slow = [[200] for _ in range(5)]
    fast = [[10] for _ in range(5)]
    _plant_windowed_sessions(db, slow + fast)

    report = aggregate_tool_latency(TraceLogger(db), windows=[LatencyWindow.LAST_5_SESSIONS])

    section = report.windowed_sections[0]
    assert section.rows[0].trend_p95 is TrendDirection.IMPROVING
    assert section.rows[0].delta_p95_ms == -190.0  # 10 - 200


def test_windowed_trend_stable_when_within_5_percent(tmp_path):
    """|delta|/previous ≤ 5% → STABLE (issue #877 noise threshold)."""
    db = tmp_path / "traces.db"
    # previous=100ms, current=103ms → 3% change → stable.
    previous = [[100] for _ in range(5)]
    current = [[103] for _ in range(5)]
    _plant_windowed_sessions(db, previous + current)

    report = aggregate_tool_latency(TraceLogger(db), windows=[LatencyWindow.LAST_5_SESSIONS])

    section = report.windowed_sections[0]
    assert section.rows[0].trend_p95 is TrendDirection.STABLE
    assert section.rows[0].delta_p95_ms == 3.0


def test_windowed_trend_unknown_when_previous_window_empty(tmp_path):
    """Fewer than 10 sessions → previous window has no data → UNKNOWN."""
    db = tmp_path / "traces.db"
    # Only 3 sessions: the "previous 5" window is empty.
    _plant_windowed_sessions(db, [[50], [60], [70]])

    report = aggregate_tool_latency(TraceLogger(db), windows=[LatencyWindow.LAST_5_SESSIONS])

    section = report.windowed_sections[0]
    assert section.rows[0].trend_p95 is TrendDirection.UNKNOWN
    assert section.rows[0].delta_p95_ms is None
    # The current window still surfaces events; only the trend is unknown.
    assert section.rows[0].count == 3


def test_windowed_last_24h_time_based_against_marker(tmp_path):
    """Time-based window with hand-stamped events at known offsets.

    The test plants:
      * 3 events in the *previous* 24h window at offsets 26h–27h ago
      * 3 events in the *current*  24h window at offsets 0–1h ago

    Both windows have p95 = 200ms for the older set and 50ms for the
    newer set, so the trend must read IMPROVING.
    """
    db = tmp_path / "traces.db"
    TraceLogger(db)  # ensure schema exists
    # Build offsets: each tuple is (session_started_offset, duration_ms, event_offset).
    # Sessions started 27h and 1h ago respectively.
    per_event = [
        # session 0 (older): events at -27h, -26.5h, -26h
        (-27 * 3600, 200, -27 * 3600),
        (-27 * 3600, 200, int(-26.5 * 3600)),
        (-27 * 3600, 200, -26 * 3600),
        # session 1 (newer): events at -1h, -0.5h, 0
        (-3600, 50, -3600),
        (-3600, 50, int(-0.5 * 3600)),
        (-3600, 50, 0),
    ]
    _plant_windowed_with_custom_timestamps(db, per_event)

    report = aggregate_tool_latency(TraceLogger(db), windows=[LatencyWindow.LAST_24H])

    section = report.windowed_sections[0]
    assert section.window is LatencyWindow.LAST_24H
    assert section.total_calls == 3  # only current-window events count
    assert len(section.rows) == 1
    row = section.rows[0]
    assert row.p95_ms == 50.0
    assert row.delta_p95_ms == -150.0  # 50 - 200
    assert row.trend_p95 is TrendDirection.IMPROVING


def test_windowed_multiple_windows_in_single_call(tmp_path):
    """Requesting two windows returns one section per spec.

    The fixture plants 10 sessions with a deliberate gap between the
    two groups so both windows distinguish the trend:

    * 5 slow sessions at -46h..-26h (200ms each)
    * 5 fast sessions at -4h..0h (50ms each)

    For ``LAST_5_SESSIONS`` the current window is the 5 fast sessions
    and the previous window is the 5 slow ones — IMPROVING.

    For ``LAST_24H`` (anchored at the latest event timestamp ``0h``):
    the current window covers [-24h, 0h] (5 fast sessions) and the
    previous window covers [-48h, -24h] (5 slow sessions) — also
    IMPROVING. Without the gap, the 24h window would absorb every
    session and the trend would collapse to STABLE.
    """
    db = tmp_path / "traces.db"
    TraceLogger(db)
    per_event: list[tuple[int, int, int]] = []
    # 5 slow sessions: -46h, -41h, -36h, -31h, -26h (all in previous 24h window).
    for idx in range(5):
        session_offset = -3600 * (46 - idx * 5)
        for event_idx, ms in enumerate([200] * 3):
            per_event.append((session_offset, ms, session_offset + event_idx))
    # 5 fast sessions: -4h, -3h, -2h, -1h, 0h (all in current 24h window).
    for idx in range(5):
        session_offset = -3600 * (4 - idx)
        for event_idx, ms in enumerate([50] * 3):
            per_event.append((session_offset, ms, session_offset + event_idx))
    _plant_windowed_with_custom_timestamps(db, per_event)

    report = aggregate_tool_latency(
        TraceLogger(db),
        windows=[LatencyWindow.LAST_5_SESSIONS, LatencyWindow.LAST_24H],
    )

    assert [s.window for s in report.windowed_sections] == [
        LatencyWindow.LAST_5_SESSIONS,
        LatencyWindow.LAST_24H,
    ]
    # Both windows should classify the speedup as IMPROVING.
    for section in report.windowed_sections:
        assert section.rows[0].trend_p95 is TrendDirection.IMPROVING, (
            f"{section.window}: expected IMPROVING, got {section.rows[0].trend_p95}"
        )


def test_windowed_no_windows_param_returns_empty_sections():
    """Default aggregate path leaves ``windowed_sections`` empty (backward compat)."""
    report = aggregate_tool_latency  # noqa: F841 — only used for the annotation below
    empty_report = ToolLatencyReport()
    assert empty_report.windowed_sections == []


def test_windowed_empty_store_returns_empty_sections(tmp_path):
    """An empty trace store produces one empty section per requested window."""
    db = tmp_path / "traces.db"
    TraceLogger(db)  # schema only, no events

    report = aggregate_tool_latency(
        TraceLogger(db),
        windows=[LatencyWindow.LAST_5_SESSIONS, LatencyWindow.LAST_24H],
    )

    assert report.rows == []
    assert report.total_calls == 0
    assert len(report.windowed_sections) == 2
    for section in report.windowed_sections:
        assert isinstance(section, WindowedLatencySection)
        assert section.rows == []
        assert section.total_calls == 0
        assert section.current_start is None
        assert section.previous_start is None


def test_render_markdown_includes_trend_section_when_windows_present():
    """Renderer appends a per-window table with the trend column when sections exist."""
    report = ToolLatencyReport(
        rows=[
            ToolLatencyRow(tool="read_file", count=10, p50_ms=50.0, p95_ms=100.0, p99_ms=100.0),
        ],
        total_calls=10,
        window_start=_ts(0.0),
        window_end=_ts(1.0),
        windowed_sections=[
            WindowedLatencySection(
                window=LatencyWindow.LAST_5_SESSIONS,
                current_start=_ts(0.0),
                current_end=_ts(1.0),
                previous_start=_ts(2.0),
                previous_end=_ts(3.0),
                rows=[
                    WindowedToolLatencyRow(
                        tool="read_file",
                        count=5,
                        p50_ms=80.0,
                        p95_ms=150.0,
                        p99_ms=150.0,
                        trend_p95=TrendDirection.DEGRADING,
                        delta_p95_ms=50.0,
                    ),
                ],
                total_calls=5,
            ),
        ],
    )

    rendered = render_tool_latency_markdown(report)

    # Existing single-aggregate table is preserved.
    assert "| Tool | Calls | p50 (ms) | p95 (ms) | p99 (ms) |" in rendered
    # The new per-window section header and trend column appear.
    assert "## Window: Last 5 sessions" in rendered
    assert "| read_file | 5 | 80 | 150 | 150 | +50 | ^ degrading |" in rendered
    assert "- Current window:" in rendered
    assert "- Previous window:" in rendered


def test_render_markdown_trend_unknown_shows_placeholder():
    """UNKNOWN trend renders the ``? unknown`` marker; ``Δ p95`` is blank."""
    report = ToolLatencyReport(
        rows=[],
        total_calls=0,
        windowed_sections=[
            WindowedLatencySection(
                window=LatencyWindow.LAST_5_SESSIONS,
                current_start=None,
                current_end=None,
                previous_start=None,
                previous_end=None,
                rows=[
                    WindowedToolLatencyRow(
                        tool="write_file",
                        count=2,
                        p50_ms=10.0,
                        p95_ms=20.0,
                        p99_ms=20.0,
                        trend_p95=TrendDirection.UNKNOWN,
                        delta_p95_ms=None,
                    ),
                ],
                total_calls=2,
            ),
        ],
    )

    rendered = render_tool_latency_markdown(report)

    assert "Previous window: no prior data" in rendered
    assert "| write_file | 2 | 10 | 20 | 20 |  | ? unknown |" in rendered


def test_render_markdown_no_windowed_sections_unchanged():
    """No-window path is byte-identical to the pre-#877 output shape."""
    report = ToolLatencyReport(
        rows=[
            ToolLatencyRow(tool="read_file", count=4, p50_ms=10.0, p95_ms=20.0, p99_ms=20.0),
        ],
        total_calls=4,
        window_start=_ts(0.0),
        window_end=_ts(1.0),
    )

    rendered = render_tool_latency_markdown(report)

    assert "## Window:" not in rendered
    assert "Trend" not in rendered
    assert "no tool_call events in window" not in rendered
    assert "| read_file | 4 | 10 | 20 | 20 |" in rendered


def test_cli_tool_latency_window_flag_emits_trend_section(tmp_path, capsys):
    """``--window`` surfaces the trend table through the CLI."""
    db = tmp_path / "traces.db"
    TraceLogger(db)
    per_event: list[tuple[int, int, int]] = []
    # 5 slow sessions (outside current 24h window) + 5 fast sessions.
    for idx in range(5):
        session_offset = -3600 * (46 - idx * 5)
        for event_idx, ms in enumerate([200] * 3):
            per_event.append((session_offset, ms, session_offset + event_idx))
    for idx in range(5):
        session_offset = -3600 * (4 - idx)
        for event_idx, ms in enumerate([50] * 3):
            per_event.append((session_offset, ms, session_offset + event_idx))
    _plant_windowed_with_custom_timestamps(db, per_event)

    rc = cli_main(
        [
            "tool-latency",
            "--db",
            str(db),
            "--window",
            "last_5_sessions",
            "--window",
            "last_24h",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "## Window: Last 5 sessions" in out
    assert "## Window: Last 24 hours" in out
    assert "improving" in out


def test_cli_tool_latency_window_flag_rejects_unknown_spec(tmp_path, capsys):
    """Unknown ``--window`` values exit non-zero (argparse error)."""
    db = tmp_path / "traces.db"
    TraceLogger(db)
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["tool-latency", "--db", str(db), "--window", "last_99_parsecs"])
    assert exc_info.value.code == 2


def test_windowed_aggregator_stays_under_2s_for_1000_sessions(tmp_path):
    """Acceptance criterion: windowed query ≤ 2s for 1000 sessions.

    Plants 1000 sessions with 3 ``tool_call`` events each, runs the
    full windowed pass against the largest spec set, and asserts the
    elapsed wall-clock is below the documented 2s budget. The threshold
    is intentionally generous so the test does not flake on slow CI
    runners; the design (one ``query_events`` cursor over all
    ``tool_call`` events, then in-memory bucketing per window) leaves
    substantial headroom under the budget on the test host.
    """
    import time

    db = tmp_path / "traces.db"
    TraceLogger(db)
    per_event: list[tuple[int, int, int]] = []
    # Spread 1000 sessions across ~30 days, 3 events per session.
    for idx in range(1000):
        # Sessions run from -30 days (-2_592_000s) up to 0.
        session_offset = -2_592_000 + idx * 2600
        for event_idx in range(3):
            per_event.append((session_offset, 50 + (idx % 100), session_offset + event_idx))
    _plant_windowed_with_custom_timestamps(db, per_event)

    start = time.perf_counter()
    report = aggregate_tool_latency(
        TraceLogger(db),
        windows=[
            LatencyWindow.LAST_5_SESSIONS,
            LatencyWindow.LAST_24H,
            LatencyWindow.LAST_7D,
        ],
    )
    elapsed = time.perf_counter() - start

    # Sanity: the windowed pass produces one section per spec.
    assert len(report.windowed_sections) == 3
    # Performance budget: 2s for 1000 sessions × 3 events (issue #877).
    assert elapsed < 2.0, f"windowed aggregation took {elapsed:.2f}s, expected < 2.0s"
