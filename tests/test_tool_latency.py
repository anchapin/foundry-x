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
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

from foundry_x.observability.cli import main as cli_main
from foundry_x.observability.tool_latency import (
    ToolLatencyReport,
    ToolLatencyRow,
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
