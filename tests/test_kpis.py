"""Tests for the KPI computation and CLI (issue #39)."""

from __future__ import annotations

import time

from foundry_x.observability.kpis import KpiSummary, compute_kpis, main
from foundry_x.trace.logger import TraceLogger


def _seed_session(
    logger: TraceLogger,
    harness_version: str,
    verdict: str | None = None,
    regression: bool = False,
    injection_block_count: int = 0,
) -> str:
    """Create a session with task_received + optional critic_verdict.

    Issue #120 adds the optional ``injection_block_count`` parameter: when
    >0, that many ``injection_blocked`` events are planted so the per-
    session KPI counter has something to surface.
    """
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        if verdict is not None:
            # Small delay so cycle-time is measurably positive.
            time.sleep(0.01)
            logger.record(
                sid,
                kind="critic_verdict",
                payload={"verdict": verdict, "regression": regression},
            )
        for i in range(injection_block_count):
            logger.record(
                sid,
                kind="injection_blocked",
                payload={
                    "markers": ["ignore_previous"],
                    "tool": "read_file",
                    "preview": f"block {i}",
                },
            )
    return sid


def test_compute_kpis_with_planted_data(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # 2 approved, 1 rejected (with regression) → improvement 2/3, regression 1/3
    _seed_session(logger, "v1", verdict="approved")
    _seed_session(logger, "v1", verdict="approved")
    _seed_session(logger, "v1", verdict="rejected", regression=True)

    summary = compute_kpis(logger)

    assert isinstance(summary, KpiSummary)
    assert summary.cycle_time_seconds is not None
    assert summary.cycle_time_seconds > 0.0
    assert 0.0 <= summary.regression_rate <= 1.0
    assert summary.improvement_rate == 2 / 3
    assert summary.regression_rate == 1 / 3
    assert summary.injection_blocks == {}


def test_compute_kpis_empty_db(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    summary = compute_kpis(logger)
    assert summary.cycle_time_seconds is None
    assert summary.regression_rate == 0.0
    assert summary.improvement_rate == 0.0
    assert summary.injection_blocks == {}


def test_compute_kpis_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict="approved")
    _seed_session(logger, "v2", verdict="rejected")

    summary = compute_kpis(logger, harness_version="v1")
    assert summary.improvement_rate == 1.0

    summary_v2 = compute_kpis(logger, harness_version="v2")
    assert summary_v2.improvement_rate == 0.0


def test_main_prints_markdown_table(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict="approved")
    _seed_session(logger, "v1", verdict="rejected")

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()

    assert rc == 0
    output = captured.out
    assert "Cycle Time" in output
    assert "Regression Rate" in output
    assert "Improvement Rate" in output
    # No injection blocks planted → no extra section.
    assert "Injection Blocked" not in output


# ---------------------------------------------------------------------------
# Issue #120: per-session ``injection_blocked`` count is surfaced by the
# ``foundry-kpis`` CLI when ≥1 session has ≥1 block. A clean trace store
# stays compact (no extra rows in the markdown table).
# ---------------------------------------------------------------------------


def test_injection_blocks_counted_per_session(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_session(logger, "v1", verdict="approved", injection_block_count=2)
    s2 = _seed_session(logger, "v1", verdict="approved", injection_block_count=1)
    # Clean session contributes nothing to the map.
    _seed_session(logger, "v1", verdict="approved")

    summary = compute_kpis(logger)

    assert summary.injection_blocks == {s1: 2, s2: 1}
    assert sum(summary.injection_blocks.values()) == 3


def test_injection_blocks_empty_when_no_events(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict="approved")

    summary = compute_kpis(logger)
    assert summary.injection_blocks == {}


def test_main_renders_injection_block_section_when_present(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_session(logger, "v1", verdict="approved", injection_block_count=3)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Injection Blocked" in output
    assert "3 block(s) across 1 session(s)" in output
    assert s1 in output
    assert "| 3 |" in output


def test_main_omits_injection_block_section_when_clean(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict="approved")

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    # Compact output for a clean store — no extra section, no extra table.
    assert "Injection Blocked" not in captured.out
