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
) -> str:
    """Create a session with task_received + optional critic_verdict."""
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


def test_compute_kpis_empty_db(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    summary = compute_kpis(logger)
    assert summary.cycle_time_seconds is None
    assert summary.regression_rate == 0.0
    assert summary.improvement_rate == 0.0


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
