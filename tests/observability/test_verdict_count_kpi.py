"""KPI tests for issue #1461 — verdict_count / sessions_with_verdicts.

The aggregate ``regression_rate`` and ``improvement_rate`` are bare floats;
without their denominators a rate of 0.50 cannot distinguish 1-of-2 (noise)
from 50-of-100 (signal). ADR-0028 warns N<5 is too small to trust. These
tests verify the counts are populated on :class:`KpiSummary`, surfaced next
to each rate in the markdown output, and that a small-sample advisory fires
when ``sessions_with_verdicts < 5``.
"""

from __future__ import annotations

import time

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.kpis import (
    compute_kpis,
    main,
)
from foundry_x.observability.regression_report import record_verdict
from foundry_x.trace.logger import TraceLogger


def _seed_session(
    logger: TraceLogger,
    harness_version: str,
    verdict: bool | None = None,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
) -> str:
    """Create a session with ``task_received`` + optional persisted verdict."""
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        if verdict is not None:
            time.sleep(0.01)
            record_verdict(
                logger,
                sid,
                CriticVerdict(
                    verdict=verdict,
                    passed_checks=passed_checks or [],
                    failed_checks=failed_checks or [],
                ),
            )
    return sid


def test_verdict_counts_populated_on_summary(tmp_path):
    """``verdict_count`` / ``sessions_with_verdicts`` come from ``_verdict_rates``."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=False)

    summary = compute_kpis(logger)

    # Two sessions, each with one verdict → 2 verdicts across 2 sessions.
    assert summary.verdict_count == 2
    assert summary.sessions_with_verdicts == 2
    # Rates must still agree with the denominators.
    assert summary.improvement_rate == 1 / 2
    assert summary.regression_rate == 0 / 2


def test_verdict_counts_zero_on_clean_store(tmp_path):
    """A trace store with no verdicts yields zero denominators, not None."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # A session with no verdict — only task_received.
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})

    summary = compute_kpis(logger)

    assert summary.verdict_count == 0
    assert summary.sessions_with_verdicts == 0
    assert summary.improvement_rate == 0.0
    assert summary.regression_rate == 0.0


def test_markdown_shows_counts_next_to_rates(tmp_path, capsys):
    """The markdown table surfaces the denominator next to each rate."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=False)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=False)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()

    assert rc == 0
    output = captured.out
    regression_line = next(line for line in output.splitlines() if "Regression Rate" in line)
    improvement_line = next(line for line in output.splitlines() if "Improvement Rate" in line)
    # 5 sessions with verdicts feed regression_rate.
    assert "N=5 session(s)" in regression_line
    # 5 verdicts feed improvement_rate.
    assert "N=5 verdict(s)" in improvement_line


def test_markdown_small_sample_advisory_when_few_sessions(tmp_path, capsys):
    """An advisory fires when ``sessions_with_verdicts < 5`` (ADR-0028)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=False)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Small sample size" in captured.out
    assert "N<5" in captured.out


def test_markdown_no_advisory_when_enough_sessions(tmp_path, capsys):
    """The advisory is suppressed once ``sessions_with_verdicts >= 5``."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    for _ in range(5):
        _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Small sample size" not in captured.out
