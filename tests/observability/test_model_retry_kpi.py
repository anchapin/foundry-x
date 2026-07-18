"""KPI and session-card coverage for model API retry events (issue #871)."""

from __future__ import annotations

import json

from foundry_x.observability.cli import main as observability_main
from foundry_x.observability.kpis import KpiSummary, compare_kpis, compute_kpis
from foundry_x.observability.kpis import main as kpi_main
from foundry_x.observability.session_card import format_session_card
from foundry_x.trace.logger import TraceLogger


def _seed_model_retries(
    logger: TraceLogger,
    harness_version: str,
    count: int,
) -> str:
    """Create one session with *count* production-shaped retry events."""
    with logger.session(harness_version=harness_version) as session_id:
        logger.record(session_id, "task_received", {"prompt": "exercise model API"})
        for attempt in range(1, count + 1):
            logger.record(
                session_id,
                "model_retry",
                {
                    "attempt": attempt,
                    "error_type": "HTTPStatusError",
                    "backoff_ms": attempt * 100,
                },
            )
    return session_id


def test_model_retry_count_defaults_to_zero_and_round_trips(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_model_retries(logger, "v1", 0)

    summary = compute_kpis(logger)
    round_tripped = KpiSummary.model_validate(summary.model_dump())

    assert summary.model_retry_count == 0
    assert round_tripped == summary


def test_model_retry_count_aggregates_and_filters_by_harness_version(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_model_retries(logger, "v1", 2)
    _seed_model_retries(logger, "v1", 1)
    _seed_model_retries(logger, "v2", 4)

    assert compute_kpis(logger).model_retry_count == 7
    assert compute_kpis(logger, harness_version="v1").model_retry_count == 3
    assert compute_kpis(logger, harness_version="v2").model_retry_count == 4


def test_model_retry_count_is_surfaced_in_kpi_cli_outputs(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_model_retries(logger, "v1", 3)

    assert kpi_main(["--db", str(db)]) == 0
    markdown = capsys.readouterr().out
    assert "Model Retries: 3 model API retry event(s) recorded by the runner." in markdown

    assert kpi_main(["--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["model_retry_count"] == 3


def test_model_retry_count_is_in_comparison_aggregates(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_model_retries(logger, "baseline", 1)
    _seed_model_retries(logger, "candidate", 4)

    comparison = compare_kpis(logger, "baseline", "candidate")

    assert comparison.baseline.model_retry_count == 1
    assert comparison.candidate.model_retry_count == 4
    assert comparison.deltas["model_retry_count"] == 3


def test_session_card_surfaces_model_retry_count(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    session_id = _seed_model_retries(logger, "v1", 2)
    session = next(item for item in logger.list_sessions() if item.session_id == session_id)

    card = format_session_card(session, logger.load_session(session_id))
    assert "model_retry=2" in card

    assert observability_main(["session-card", "--db", str(db), "--session-id", session_id]) == 0
    assert "model_retry=2" in capsys.readouterr().out
