"""KPI and session-card coverage for model_cost, model_rate_limit, and fetch_blocked events (issue #1281)."""

from __future__ import annotations

import json

from foundry_x.observability.kpis import KpiSummary, compare_kpis, compute_kpis
from foundry_x.observability.kpis import main as kpi_main
from foundry_x.trace.logger import TraceLogger


def _seed_model_costs(
    logger: TraceLogger,
    harness_version: str,
    count: int,
    cost_per_event: float = 0.01,
) -> str:
    """Create one session with *count* production-shaped model_cost events."""
    with logger.session(harness_version=harness_version) as session_id:
        logger.record(session_id, "task_received", {"prompt": "exercise model API"})
        for i in range(1, count + 1):
            logger.record(
                session_id,
                "model_cost",
                {
                    "provider": "anthropic",
                    "model": "claude-3-5-sonnet-20241022",
                    "prompt_tokens": 1000 + i,
                    "completion_tokens": 500 + i,
                    "estimated_cost_usd": cost_per_event * i,
                },
            )
    return session_id


def _seed_model_rate_limits(
    logger: TraceLogger,
    harness_version: str,
    count: int,
) -> str:
    """Create one session with *count* production-shaped model_rate_limit events."""
    with logger.session(harness_version=harness_version) as session_id:
        logger.record(session_id, "task_received", {"prompt": "exercise model API"})
        for _ in range(count):
            logger.record(
                session_id,
                "model_rate_limit",
                {
                    "requests_remaining": 100,
                    "tokens_remaining": 50000,
                    "requests_reset_seconds": 60.0,
                    "tokens_reset_seconds": 300.0,
                },
            )
    return session_id


def _seed_fetch_blocked(
    logger: TraceLogger,
    harness_version: str,
    count: int,
) -> str:
    """Create one session with *count* production-shaped fetch_blocked events."""
    with logger.session(harness_version=harness_version) as session_id:
        logger.record(session_id, "task_received", {"prompt": "block a fetch"})
        for i in range(count):
            logger.record(
                session_id,
                "fetch_blocked",
                {
                    "url": f"https://blocked-{i}.example.com/data",
                    "reason": "domain_not_in_allowlist",
                    "allowed_domains": ["allowed.example.com"],
                },
            )
    return session_id


class TestModelCostKpi:
    def test_model_cost_count_defaults_to_zero_and_round_trips(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_model_costs(logger, "v1", 0)

        summary = compute_kpis(logger)
        round_tripped = KpiSummary.model_validate(summary.model_dump())

        assert summary.model_cost_count == 0
        assert summary.total_model_cost_usd == 0.0
        assert round_tripped == summary

    def test_model_cost_count_aggregates_and_filters_by_harness_version(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_model_costs(logger, "v1", 2)
        _seed_model_costs(logger, "v1", 1)
        _seed_model_costs(logger, "v2", 4)

        assert compute_kpis(logger).model_cost_count == 7
        assert compute_kpis(logger, harness_version="v1").model_cost_count == 3
        assert compute_kpis(logger, harness_version="v2").model_cost_count == 4

    def test_total_model_cost_usd_sums_correctly(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_model_costs(logger, "v1", 3, cost_per_event=0.01)

        summary = compute_kpis(logger)
        assert summary.model_cost_count == 3
        assert abs(summary.total_model_cost_usd - 0.06) < 1e-9

    def test_model_cost_is_surfaced_in_kpi_cli_outputs(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)
        _seed_model_costs(logger, "v1", 2, cost_per_event=0.005)

        assert kpi_main(["--db", str(db)]) == 0
        markdown = capsys.readouterr().out
        assert "Model Cost: 2 cost event(s)" in markdown
        assert "total: $0.01" in markdown or "total: $0.0100" in markdown

        assert kpi_main(["--db", str(db), "--format", "json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["model_cost_count"] == 2

    def test_model_cost_is_in_comparison_aggregates(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_model_costs(logger, "baseline", 1, cost_per_event=0.01)
        _seed_model_costs(logger, "candidate", 4, cost_per_event=0.01)

        comparison = compare_kpis(logger, "baseline", "candidate")

        assert comparison.baseline.model_cost_count == 1
        assert comparison.candidate.model_cost_count == 4
        assert comparison.deltas["model_cost_count"] == 3
        # baseline: 1 * 0.01 = 0.01; candidate: (0.01 + 0.02 + 0.03 + 0.04) = 0.10
        # delta = 0.10 - 0.01 = 0.09
        assert abs(comparison.deltas["total_model_cost_usd"] - 0.09) < 1e-9


class TestModelRateLimitKpi:
    def test_model_rate_limit_count_defaults_to_zero_and_round_trips(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_model_rate_limits(logger, "v1", 0)

        summary = compute_kpis(logger)
        round_tripped = KpiSummary.model_validate(summary.model_dump())

        assert summary.model_rate_limit_count == 0
        assert round_tripped == summary

    def test_model_rate_limit_count_aggregates_and_filters_by_harness_version(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_model_rate_limits(logger, "v1", 2)
        _seed_model_rate_limits(logger, "v1", 1)
        _seed_model_rate_limits(logger, "v2", 4)

        assert compute_kpis(logger).model_rate_limit_count == 7
        assert compute_kpis(logger, harness_version="v1").model_rate_limit_count == 3
        assert compute_kpis(logger, harness_version="v2").model_rate_limit_count == 4

    def test_model_rate_limit_is_surfaced_in_kpi_cli_outputs(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)
        _seed_model_rate_limits(logger, "v1", 3)

        assert kpi_main(["--db", str(db)]) == 0
        markdown = capsys.readouterr().out
        assert "Model Rate Limits: 3" in markdown

        assert kpi_main(["--db", str(db), "--format", "json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["model_rate_limit_count"] == 3

    def test_model_rate_limit_is_in_comparison_aggregates(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_model_rate_limits(logger, "baseline", 1)
        _seed_model_rate_limits(logger, "candidate", 4)

        comparison = compare_kpis(logger, "baseline", "candidate")

        assert comparison.baseline.model_rate_limit_count == 1
        assert comparison.candidate.model_rate_limit_count == 4
        assert comparison.deltas["model_rate_limit_count"] == 3


class TestFetchBlockedKpi:
    def test_fetch_blocked_count_defaults_to_zero_and_round_trips(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_fetch_blocked(logger, "v1", 0)

        summary = compute_kpis(logger)
        round_tripped = KpiSummary.model_validate(summary.model_dump())

        assert summary.fetch_blocked_count == 0
        assert round_tripped == summary

    def test_fetch_blocked_count_aggregates_and_filters_by_harness_version(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_fetch_blocked(logger, "v1", 2)
        _seed_fetch_blocked(logger, "v1", 1)
        _seed_fetch_blocked(logger, "v2", 4)

        assert compute_kpis(logger).fetch_blocked_count == 7
        assert compute_kpis(logger, harness_version="v1").fetch_blocked_count == 3
        assert compute_kpis(logger, harness_version="v2").fetch_blocked_count == 4

    def test_fetch_blocked_is_surfaced_in_kpi_cli_outputs(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        logger = TraceLogger(db)
        _seed_fetch_blocked(logger, "v1", 2)

        assert kpi_main(["--db", str(db)]) == 0
        markdown = capsys.readouterr().out
        assert "Fetch Blocked: 2" in markdown

        assert kpi_main(["--db", str(db), "--format", "json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["fetch_blocked_count"] == 2

    def test_fetch_blocked_is_in_comparison_aggregates(self, tmp_path):
        logger = TraceLogger(tmp_path / "traces.db")
        _seed_fetch_blocked(logger, "baseline", 1)
        _seed_fetch_blocked(logger, "candidate", 4)

        comparison = compare_kpis(logger, "baseline", "candidate")

        assert comparison.baseline.fetch_blocked_count == 1
        assert comparison.candidate.fetch_blocked_count == 4
        assert comparison.deltas["fetch_blocked_count"] == 3
