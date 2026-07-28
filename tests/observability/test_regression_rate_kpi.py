"""KPI computation tests — split from test_kpis.py (issue #1282).

This file is kept for backwards compatibility. Tests have been moved to:
- tests/observability/test_cycle_time_kpi.py
- tests/observability/test_regression_rate_kpi.py
- tests/observability/test_improvement_rate_kpi.py
- tests/observability/test_context_efficiency_kpi.py
- tests/observability/test_token_budget_kpi.py
- tests/observability/test_kpi_comparison.py
- tests/observability/test_kpi_cli.py
"""

from __future__ import annotations

import time

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.kpis import (
    compute_kpis,
)
from foundry_x.observability.regression_report import record_verdict
from foundry_x.trace.logger import TraceLogger


def _seed_session(
    logger: TraceLogger,
    harness_version: str,
    verdict: bool | None = None,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
    injection_block_count: int = 0,
    failure_class: str | None = None,
    hook_registry_error: bool = False,
    wall_clock_abort: bool = False,
    token_budget_abort: tuple[int, int] | None = None,
    event_limit_abort: bool = False,
    tool_argument_parse_error_count: int = 0,
    generation_exhausted_count: int = 0,
) -> str:
    """Create a session with task_received + optional persisted critic_verdict.

    When ``verdict`` is not ``None`` a real CriticVerdict is persisted via
    ``record_verdict`` (issue #98), so the trace store holds the same
    ``VerdictRecord`` payload the production path writes.

    Issue #120 adds the optional ``injection_block_count`` parameter: when
    >0, that many ``injection_blocked`` events are planted so the per-
    session KPI counter has something to surface.

    Issue #705 adds the optional ``failure_class`` parameter: when provided,
    it is stored in the ``CriticVerdict`` so the KPI aggregation can
    compute ``failure_class_distribution``.

    Issue #800 adds ``hook_registry_error`` and ``wall_clock_abort`` parameters
    to plant the corresponding trace events for KPI computation.

    Issue #872 adds ``tool_argument_parse_error_count``: when >0, that many
    ``tool_argument_parse_error`` events are planted so the KPI aggregation
    can surface the count emitted by the runner at
    ``src/foundry_x/execution/runner.py:1684``.

    Issue #953 adds ``generation_exhausted_count``: when >0, that many
    ``generation_exhausted`` events are planted so the KPI aggregation can
    surface the evolver LLM failure count and rate.

    Issue #1112 adds ``token_budget_abort``: when provided as
    ``(tokens_used, token_budget)``, plants a ``task_aborted(reason="token_budget")``
    event with those values so the KPI aggregation can compute the overrun percentage.
    Issue #1113 adds ``event_limit_abort`` parameter to plant ``task_aborted``
    events with reason="event_limit" for cycle-time exclusion breakdown testing.
    """
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        if verdict is not None:
            # Small delay so cycle-time is measurably positive.
            time.sleep(0.01)
            record_verdict(
                logger,
                sid,
                CriticVerdict(
                    verdict=verdict,
                    passed_checks=passed_checks or [],
                    failed_checks=failed_checks or [],
                    failure_class=failure_class,
                ),
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
        if hook_registry_error:
            logger.record(
                sid,
                kind="hook_registry_error",
                payload={"error_type": "ImportError", "message": "test error"},
            )
        if wall_clock_abort:
            logger.record(
                sid,
                kind="task_aborted",
                payload={"reason": "wall_clock", "timeout_s": 1.0, "token_budget": None},
            )
        if token_budget_abort is not None:
            tokens_used, token_budget = token_budget_abort
            logger.record(
                sid,
                kind="task_aborted",
                payload={
                    "reason": "token_budget",
                    "tokens_used": tokens_used,
                    "token_budget": token_budget,
                },
            )
        if event_limit_abort:
            logger.record(
                sid,
                kind="task_aborted",
                payload={"reason": "event_limit", "event_limit": 100},
            )
        for i in range(tool_argument_parse_error_count):
            logger.record(
                sid,
                kind="tool_argument_parse_error",
                payload={
                    "step": i,
                    "call_id": f"call-{i}",
                    "name": "read_file",
                    "raw": "not-json",
                    "error": "JSONDecodeError: malformed",
                },
            )
        for i in range(generation_exhausted_count):
            logger.record(
                sid,
                kind="generation_exhausted",
                payload={
                    "max_retries": 2,
                    "final_error": f"generation failed: {i}",
                },
            )
    return sid


def test_regression_rate_counts_prior_pass_now_failing(tmp_path):
    """A task passing then failing in a later verdict counts as a regression."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", verdict=True, passed_checks=["smoke"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["smoke"])

    summary = compute_kpis(logger)

    # 1 of 2 sessions regressed; 1 of 2 verdicts approved.
    assert summary.regression_rate == 1 / 2
    assert summary.improvement_rate == 1 / 2
