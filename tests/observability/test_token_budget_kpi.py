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
    compare_kpis,
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


def _seed_model_response_usage(
    logger: TraceLogger,
    harness_version: str,
    usage_payloads: list[dict[str, int] | None],
) -> str:
    """Plant a session whose ``model_response`` events carry ``token_usage``.

    Each entry in *usage_payloads* becomes one ``model_response`` event whose
    ``usage`` key matches the runner's wire format
    (``{"prompt_tokens", "completion_tokens", "total_tokens"}``) and whose
    ``tokens_used`` is the running cumulative total, exactly as
    :func:`~foundry_x.execution.runner.run_task` records it (issues #191, #197).
    A ``None`` entry simulates an endpoint that omits usage accounting.
    """
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        running = 0
        for step, usage in enumerate(usage_payloads):
            if usage is not None:
                running += usage["total_tokens"]
            logger.record(
                sid,
                kind="model_response",
                payload={
                    "step": step,
                    "finish_reason": "stop",
                    "token_usage": usage,
                    "tokens_used": running,
                },
            )
    return sid


def test_token_budget_overrun_pct_returns_none_when_no_aborts(tmp_path):
    """When no session hit the token budget, overrun_pct is None (issue #1112)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)
    assert summary.token_budget_overrun_pct is None


def test_token_budget_overrun_pct_computes_mean_overrun(tmp_path):
    """Mean percentage overrun across sessions that hit token_budget (issue #1112).

    Two sessions hit the token budget:
    - Session 1: used 5100 of 5000 → overrun = (5100-5000)/5000 * 100 = 2%
    - Session 2: used 11000 of 10000 → overrun = (11000-10000)/10000 * 100 = 10%
    Mean = (2 + 10) / 2 = 6%
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, token_budget_abort=(5100, 5000))
    _seed_session(logger, "v1", verdict=True, token_budget_abort=(11000, 10000))
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)
    assert summary.token_budget_abort_count == 2
    assert summary.token_budget_overrun_pct == 6.0


def test_token_budget_overrun_pct_single_abort(tmp_path):
    """Single session overrun is returned directly (issue #1112)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, token_budget_abort=(5500, 5000))

    summary = compute_kpis(logger)
    assert summary.token_budget_abort_count == 1
    assert summary.token_budget_overrun_pct == 10.0


def test_token_budget_overrun_pct_compare_kpis(tmp_path):
    """Token budget overrun delta appears in compare_kpis (issue #1112)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, token_budget_abort=(5100, 5000))
    _seed_session(logger, "v2", verdict=True, token_budget_abort=(6000, 5000))

    comparison = compare_kpis(logger, "v1", "v2")
    assert comparison.baseline.token_budget_overrun_pct == 2.0
    assert comparison.candidate.token_budget_overrun_pct == 20.0
    assert comparison.deltas["token_budget_overrun_pct"] == 18.0


def test_token_totals_sums_total_tokens_per_session(tmp_path):
    """``token_totals`` accumulates ``usage.total_tokens`` per session."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_model_response_usage(
        logger,
        "v1",
        [
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        ],
    )

    summary = compute_kpis(logger)

    assert summary.token_totals == {s1: 45}


def test_token_totals_multiple_sessions(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}]
    )
    s2 = _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}]
    )

    summary = compute_kpis(logger)

    assert summary.token_totals == {s1: 10, s2: 50}
    assert sum(summary.token_totals.values()) == 60


def test_token_totals_skips_events_with_null_usage(tmp_path):
    """A ``model_response`` with ``usage: None`` contributes zero tokens."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_model_response_usage(
        logger,
        "v1",
        [None, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}],
    )

    summary = compute_kpis(logger)

    assert summary.token_totals == {s1: 10}


def test_token_totals_omits_session_with_no_usage(tmp_path):
    """A session whose ``model_response`` events all omit ``usage`` is absent."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_model_response_usage(logger, "v1", [None, None])

    summary = compute_kpis(logger)

    assert summary.token_totals == {}


def test_token_totals_empty_when_no_model_response_events(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)

    assert summary.token_totals == {}


def test_token_totals_respects_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}]
    )
    _seed_model_response_usage(
        logger, "v2", [{"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}]
    )

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert list(summary_v1.token_totals.values()) == [2]
    assert list(summary_v2.token_totals.values()) == [6]
