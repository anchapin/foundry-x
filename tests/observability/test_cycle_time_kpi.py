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
    KpiSummary,
    _format_delta,
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


def test_compute_kpis_with_planted_data(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # 2 approved, 1 rejected → improvement 2/3. The rejected session fails
    # "bench", which the two prior sessions passed → 1 regressed session of 3.
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["bench"])

    summary = compute_kpis(logger)

    assert isinstance(summary, KpiSummary)
    assert summary.cycle_time_seconds is not None
    assert summary.cycle_time_seconds > 0.0
    assert 0.0 <= summary.regression_rate <= 1.0
    assert summary.improvement_rate == 2 / 3
    assert summary.regression_rate == 1 / 3
    assert summary.injection_blocks == {}


def test_no_regression_when_failure_never_passed(tmp_path):
    """A failing task that was never previously passing is not a regression."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", verdict=True, passed_checks=["smoke"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["brand_new"])

    summary = compute_kpis(logger)

    assert summary.regression_rate == 0.0
    assert summary.improvement_rate == 1 / 2


def test_compute_kpis_empty_db(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    summary = compute_kpis(logger)
    assert summary.cycle_time_seconds is None
    assert summary.regression_rate == 0.0
    assert summary.improvement_rate == 0.0
    assert summary.injection_blocks == {}
    assert summary.token_totals == {}


# ---------------------------------------------------------------------------
# Issue #1112: token budget overrun percentage. The runner emits
# ``task_aborted(reason="token_budget")`` with ``tokens_used`` and
# ``token_budget`` in the payload; the KPI layer now extracts the overrun
# percentage so operators can distinguish a session that barely exceeded
# a correctly-sized budget from one that ran away.
# ---------------------------------------------------------------------------


def test_compute_kpis_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v2", verdict=False)

    summary = compute_kpis(logger, harness_version="v1")
    assert summary.improvement_rate == 1.0

    summary_v2 = compute_kpis(logger, harness_version="v2")
    assert summary_v2.improvement_rate == 0.0


# ---------------------------------------------------------------------------
# Issue #895: ``cycle_time_seconds`` excludes sessions that fail before the
# Critic runs (no ``critic_verdict``). The exclusion count is surfaced on
# ``KpiSummary.excluded_from_cycle_time`` so the survivorship bias is
# interpretable. ``foundry-kpis`` renders the count when > 0 and
# ``compare_kpis`` carries its baseline/candidate delta.
# ---------------------------------------------------------------------------


def test_cycle_time_excludes_session_without_critic_verdict(tmp_path):
    """A session with ``task_received`` but no ``critic_verdict`` is excluded (issue #895).

    Plants one session that fails before the Critic runs (no verdict)
    alongside one that completes normally. The exclusion count must be 1
    and the completing session still contributes to ``cycle_time_seconds``.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Session that completes: task_received + critic_verdict.
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Session that fails before the Critic runs: task_received only.
    _seed_session(logger, "v1", verdict=None)

    summary = compute_kpis(logger)

    # The excluding session is counted, not silently dropped.
    assert summary.excluded_from_cycle_time == 1
    # The completing session still produces a positive cycle time.
    assert summary.cycle_time_seconds is not None
    assert summary.cycle_time_seconds > 0.0


def test_cycle_time_excluded_count_is_zero_when_all_sessions_verdict(tmp_path):
    """Every session reaching a ``critic_verdict`` → exclusion count is 0."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["bench"])

    summary = compute_kpis(logger)

    assert summary.excluded_from_cycle_time == 0
    assert summary.cycle_time_seconds is not None


def test_cycle_time_excluded_count_respects_harness_version_filter(tmp_path):
    """The exclusion count honors ``harness_version`` like the other KPIs."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # v1: one completes, one excluded.
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=None)
    # v2: clean (both complete).
    _seed_session(logger, "v2", verdict=True)
    _seed_session(logger, "v2", verdict=True)

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert summary_v1.excluded_from_cycle_time == 1
    assert summary_v2.excluded_from_cycle_time == 0


def test_cycle_time_exclusion_breakdown_by_abort_reason(tmp_path):
    """Excluded sessions are attributed to the correct abort reason (issue #1113).

    Plants 1 completing session, plus 2 wall_clock, 3 token_budget,
    1 event_limit, and 1 with no abort event (other).  Verifies the breakdown
    counts match the acceptance criteria.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", wall_clock_abort=True)
    _seed_session(logger, "v1", wall_clock_abort=True)
    _seed_session(logger, "v1", token_budget_abort=(5000, 10000))
    _seed_session(logger, "v1", token_budget_abort=(5000, 10000))
    _seed_session(logger, "v1", token_budget_abort=(5000, 10000))
    _seed_session(logger, "v1", event_limit_abort=True)
    _seed_session(logger, "v1", verdict=None)

    summary = compute_kpis(logger)

    assert summary.excluded_from_cycle_time == 7
    assert summary.excluded_wall_clock == 2
    assert summary.excluded_token_budget == 3
    assert summary.excluded_event_limit == 1
    assert summary.excluded_other == 1
    assert summary.cycle_time_seconds is not None
    assert summary.cycle_time_seconds > 0.0


def test_cycle_time_excluded_other_when_no_abort_event(tmp_path):
    """Sessions excluded without a ``task_aborted`` event are counted as ``excluded_other``."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=None)

    summary = compute_kpis(logger)

    assert summary.excluded_from_cycle_time == 1
    assert summary.excluded_wall_clock == 0
    assert summary.excluded_token_budget == 0
    assert summary.excluded_event_limit == 0
    assert summary.excluded_other == 1


def test_cycle_time_exclusion_breakdown_respects_harness_version(tmp_path):
    """The per-reason breakdown honors ``harness_version`` filtering."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", wall_clock_abort=True)
    _seed_session(logger, "v2", verdict=True)
    _seed_session(logger, "v2", token_budget_abort=(5000, 10000))

    v1_summary = compute_kpis(logger, harness_version="v1")
    v2_summary = compute_kpis(logger, harness_version="v2")

    assert v1_summary.excluded_wall_clock == 1
    assert v1_summary.excluded_token_budget == 0
    assert v1_summary.excluded_event_limit == 0
    assert v1_summary.excluded_other == 0

    assert v2_summary.excluded_wall_clock == 0
    assert v2_summary.excluded_token_budget == 1
    assert v2_summary.excluded_event_limit == 0
    assert v2_summary.excluded_other == 0


def test_injection_blocks_counted_per_session(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_session(logger, "v1", verdict=True, injection_block_count=2)
    s2 = _seed_session(logger, "v1", verdict=True, injection_block_count=1)
    # Clean session contributes nothing to the map.
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)

    assert summary.injection_blocks == {s1: 2, s2: 1}
    assert sum(summary.injection_blocks.values()) == 3


def test_injection_blocks_empty_when_no_events(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)
    assert summary.injection_blocks == {}


def test_failure_class_distribution_aggregates_across_sessions(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, failure_class="tool-error")
    _seed_session(logger, "v1", verdict=False, failure_class="tool-error")
    _seed_session(logger, "v1", verdict=True, failure_class="bad-prompt")

    summary = compute_kpis(logger)

    assert summary.failure_class_distribution == {
        "tool-error": 2,
        "bad-prompt": 1,
    }


def test_failure_class_distribution_empty_when_no_verdicts(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)
    assert summary.failure_class_distribution == {}


def test_failure_class_distribution_respects_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, failure_class="tool-error")
    _seed_session(logger, "v2", verdict=True, failure_class="bad-prompt")

    v1_summary = compute_kpis(logger, harness_version="v1")
    v2_summary = compute_kpis(logger, harness_version="v2")

    assert v1_summary.failure_class_distribution == {"tool-error": 1}
    assert v2_summary.failure_class_distribution == {"bad-prompt": 1}


def test_failure_class_distribution_omits_none_values(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, failure_class=None)

    summary = compute_kpis(logger)
    assert summary.failure_class_distribution == {}


def test_format_delta_sign_convention():
    # Improvement-rate increase is positive (good).
    assert _format_delta(0.33, 0.50, higher_is_better=True) == "+0.17 (positive)"
    # Improvement-rate decrease is negative (bad).
    assert _format_delta(1.00, 0.50, higher_is_better=True) == "-0.50 (negative)"
    # Regression-rate increase is negative (bad).
    assert _format_delta(0.00, 0.50, higher_is_better=False) == "+0.50 (negative)"
    # Cycle-time increase is negative (bad).
    assert _format_delta(5.00, 10.00, higher_is_better=False) == "+5.00 (negative)"
    # Cycle-time decrease is positive (good).
    assert _format_delta(10.00, 5.00, higher_is_better=False) == "-5.00 (positive)"
    # No movement is neutral.
    assert _format_delta(0.50, 0.50, higher_is_better=True) == "+0.00 (neutral)"
    # An unmeasured side is N/A.
    assert _format_delta(None, 1.0, higher_is_better=True) == "N/A"
    assert _format_delta(1.0, None, higher_is_better=False) == "N/A"


def test_cycle_time_alert_threshold_exits_nonzero_when_exceeded(tmp_path, capsys):
    """Alert fires and main returns 1 when cycle_time exceeds threshold."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db), "--cycle-time-alert-threshold", "0.001"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "cycle_time_seconds" in captured.err
    assert "exceeds threshold" in captured.err


def test_cycle_time_alert_threshold_exits_zero_when_within_threshold(tmp_path, capsys):
    """No alert when cycle_time is at or below the threshold."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db), "--cycle-time-alert-threshold", "3600"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "ALERT" not in captured.err


def test_cycle_time_alert_threshold_exits_zero_when_no_cycle_time_data(tmp_path, capsys):
    """No alert is possible when cycle_time_seconds is None (no data)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # No verdicts → no critic_verdict event → no cycle time can be computed.
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})

    rc = main(["--db", str(db), "--cycle-time-alert-threshold", "1.0"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "ALERT" not in captured.err


def test_cycle_time_alert_threshold_message_includes_values(tmp_path, capsys):
    """Alert message names the KPI and both the actual and threshold values."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db), "--cycle-time-alert-threshold", "0.001"])
    captured = capsys.readouterr()

    assert rc == 1
    # Message contains cycle_time_seconds keyword and the threshold value.
    assert "cycle_time_seconds" in captured.err
    assert "exceeds threshold" in captured.err
    # The actual cycle time is a positive number and the threshold was 0.001.
    assert "0.00" in captured.err or "0.01" in captured.err


# ---------------------------------------------------------------------------
# Issue #872: tool-call argument parse-error count is exposed as an
# auxiliary KPI. The runner records a ``tool_argument_parse_error`` event
# whenever the model emits a tool call whose ``arguments`` JSON cannot be
# parsed (see ``src/foundry_x/execution/runner.py:1684``). The KPI counter
# is a session-aggregated scalar — surfaced in the markdown table when
# non-zero and in the JSON contract as ``tool_argument_parse_error_count``.
# ---------------------------------------------------------------------------


def test_tool_argument_parse_error_count_zero_when_clean(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)

    assert summary.tool_argument_parse_error_count == 0


def test_tool_argument_parse_error_count_aggregates_across_sessions(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, tool_argument_parse_error_count=2)
    _seed_session(logger, "v1", verdict=True, tool_argument_parse_error_count=3)
    _seed_session(logger, "v1", verdict=True)  # session with zero parse errors

    summary = compute_kpis(logger)

    assert summary.tool_argument_parse_error_count == 5


def test_tool_argument_parse_error_count_respects_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, tool_argument_parse_error_count=4)
    _seed_session(logger, "v2", verdict=True, tool_argument_parse_error_count=7)

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert summary_v1.tool_argument_parse_error_count == 4
    assert summary_v2.tool_argument_parse_error_count == 7


def test_tool_argument_parse_error_count_round_trips_through_kpi_summary(tmp_path):
    """The new field round-trips through ``KpiSummary.model_validate``."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, tool_argument_parse_error_count=2)

    summary = compute_kpis(logger)
    round_tripped = KpiSummary.model_validate(summary.model_dump())
    assert round_tripped == summary
    assert round_tripped.tool_argument_parse_error_count == 2
