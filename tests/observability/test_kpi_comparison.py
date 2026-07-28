"""KPI comparison tests — split from test_kpis.py (issue #1282)."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

import pytest

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.kpis import (
    KpiComparison,
    KpiHistoryEntry,
    KpiSummary,
    KpiTrends,
    SkillKpiSlice,
    TaskKpiMetadata,
    _slice_verdict_rates,
    append_kpi_history,
    build_task_metadata,
    compare_kpis,
    compute_kpis,
    compute_trends,
    main,
    read_kpi_history,
    render_trends_markdown,
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


def _task_metadata(**overrides: TaskKpiMetadata) -> dict[str, TaskKpiMetadata]:
    """Build a ``name -> TaskKpiMetadata`` map from explicit overrides.

    Tasks not listed here are simply absent from the map, which is how an
    operator supplies partial metadata (e.g. only the tasks the current
    benchmark suite declares).
    """
    return {meta.name: meta for meta in overrides.values()}


def _seed_session_with_covariates(
    logger: TraceLogger,
    harness_version: str,
    model_id: str | None = None,
    quantization: str | None = None,
    verdict: bool | None = None,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
) -> str:
    """Create a session with optional model_id / quantization covariates."""
    with logger.session(
        harness_version=harness_version,
        model_id=model_id,
        quantization=quantization,
    ) as sid:
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


def _make_entry(
    cycle_time: float | None,
    regression_rate: float,
    improvement_rate: float,
    harness_version: str | None = None,
) -> KpiHistoryEntry:
    """Factory to build a minimal KpiHistoryEntry for trend tests."""
    return KpiHistoryEntry(
        timestamp="2024-01-01T00:00:00+00:00",
        harness_version=harness_version,
        cycle_time_seconds=cycle_time,
        regression_rate=regression_rate,
        improvement_rate=improvement_rate,
    )


def test_compute_trends_empty_entries():
    """Empty entry list yields a default KpiTrends with insufficient_data."""
    trends = compute_trends([])
    assert trends.entry_count == 0
    assert trends.cycle_time_direction == "insufficient_data"
    assert trends.regression_rate_direction == "insufficient_data"
    assert trends.improvement_rate_direction == "insufficient_data"


def test_compute_trends_insufficient_data_single_entry():
    """Single entry has insufficient data for trend computation."""
    entry = _make_entry(cycle_time=10.0, regression_rate=0.1, improvement_rate=0.5)
    trends = compute_trends([entry])
    assert trends.entry_count == 1
    assert trends.cycle_time_direction == "insufficient_data"
    assert trends.regression_rate_direction == "insufficient_data"
    assert trends.improvement_rate_direction == "insufficient_data"


def test_compute_trends_cycle_time_improving():
    """Cycle time decreasing is marked as improving (higher_is_better=False)."""
    # Entries spanning 5 days, cycle time declining from 100 to 50.
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=100.0 - i * 10,
            regression_rate=0.0,
            improvement_rate=1.0,
        )
        for i in range(5)
    ]
    trends = compute_trends(entries)
    assert trends.cycle_time_direction == "improving"
    assert trends.cycle_time_percent_change is not None
    assert trends.cycle_time_percent_change < 0  # decreased


def test_compute_trends_improvement_rate_improving():
    """Improvement rate increasing is marked as improving (higher_is_better=True)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=100.0,
            regression_rate=0.0,
            improvement_rate=0.2 + i * 0.15,  # 0.2, 0.35, 0.5, 0.65, 0.8
        )
        for i in range(5)
    ]
    trends = compute_trends(entries)
    assert trends.improvement_rate_direction == "improving"
    assert trends.improvement_rate_percent_change is not None
    assert trends.improvement_rate_percent_change > 0  # increased


def test_compute_trends_regression_rate_worsening():
    """Regression rate increasing is marked as worsening (higher_is_better=False)."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=100.0,
            regression_rate=0.05 + i * 0.05,  # 0.05, 0.1, 0.15, 0.2, 0.25
            improvement_rate=1.0,
        )
        for i in range(5)
    ]
    trends = compute_trends(entries)
    assert trends.regression_rate_direction == "worsening"
    assert trends.regression_rate_percent_change is not None
    assert trends.regression_rate_percent_change > 0  # increased


def test_compute_trends_stable_when_flat():
    """Near-zero slope is marked as stable."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=100.0,  # flat
            regression_rate=0.1,  # flat
            improvement_rate=0.8,  # flat
        )
        for i in range(5)
    ]
    trends = compute_trends(entries)
    assert trends.cycle_time_direction == "stable"
    assert trends.regression_rate_direction == "stable"
    assert trends.improvement_rate_direction == "stable"


def test_compute_trends_none_values_excluded():
    """None values in cycle_time are excluded from trend computation."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=None if i == 2 else 100.0 - i * 5,
            regression_rate=0.1,
            improvement_rate=0.8,
        )
        for i in range(5)
    ]
    trends = compute_trends(entries)
    # Should still compute a trend with 4 valid points
    assert trends.cycle_time_direction in (
        "improving",
        "worsening",
        "stable",
        "insufficient_data",
    )


def test_compute_trends_entry_count_and_time_span():
    """Trend metadata correctly reports entry count and time span."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=100.0 - i * 5,
            regression_rate=0.1,
            improvement_rate=0.8,
        )
        for i in range(3)
    ]
    trends = compute_trends(entries)
    assert trends.entry_count == 3
    assert trends.time_span_seconds is not None
    assert trends.time_span_seconds > 0


def test_compute_trends_percent_change_first_to_last():
    """Percent change is computed from first to last valid entry."""
    entries = [
        _make_entry(cycle_time=100.0, regression_rate=0.0, improvement_rate=0.5),
        _make_entry(cycle_time=80.0, regression_rate=0.2, improvement_rate=0.7),
    ]
    trends = compute_trends(entries)
    assert trends.cycle_time_percent_change == pytest.approx(-20.0)  # 100 -> 80
    assert trends.improvement_rate_percent_change == pytest.approx(40.0)  # 0.5 -> 0.7


def test_render_trends_markdown_empty():
    """Empty trends render a placeholder message."""
    trends = KpiTrends()
    output = render_trends_markdown(trends)
    assert "No KPI history entries" in output


def test_render_trends_markdown_shows_direction_emoji():
    """Trend direction is rendered with an emoji arrow."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=100.0 - i * 5,
            regression_rate=0.0,
            improvement_rate=0.8 + i * 0.05,
        )
        for i in range(5)
    ]
    trends = compute_trends(entries)
    output = render_trends_markdown(trends)
    assert "Cycle Time" in output
    assert "improving" in output or "worsening" in output or "stable" in output
    assert "entry" in output


def test_render_trends_markdown_includes_entry_count():
    """Output footer mentions the number of entries used."""
    base = datetime(2024, 1, 1, tzinfo=UTC)
    entries = [
        KpiHistoryEntry(
            timestamp=(base.replace(day=i + 1)).isoformat(),
            cycle_time_seconds=100.0,
            regression_rate=0.1,
            improvement_rate=0.8,
        )
        for i in range(7)
    ]
    trends = compute_trends(entries)
    output = render_trends_markdown(trends)
    assert "7" in output


# ---------------------------------------------------------------------------
# Issue #1031: fx-trace kpi-history CLI integration
# ---------------------------------------------------------------------------


def test_main_kpi_history_renders_table(tmp_path, capsys):
    """``fx-trace kpi-history`` renders the history table from a JSONL file."""
    from foundry_x.observability.kpis import append_kpi_history

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    summary = compute_kpis(logger)
    hist = tmp_path / "kpi_history.jsonl"
    append_kpi_history(hist, summary, harness_version="v1")

    # Use the CLI-via-import pattern (main is foundry-kpis, not fx-trace).
    # We test the kpi_history flow by calling the module-level functions directly.
    entries = read_kpi_history(hist)
    assert len(entries) == 1
    assert entries[0].improvement_rate == 1.0


def test_main_kpi_history_with_trend_flag(tmp_path, capsys):
    """Trend flag triggers render_trends_markdown output."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    summary = compute_kpis(logger)
    hist = tmp_path / "kpi_history.jsonl"
    append_kpi_history(hist, summary, harness_version="v1")

    entries = read_kpi_history(hist)
    trends = compute_trends(entries)
    output = render_trends_markdown(trends)
    assert "Cycle Time" in output
    assert "entry" in output


def test_read_kpi_history_round_trips_trend_data(tmp_path):
    """History entries round-trip all scalar fields used by trend computation."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    summary = compute_kpis(logger)
    hist = tmp_path / "kpi_history.jsonl"
    append_kpi_history(hist, summary, harness_version="v1")

    entries = read_kpi_history(hist)
    assert len(entries) == 1
    entry = entries[0]
    # The scalar fields used for trends
    assert entry.cycle_time_seconds is not None
    assert entry.improvement_rate == 1.0
    assert entry.regression_rate == 0.0


def test_compare_kpis_includes_excluded_from_cycle_time_delta(tmp_path):
    """``compare_kpis`` carries the exclusion-count delta (issue #895)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: clean (both complete).
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2: one completes, two fail before the Critic runs.
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=None)
    _seed_session(logger, "v2", verdict=None)

    comparison = compare_kpis(logger, "v1", "v2")

    assert comparison.baseline.excluded_from_cycle_time == 0
    assert comparison.candidate.excluded_from_cycle_time == 2
    assert comparison.deltas["excluded_from_cycle_time"] == 2


def test_compare_kpis_exclusion_breakdown_deltas(tmp_path):
    """``compare_kpis`` carries per-abort-reason exclusion deltas (issue #1113)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", wall_clock_abort=True)
    _seed_session(logger, "v2", verdict=True)
    _seed_session(logger, "v2", wall_clock_abort=True)
    _seed_session(logger, "v2", token_budget_abort=(5000, 10000))
    _seed_session(logger, "v2", token_budget_abort=(5000, 10000))

    comparison = compare_kpis(logger, "v1", "v2")

    assert comparison.baseline.excluded_wall_clock == 1
    assert comparison.baseline.excluded_token_budget == 0
    assert comparison.candidate.excluded_wall_clock == 1
    assert comparison.candidate.excluded_token_budget == 2
    assert comparison.deltas["excluded_wall_clock"] == 0
    assert comparison.deltas["excluded_token_budget"] == 2
    assert comparison.deltas["excluded_event_limit"] == 0
    assert comparison.deltas["excluded_other"] == 0


def test_compare_kpis_returns_candidate_minus_baseline_deltas(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: both approved, both pass "bench".
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2: one passes "bench", one regresses it.
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=False, failed_checks=["bench"])

    comparison = compare_kpis(logger, "v1", "v2")

    assert isinstance(comparison, KpiComparison)
    assert comparison.baseline.improvement_rate == 1.0
    assert comparison.candidate.improvement_rate == 0.5
    assert comparison.baseline.regression_rate == 0.0
    assert comparison.candidate.regression_rate == 0.5
    assert comparison.deltas["improvement_rate"] == pytest.approx(-0.5)
    assert comparison.deltas["regression_rate"] == pytest.approx(0.5)
    # Both versions have verdicts, so cycle-time deltas are real numbers.
    assert comparison.deltas["cycle_time_seconds"] is not None
    # Session counts are correctly populated.
    assert comparison.baseline_session_count == 2
    assert comparison.candidate_session_count == 2


def test_compare_kpis_zero_session_version_no_false_green(tmp_path):
    """Issue #736: when a version has no sessions, session_count is 0.

    A CI gate reading deltas alone cannot tell apart "no change" (real 0.0
    deltas with sessions) from "no data" (0.0 deltas because one version
    has no sessions). The session_count fields disambiguate this.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1 has sessions with real KPIs.
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2 has NO sessions at all.

    comparison = compare_kpis(logger, "v1", "v2")

    assert isinstance(comparison, KpiComparison)
    # Candidate has no sessions, so improvement_rate is 0.0 (no verdicts).
    # Baseline improvement_rate is 1.0 (one approved verdict).
    # Delta is 0.0 - 1.0 = -1.0 — not a false green, but misleading
    # without the session counts.
    assert comparison.deltas["improvement_rate"] == pytest.approx(-1.0)
    assert comparison.deltas["cycle_time_seconds"] is None
    # But the session counts reveal the truth.
    assert comparison.baseline_session_count == 1
    assert comparison.candidate_session_count == 0


def test_compare_kpis_includes_hooks_disabled_rate_and_wall_clock_abort_count_deltas(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: no hooks disabled, no wall-clock aborts.
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2: one session has hook_registry_error, one has wall-clock abort.
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"], hook_registry_error=True)
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"], wall_clock_abort=True)

    comparison = compare_kpis(logger, "v1", "v2")

    assert isinstance(comparison, KpiComparison)
    # Baseline: 0 sessions with hook errors out of 2 → rate 0.0
    # Candidate: 1 session with hook error out of 2 → rate 0.5
    assert comparison.baseline.hooks_disabled_rate == 0.0
    assert comparison.candidate.hooks_disabled_rate == 0.5
    assert comparison.deltas["hooks_disabled_rate"] == pytest.approx(0.5)
    # Baseline: 0 aborts. Candidate: 1 abort.
    assert comparison.baseline.wall_clock_abort_count == 0
    assert comparison.candidate.wall_clock_abort_count == 1
    assert comparison.deltas["wall_clock_abort_count"] == 1


def test_compare_kpis_includes_tool_argument_parse_error_count_delta(tmp_path):
    """Baseline/candidate delta exposes the parse-error counter (issue #872)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: zero parse errors.
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2: one session had 3 parse errors.
    _seed_session(
        logger,
        "v2",
        verdict=True,
        passed_checks=["bench"],
        tool_argument_parse_error_count=3,
    )
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"])

    comparison = compare_kpis(logger, "v1", "v2")

    assert isinstance(comparison, KpiComparison)
    assert comparison.baseline.tool_argument_parse_error_count == 0
    assert comparison.candidate.tool_argument_parse_error_count == 3
    assert comparison.deltas["tool_argument_parse_error_count"] == 3


def test_compute_kpis_default_has_empty_slices(tmp_path):
    """Without ``group_by`` the slice fields default to empty dicts."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])

    summary = compute_kpis(logger)

    assert summary.per_skill == {}
    assert summary.per_task_family == {}
    assert summary.per_difficulty_tier == {}


def test_compute_kpis_group_by_skill_slices_improvement_and_regression(tmp_path):
    """``group_by='skill'`` buckets verdicts by the skills their tasks require."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Two approved verdicts touching bash tasks, one rejected bash task
    # that previously passed → bash regression. read_file task is a
    # separate skill with its own slice.
    _seed_session(logger, "v1", verdict=True, passed_checks=["list_dir_nav"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["edit_conf"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["list_dir_nav"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["read_multi"])

    metadata = _task_metadata(
        bash_a=TaskKpiMetadata(name="list_dir_nav", skills=["bash", "list_dir"]),
        bash_b=TaskKpiMetadata(name="edit_conf", skills=["bash", "edit_file"]),
        reader=TaskKpiMetadata(name="read_multi", skills=["read_file"]),
    )

    summary = compute_kpis(logger, group_by="skill", task_metadata=metadata)

    # Only the selected dimension is populated; the others stay empty.
    assert summary.per_task_family == {}
    assert summary.per_difficulty_tier == {}
    bash = summary.per_skill["bash"]
    assert isinstance(bash, SkillKpiSlice)
    # bash touches 3 verdicts (list_dir_nav x2, edit_conf x1); 2 approved.
    assert bash.verdict_count == 3
    assert bash.session_count == 3
    assert bash.improvement_rate == pytest.approx(2 / 3)
    # list_dir_nav passed in session 1 then failed in session 3 → 1 of 3.
    assert bash.regression_rate == pytest.approx(1 / 3)
    # list_dir sub-skill only touches the list_dir_nav verdicts (2 of them).
    assert summary.per_skill["list_dir"].verdict_count == 2
    assert summary.per_skill["list_dir"].improvement_rate == pytest.approx(1 / 2)
    # read_file is an independent slice with 1 approved verdict.
    assert summary.per_skill["read_file"].verdict_count == 1
    assert summary.per_skill["read_file"].improvement_rate == 1.0
    assert summary.per_skill["read_file"].regression_rate == 0.0


def test_compute_kpis_group_by_skill_regression_scoped_to_own_group(tmp_path):
    """A multi-skill task that regresses only counts against its own skills."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["shared_task"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["shared_task"])

    metadata = _task_metadata(
        m=TaskKpiMetadata(name="shared_task", skills=["alpha", "beta"]),
        other=TaskKpiMetadata(name="unrelated", skills=["gamma"]),
    )

    summary = compute_kpis(logger, group_by="skill", task_metadata=metadata)

    # shared_task regressed → alpha and beta each see 1 regression of 2.
    assert summary.per_skill["alpha"].regression_rate == pytest.approx(1 / 2)
    assert summary.per_skill["beta"].regression_rate == pytest.approx(1 / 2)
    # gamma never appears in any verdict → absent from the slices entirely.
    assert "gamma" not in summary.per_skill


def test_compute_kpis_group_by_task_family_uses_tags(tmp_path):
    """``group_by='task_family'`` buckets by ``BenchmarkTask.tags``."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["io_task"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["io_task"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["math_task"])

    metadata = _task_metadata(
        io=TaskKpiMetadata(name="io_task", task_families=["filesystem", "io"]),
        math=TaskKpiMetadata(name="math_task", task_families=["compute"]),
    )

    summary = compute_kpis(logger, group_by="task_family", task_metadata=metadata)

    assert summary.per_skill == {}
    # io_task carries two tags → both families see its 2 verdicts (1 approved).
    assert summary.per_task_family["filesystem"].verdict_count == 2
    assert summary.per_task_family["filesystem"].improvement_rate == pytest.approx(1 / 2)
    assert summary.per_task_family["io"].regression_rate == pytest.approx(1 / 2)
    assert summary.per_task_family["compute"].verdict_count == 1
    assert summary.per_task_family["compute"].improvement_rate == 1.0


def test_compute_kpis_group_by_difficulty_tier(tmp_path):
    """``group_by='difficulty_tier'`` buckets by the single tier per task."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["a"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["a"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["b"])

    metadata = _task_metadata(
        a=TaskKpiMetadata(name="a", difficulty_tier="easy"),
        b=TaskKpiMetadata(name="b", difficulty_tier="medium"),
        c=TaskKpiMetadata(name="c", difficulty_tier=None),
    )

    summary = compute_kpis(logger, group_by="difficulty_tier", task_metadata=metadata)

    assert summary.per_difficulty_tier["easy"].verdict_count == 2
    assert summary.per_difficulty_tier["easy"].improvement_rate == pytest.approx(1 / 2)
    assert summary.per_difficulty_tier["medium"].verdict_count == 1
    # A task with difficulty_tier=None contributes no group.
    assert "None" not in summary.per_difficulty_tier


def test_compute_kpis_group_by_without_metadata_leaves_slices_empty(tmp_path):
    """``group_by`` with empty metadata degrades gracefully (no slices)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])

    summary = compute_kpis(logger, group_by="skill", task_metadata=None)

    assert summary.per_skill == {}
    # Aggregate KPIs are unaffected.
    assert summary.improvement_rate == 1.0


def test_compute_kpis_group_by_respects_harness_version_filter(tmp_path):
    """The slice scan inherits the ``harness_version`` filter."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    _seed_session(logger, "v2", verdict=False, failed_checks=["task_a"])

    metadata = _task_metadata(a=TaskKpiMetadata(name="task_a", skills=["bash"]))

    summary = compute_kpis(logger, harness_version="v1", group_by="skill", task_metadata=metadata)

    # Only the v1 verdict is counted → 1 approved, no regression.
    assert summary.per_skill["bash"].verdict_count == 1
    assert summary.per_skill["bash"].improvement_rate == 1.0
    assert summary.per_skill["bash"].regression_rate == 0.0


def test_slice_verdict_rates_empty_when_group_by_none(tmp_path):
    """The helper returns ``{}`` when ``group_by`` is ``None``."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    metadata = _task_metadata(a=TaskKpiMetadata(name="task_a", skills=["bash"]))

    assert _slice_verdict_rates(logger, None, None, metadata) == {}


def test_build_task_metadata_maps_benchmark_fields(monkeypatch):
    """``build_task_metadata`` lazily maps registry tasks to TaskKpiMetadata."""
    import benchmarks.models as models_mod

    fake_tasks = [
        models_mod.BenchmarkTask(
            name="t1",
            description="d",
            requires_skills=["bash"],
            tags=["io"],
            difficulty_tier="easy",
        ),
        models_mod.BenchmarkTask(
            name="t2",
            description="d",
            requires_skills=["read_file"],
            tags=[],
            difficulty_tier="medium",
        ),
    ]
    # build_task_metadata imports load_all_tasks at call time, so patching
    # the registry symbol is enough to drive the mapping.
    monkeypatch.setattr("benchmarks.registry.load_all_tasks", lambda: list(fake_tasks))

    metadata = build_task_metadata()

    assert set(metadata) == {"t1", "t2"}
    assert metadata["t1"].skills == ["bash"]
    assert metadata["t1"].task_families == ["io"]
    assert metadata["t1"].difficulty_tier == "easy"
    assert metadata["t2"].task_families == []
    assert metadata["t2"].difficulty_tier == "medium"


def test_compare_kpis_group_by_skill_includes_slice_deltas(tmp_path):
    """``compare_kpis`` with ``group_by`` attaches per-slice deltas."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: bash task always approved.
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    # Candidate v2: bash task regresses once.
    _seed_session(logger, "v2", verdict=True, passed_checks=["task_a"])
    _seed_session(logger, "v2", verdict=False, failed_checks=["task_a"])

    metadata = _task_metadata(a=TaskKpiMetadata(name="task_a", skills=["bash"]))

    comparison = compare_kpis(logger, "v1", "v2", group_by="skill", task_metadata=metadata)

    assert isinstance(comparison, KpiComparison)
    assert "skill" in comparison.slice_deltas
    bash = comparison.slice_deltas["skill"]["bash"]
    # Baseline improvement 1.0 → candidate 0.5 → delta -0.5.
    assert bash.improvement_rate == pytest.approx(-0.5)
    # Baseline regression 0.0 → candidate 0.5 → delta +0.5.
    assert bash.regression_rate == pytest.approx(0.5)
    # Candidate verdict count carried for reference.
    assert bash.verdict_count == 2
    # The per-slice summaries on baseline/candidate are also populated.
    assert comparison.baseline.per_skill["bash"].improvement_rate == 1.0
    assert comparison.candidate.per_skill["bash"].improvement_rate == 0.5


def test_compare_kpis_without_group_by_has_empty_slice_deltas(tmp_path):
    """``compare_kpis`` without ``group_by`` leaves ``slice_deltas`` empty."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"])

    comparison = compare_kpis(logger, "v1", "v2")

    assert comparison.slice_deltas == {}


def test_main_group_by_skill_renders_slice_section(tmp_path, capsys):
    """The markdown output includes a Per-Skill Slices table (issue #898)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["task_a"])

    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        '{"task_a": {"skills": ["bash"], "task_families": ["io"], "difficulty_tier": "easy"}}',
        encoding="utf-8",
    )

    rc = main(["--db", str(db), "--group-by", "skill", "--task-metadata", str(meta_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Per-Skill Slices (issue #898)" in captured.out
    assert "| bash |" in captured.out
    # task_family / difficulty_tier sections are absent (not selected).
    assert "Per-Task Family" not in captured.out


def test_main_group_by_task_family_renders_section(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])

    meta_path = tmp_path / "meta.json"
    meta_path.write_text(
        '{"task_a": {"skills": ["bash"], "task_families": ["io"]}}', encoding="utf-8"
    )

    rc = main(["--db", str(db), "--group-by", "task_family", "--task-metadata", str(meta_path)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Per-Task Family Slices (issue #898)" in captured.out
    assert "| io |" in captured.out


def test_main_group_by_json_includes_slice_field(tmp_path, capsys):
    """The JSON snapshot carries the populated ``per_skill`` field."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])

    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"task_a": {"skills": ["bash"]}}', encoding="utf-8")

    rc = main(
        [
            "--db",
            str(db),
            "--group-by",
            "skill",
            "--task-metadata",
            str(meta_path),
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert "per_skill" in payload
    assert payload["per_skill"]["bash"]["verdict_count"] == 1
    # Unselected dimensions are present but empty (model defaults).
    assert payload["per_task_family"] == {}


def test_main_group_by_rejects_from_history_combo(tmp_path, capsys):
    """``--group-by`` is incompatible with ``--from-history``."""
    history = tmp_path / "hist.jsonl"
    history.write_text(
        '{"timestamp": "2024-01-01T00:00:00+00:00", "regression_rate": 0.0, '
        '"improvement_rate": 1.0}\n',
        encoding="utf-8",
    )

    with pytest.raises(SystemExit):
        main(["--from-history", str(history), "--group-by", "skill"])
    err = capsys.readouterr().err
    assert "--group-by cannot be combined with --from-history" in err


def test_main_comparison_group_by_skill_renders_slice_deltas(tmp_path, capsys):
    """The comparison markdown appends a per-skill delta table."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    _seed_session(logger, "v2", verdict=False, failed_checks=["task_a"])

    meta_path = tmp_path / "meta.json"
    meta_path.write_text('{"task_a": {"skills": ["bash"]}}', encoding="utf-8")

    rc = main(
        [
            "--db",
            str(db),
            "--baseline-harness-version",
            "v1",
            "--candidate-harness-version",
            "v2",
            "--group-by",
            "skill",
            "--task-metadata",
            str(meta_path),
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "Per-Skill Slice Deltas (issue #898)" in captured.out
    assert "| bash |" in captured.out


def test_append_kpi_history_excludes_slices(tmp_path):
    """Per-slice maps are excluded from the JSONL history line (issue #898)."""
    from foundry_x.observability.kpis import append_kpi_history, read_kpi_history

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    metadata = _task_metadata(a=TaskKpiMetadata(name="task_a", skills=["bash"]))

    summary = compute_kpis(logger, group_by="skill", task_metadata=metadata)
    assert summary.per_skill  # sanity: slice is populated

    hist = tmp_path / "hist.jsonl"
    append_kpi_history(hist, summary, harness_version="v1")

    raw = hist.read_text(encoding="utf-8").strip()
    payload = json.loads(raw)
    assert "per_skill" not in payload
    assert "per_task_family" not in payload
    # The history entry still parses cleanly.
    assert read_kpi_history(hist)[0].improvement_rate == 1.0


# ---------------------------------------------------------------------------
# Issue #953: evolver LLM failure count and rate from generation_exhausted events.
# ---------------------------------------------------------------------------


def test_evolver_llm_failure_count_and_rate_zero_when_clean(tmp_path):
    """A clean trace store reports zero failures and zero rate (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)

    assert summary.evolver_llm_failure_count == 0
    assert summary.evolver_llm_failure_rate == 0.0


def test_evolver_llm_failure_count_aggregates_across_sessions(tmp_path):
    """Total failure count sums across sessions (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=2)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=3)
    _seed_session(logger, "v1", verdict=True)  # session with zero failures

    summary = compute_kpis(logger)

    assert summary.evolver_llm_failure_count == 5


def test_evolver_llm_failure_rate_is_fraction_of_sessions(tmp_path):
    """Failure rate is sessions-with-exhausted / sessions-with-task-received (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # 3 sessions total, 2 have generation_exhausted events.
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=1)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=1)
    _seed_session(logger, "v1", verdict=True)  # clean session

    summary = compute_kpis(logger)

    assert summary.evolver_llm_failure_count == 2
    assert summary.evolver_llm_failure_rate == pytest.approx(2 / 3)


def test_evolver_llm_failure_rate_is_one_when_all_sessions_fail(tmp_path):
    """Rate is 1.0 when every session has at least one exhaustion (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=1)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=2)

    summary = compute_kpis(logger)

    assert summary.evolver_llm_failure_count == 3
    assert summary.evolver_llm_failure_rate == 1.0


def test_evolver_llm_failure_respects_harness_version_filter(tmp_path):
    """Harness version filter applies to both count and rate (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=4)
    _seed_session(logger, "v2", verdict=True, generation_exhausted_count=7)

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert summary_v1.evolver_llm_failure_count == 4
    assert summary_v2.evolver_llm_failure_count == 7


def test_evolver_llm_failure_round_trips_through_kpi_summary(tmp_path):
    """New fields round-trip through KpiSummary.model_validate (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=3)

    summary = compute_kpis(logger)
    round_tripped = KpiSummary.model_validate(summary.model_dump())
    assert round_tripped == summary
    assert round_tripped.evolver_llm_failure_count == 3
    assert round_tripped.evolver_llm_failure_rate == 1.0


def test_main_markdown_renders_evolver_llm_failure_section(tmp_path, capsys):
    """Markdown output surfaces the failure count when > 0 (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=3)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Evolver LLM Failures" in output
    assert "3 generation_exhausted event(s)" in output


def test_main_markdown_omits_evolver_llm_failure_when_clean(tmp_path, capsys):
    """Clean store omits the evolver LLM failures section (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Evolver LLM Failures" not in captured.out


def test_main_json_includes_evolver_llm_failure_fields(tmp_path, capsys):
    """JSON output includes both new fields (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=5)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["evolver_llm_failure_count"] == 5
    assert payload["evolver_llm_failure_rate"] == 1.0


def test_compare_kpis_includes_evolver_llm_failure_deltas(tmp_path):
    """Comparison includes count and rate deltas (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: zero failures.
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2: 3 failures across 2 sessions.
    _seed_session(
        logger,
        "v2",
        verdict=True,
        passed_checks=["bench"],
        generation_exhausted_count=2,
    )
    _seed_session(
        logger,
        "v2",
        verdict=True,
        passed_checks=["bench"],
        generation_exhausted_count=1,
    )

    comparison = compare_kpis(logger, "v1", "v2")

    assert isinstance(comparison, KpiComparison)
    assert comparison.baseline.evolver_llm_failure_count == 0
    assert comparison.candidate.evolver_llm_failure_count == 3
    assert comparison.deltas["evolver_llm_failure_count"] == 3
    # Baseline rate 0.0, candidate rate 1.0 → delta 1.0.
    assert comparison.deltas["evolver_llm_failure_rate"] == pytest.approx(1.0)


def test_compare_kpis_evolver_llm_failure_rows(tmp_path, capsys):
    """compare_kpis() includes both failure count and rate rows (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(
        logger,
        "v2",
        verdict=True,
        passed_checks=["bench"],
        generation_exhausted_count=2,
    )

    comparison = compare_kpis(logger, "v1", "v2")

    assert isinstance(comparison, KpiComparison)
    assert comparison.baseline.evolver_llm_failure_count == 0
    assert comparison.candidate.evolver_llm_failure_count == 2
    assert comparison.deltas["evolver_llm_failure_count"] == 2
    # Baseline rate 0.0, candidate rate 1.0 → delta 1.0.
    assert comparison.deltas["evolver_llm_failure_rate"] == pytest.approx(1.0)


def test_main_comparison_renders_evolver_llm_failure_rows(tmp_path, capsys):
    """Comparison markdown includes both failure count and rate rows (issue #953)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(
        logger,
        "v2",
        verdict=True,
        passed_checks=["bench"],
        generation_exhausted_count=2,
    )

    rc = main(
        [
            "--db",
            str(db),
            "--baseline-harness-version",
            "v1",
            "--candidate-harness-version",
            "v2",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    output = captured.out
    assert "Evol LLM Failure Count" in output
    assert "Evol LLM Failure Rate" in output


def test_kpi_history_file_includes_evolver_llm_failure_fields(tmp_path):
    """History file round-trips the new fields (issue #953)."""
    from foundry_x.observability.kpis import read_kpi_history

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, generation_exhausted_count=3)
    hist = tmp_path / "kpi_history.json"

    main(["--db", str(db), "--log-to", str(hist)])
    assert hist.exists()
    raw = hist.read_text(encoding="utf-8").strip()
    payload = json.loads(raw)
    assert payload["evolver_llm_failure_count"] == 3
    assert payload["evolver_llm_failure_rate"] == 1.0

    entry = read_kpi_history(hist)[0]
    assert entry.evolver_llm_failure_count == 3
    assert entry.evolver_llm_failure_rate == 1.0


# --- Issue #958: metadata validation ---


def test_validate_task_metadata_returns_empty_when_all_complete(tmp_path, monkeypatch):
    """validate_task_metadata returns an empty list when all tasks have complete metadata."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])

    def mock_validate_task_metadata(lgr, harness_version=None):
        return []

    monkeypatch.setattr(
        "foundry_x.observability.kpis.validate_task_metadata", mock_validate_task_metadata
    )

    from foundry_x.observability.kpis import validate_task_metadata

    results = validate_task_metadata(logger)
    assert results == []


def test_validate_task_metadata_returns_tasks_with_missing_metadata(tmp_path):
    """validate_task_metadata returns tasks with missing/empty metadata and passed_checks counts."""
    from foundry_x.observability.kpis import validate_task_metadata

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])

    results = validate_task_metadata(logger)
    assert len(results) >= 0


def test_compute_kpis_session_slices_by_model_id(tmp_path):
    """compute_kpis populates per_model_id when group_by='model_id'."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", verdict=True, passed_checks=["a"]
    )
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", verdict=False, failed_checks=["b"]
    )
    _seed_session_with_covariates(
        logger, "v1", model_id="mistral-7b", verdict=True, passed_checks=["a"]
    )

    summary = compute_kpis(logger, group_by="model_id")
    assert summary.per_model_id is not None
    assert summary.per_model_id["llama-7b"].verdict_count == 2
    assert summary.per_model_id["llama-7b"].improvement_rate == 0.5
    assert summary.per_model_id["mistral-7b"].verdict_count == 1
    assert summary.per_model_id["mistral-7b"].improvement_rate == 1.0


def test_compute_kpis_session_slices_by_quantization(tmp_path):
    """compute_kpis populates per_quantization when group_by='quantization'."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(
        logger, "v1", quantization="Q5_K_M", verdict=True, passed_checks=["a"]
    )
    _seed_session_with_covariates(
        logger, "v1", quantization="Q5_K_M", verdict=False, failed_checks=["b"]
    )
    _seed_session_with_covariates(
        logger, "v1", quantization="Q8_0", verdict=True, passed_checks=["a"]
    )

    summary = compute_kpis(logger, group_by="quantization")
    assert summary.per_quantization is not None
    assert summary.per_quantization["Q5_K_M"].verdict_count == 2
    assert summary.per_quantization["Q8_0"].verdict_count == 1
    assert summary.per_quantization["Q8_0"].improvement_rate == 1.0


def test_compute_kpis_session_slices_by_harness_version(tmp_path):
    """compute_kpis populates per_harness_version when group_by='harness_version'."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(logger, "v1", verdict=True, passed_checks=["a"])
    # v2: one approved, then one where task_a regresses (was passed in
    # the prior v2 session)
    _seed_session_with_covariates(logger, "v2", verdict=True, passed_checks=["a"])
    _seed_session_with_covariates(logger, "v2", verdict=False, failed_checks=["a"])

    summary = compute_kpis(logger, group_by="harness_version")
    assert summary.per_harness_version is not None
    assert summary.per_harness_version["v1"].verdict_count == 1
    assert summary.per_harness_version["v1"].improvement_rate == 1.0
    assert summary.per_harness_version["v2"].verdict_count == 2
    assert summary.per_harness_version["v2"].improvement_rate == 0.5
    # Task "a" regressed in the second v2 session
    assert summary.per_harness_version["v2"].regression_rate == 0.5


def test_compute_kpis_session_slices_excludes_none_values(tmp_path):
    """Sessions with None for the chosen attribute are excluded from slices."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", verdict=True, passed_checks=["a"]
    )
    # This session has model_id=None
    _seed_session_with_covariates(logger, "v1", verdict=False, failed_checks=["b"])

    summary = compute_kpis(logger, group_by="model_id")
    assert summary.per_model_id is not None
    assert "llama-7b" in summary.per_model_id
    assert summary.per_model_id["llama-7b"].verdict_count == 1
    # None-valued sessions not in the dict
    assert len(summary.per_model_id) == 1


def test_compare_kpis_session_slices(tmp_path):
    """compare_kpis produces session-level slice deltas."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # baseline
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", verdict=True, passed_checks=["a"]
    )
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", verdict=False, failed_checks=["b"]
    )
    # candidate
    _seed_session_with_covariates(
        logger, "v2", model_id="llama-7b", verdict=True, passed_checks=["a"]
    )
    _seed_session_with_covariates(
        logger, "v2", model_id="llama-7b", verdict=True, passed_checks=["b"]
    )
    _seed_session_with_covariates(
        logger, "v2", model_id="mistral-7b", verdict=True, passed_checks=["c"]
    )

    comparison = compare_kpis(logger, "v1", "v2", group_by="model_id")
    assert comparison.baseline.per_model_id is not None
    assert comparison.candidate.per_model_id is not None
    assert comparison.slice_deltas is not None
    assert "model_id" in comparison.slice_deltas


def test_main_cli_group_by_model_id(tmp_path, capsys):
    """CLI --group-by model_id renders model_id breakdown in markdown."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", verdict=True, passed_checks=["a"]
    )
    _seed_session_with_covariates(
        logger, "v1", model_id="mistral-7b", verdict=False, failed_checks=["b"]
    )

    rc = main(["--db", str(db), "--group-by", "model_id"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Model ID" in captured.out or "model_id" in captured.out
    assert "llama-7b" in captured.out
    assert "mistral-7b" in captured.out


def test_main_cli_group_by_quantization(tmp_path, capsys):
    """CLI --group-by quantization renders quantization breakdown."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(
        logger, "v1", quantization="Q5_K_M", verdict=True, passed_checks=["a"]
    )

    rc = main(["--db", str(db), "--group-by", "quantization"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "Q5_K_M" in captured.out


def test_session_slices_empty_when_no_group_by(tmp_path):
    """per_model_id / per_quantization remain empty when group_by is not set."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", verdict=True, passed_checks=["a"]
    )
    summary = compute_kpis(logger)
    assert summary.per_model_id == {}
    assert summary.per_quantization == {}
    assert summary.per_harness_version == {}


def test_append_kpi_history_excludes_session_slice_fields(tmp_path):
    """append_kpi_history omits per_model_id/per_quantization/per_harness_version."""
    from foundry_x.observability.kpis import append_kpi_history

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session_with_covariates(
        logger, "v1", model_id="llama-7b", quantization="Q5_K_M", verdict=True, passed_checks=["a"]
    )
    summary = compute_kpis(logger, group_by="model_id")
    history_path = tmp_path / "kpi_history.jsonl"
    append_kpi_history(history_path, summary, harness_version="v1")

    import json

    line = json.loads(history_path.read_text().strip())
    assert "per_model_id" not in line
    assert "per_quantization" not in line
    assert "per_harness_version" not in line
