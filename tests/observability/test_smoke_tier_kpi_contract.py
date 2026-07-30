"""Smoke-tier KPI exclusion contract tests (issue #1154, ADR-0034 §2).

ADR-0034 §2 establishes that smoke-tier tasks are:
- **Excluded** from the improvement-rate KPI  (they do not exercise agent capability)
- **Included** in the regression-rate KPI (a smoke failure indicates broken pipeline)

This module provides an automated test that verifies the KPI computation
code respects this contract.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.registry import load_all_tasks
from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.kpis import (
    TaskKpiMetadata,
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
) -> str:
    """Create a session with an optional critic verdict."""
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        if verdict is not None:
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


def _smoke_task_names() -> set[str]:
    """Return the set of all benchmark task names with difficulty_tier='smoke'."""
    return {task.name for task in load_all_tasks() if task.difficulty_tier == "smoke"}


def _task_metadata_for_names(
    names: set[str],
) -> dict[str, TaskKpiMetadata]:
    """Build task metadata for a set of task names.

    All tasks default to difficulty_tier='easy' unless they are in the
    smoke set (loaded from the live registry).
    """
    smoke = _smoke_task_names()
    out: dict[str, TaskKpiMetadata] = {}
    for name in names:
        out[name] = TaskKpiMetadata(
            name=name,
            difficulty_tier="smoke" if name in smoke else "easy",
        )
    return out


def test_smoke_tier_excluded_from_improvement_rate(tmp_path):
    """Smoke-tier verdicts must not appear in the improvement-rate denominator.

    Per ADR-0034 §2: smoke tasks do not exercise agent capability, so a
    harness that passes all smoke tasks but fails all easy/medium/hard
    tasks has NOT improved.  The improvement-rate denominator must exclude
    smoke verdicts.

    This test creates three sessions:
    - s1: 1 non-smoke verdict, approved
    - s2: 1 non-smoke verdict, rejected
    - s3: 1 smoke verdict, approved

    improvement_rate = approved_non_smoke / total_non_smoke = 1/2 (NOT 2/3).
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    non_smoke = "bench"
    smoke_task = next(iter(_smoke_task_names())) if _smoke_task_names() else None

    # s1: non-smoke, approved
    _seed_session(logger, "v1", verdict=True, passed_checks=[non_smoke])
    # s2: non-smoke, rejected
    _seed_session(logger, "v1", verdict=False, failed_checks=[non_smoke])
    # s3: smoke, approved
    if smoke_task:
        _seed_session(logger, "v1", verdict=True, passed_checks=[smoke_task])

    metadata = _task_metadata_for_names({non_smoke} | ({smoke_task} if smoke_task else set()))

    summary = compute_kpis(logger, task_metadata=metadata)

    # Only the two non-smoke verdicts count toward improvement-rate.
    # smoke verdicts are excluded per ADR-0034 §2.
    expected_imp = 1 / 2  # 1 approved / 2 total non-smoke
    assert summary.improvement_rate == pytest.approx(expected_imp), (
        f"improvement_rate={summary.improvement_rate} but expected {expected_imp} "
        f"(smoke verdicts must be excluded per ADR-0034 §2)"
    )


def test_smoke_tier_included_in_regression_rate(tmp_path):
    """A smoke-tier failure that follows a prior pass MUST count as a regression.

    Per ADR-0034 §2: smoke tasks ARE included in regression-rate because a
    smoke-task failure indicates a broken pipeline, not an agent regression.

    This test creates:
    - s1: smoke task, approved (passes first)
    - s2: same smoke task, failed (regressed)

    regression_rate = sessions_with_smoke_regression / total_sessions = 1/2.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    smoke_task = next(iter(_smoke_task_names())) if _smoke_task_names() else None
    if not smoke_task:
        pytest.skip("no smoke-tier tasks found in registry")

    # s1: smoke task passes
    _seed_session(logger, "v1", verdict=True, passed_checks=[smoke_task])
    # s2: same smoke task fails (regression)
    _seed_session(logger, "v1", verdict=False, failed_checks=[smoke_task])

    metadata = _task_metadata_for_names({smoke_task})

    summary = compute_kpis(logger, task_metadata=metadata)

    # Smoke tasks ARE counted in regression-rate per ADR-0034 §2.
    # The smoke task passed in s1 and failed in s2 → 1 regression out of 2 sessions.
    assert summary.regression_rate == pytest.approx(1 / 2), (
        f"regression_rate={summary.regression_rate} but expected {1 / 2} "
        f"(smoke-task regressions must be included per ADR-0034 §2)"
    )


def test_smoke_tier_kpi_contract_with_live_registry(tmp_path):
    """Integration test using the live benchmark registry.

    Verifies the smoke-tier exclusion contract against the actual declared
    benchmark tasks.  This test will change if new smoke tasks are added
    or existing ones reclassified.
    """
    smoke_tasks = _smoke_task_names()
    if len(smoke_tasks) < 2:
        pytest.skip(f"need at least 2 smoke tasks for this test, found: {smoke_tasks}")

    smoke_list = list(smoke_tasks)
    smoke_a, smoke_b = smoke_list[0], smoke_list[1]
    non_smoke = "bench"

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # s1: non-smoke approved, smoke_a approved
    _seed_session(logger, "v1", verdict=True, passed_checks=[non_smoke, smoke_a])
    # s2: non-smoke rejected, smoke_b approved
    _seed_session(logger, "v1", verdict=False, failed_checks=[non_smoke], passed_checks=[smoke_b])
    # s3: smoke_a passed earlier (s1), now failed → regression
    _seed_session(logger, "v1", verdict=False, failed_checks=[smoke_a])

    metadata = _task_metadata_for_names({non_smoke, smoke_a, smoke_b})

    summary = compute_kpis(logger, task_metadata=metadata)

    # improvement-rate: only non-smoke verdicts count
    # s1 (non-smoke approved) + s2 (non-smoke rejected) = 2 non-smoke verdicts, 1 approved
    expected_imp = 1 / 2
    assert summary.improvement_rate == pytest.approx(expected_imp), (
        f"improvement_rate={summary.improvement_rate} but expected {expected_imp}; "
        f"smoke verdicts must not appear in improvement-rate denominator (ADR-0034 §2)"
    )

    # regression-rate: smoke regressions count
    # s1 passed smoke_a; s2 bench regressed; s3 smoke_a regressed → 2 regressions out of 3 sessions
    expected_reg = 2 / 3
    assert summary.regression_rate == pytest.approx(expected_reg), (
        f"regression_rate={summary.regression_rate} but expected {expected_reg}; "
        f"smoke-task regressions must be included (ADR-0034 §2)"
    )


def test_smoke_tier_kpi_slices_show_correct_rates(tmp_path):
    """The per-difficulty_tier slices must also respect the exclusion contract.

    When grouping by difficulty_tier, the 'smoke' slice improvement_rate
    should reflect only non-regression (i.e., whether smoke tasks pass),
    while the aggregate improvement_rate excludes smoke entirely.
    """
    smoke_tasks = _smoke_task_names()
    if not smoke_tasks:
        pytest.skip("no smoke-tier tasks found in registry")

    smoke = next(iter(smoke_tasks))
    non_smoke = "bench"

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # s1: both approved
    _seed_session(logger, "v1", verdict=True, passed_checks=[non_smoke, smoke])
    # s2: non-smoke approved, smoke rejected (verdict=False because smoke failed)
    _seed_session(logger, "v1", verdict=False, failed_checks=[smoke], passed_checks=[non_smoke])

    metadata = _task_metadata_for_names({non_smoke, smoke})

    summary = compute_kpis(logger, group_by="difficulty_tier", task_metadata=metadata)

    # Aggregate improvement-rate excludes smoke: 1 approved / 1 non-smoke verdict = 1.0
    assert summary.improvement_rate == pytest.approx(1.0), (
        "aggregate improvement_rate must exclude smoke verdicts (ADR-0034 §2)"
    )

    # The smoke slice improvement_rate = 1/2 (smoke approved in s1, rejected in s2)
    assert "smoke" in summary.per_difficulty_tier
    smoke_slice = summary.per_difficulty_tier["smoke"]
    assert smoke_slice.verdict_count == 2
    assert smoke_slice.improvement_rate == pytest.approx(1 / 2)

    # The easy slice improvement_rate = 1.0 (non-smoke approved in both sessions)
    assert "easy" in summary.per_difficulty_tier
    easy_slice = summary.per_difficulty_tier["easy"]
    assert easy_slice.verdict_count == 2
    assert easy_slice.improvement_rate == pytest.approx(1.0)
