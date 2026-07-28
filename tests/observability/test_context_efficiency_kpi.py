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

import pytest

from foundry_x.observability.kpis import (
    compute_kpis,
)
from foundry_x.trace.logger import TraceLogger


def _seed_context_pruned(
    logger: TraceLogger,
    harness_version: str,
    prune_count: int = 0,
) -> str:
    """Create a session with ``context_pruned`` events (issue #626).

    When ``prune_count`` > 0, that many ``context_pruned`` events are planted
    so the per-session KPI counter has something to surface.
    """
    from foundry_x.observability.kpis import CONTEXT_PRUNED_KIND

    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        for i in range(prune_count):
            logger.record(
                sid,
                kind=CONTEXT_PRUNED_KIND,
                payload={"dropped": i + 1, "threshold": 200},
            )
    return sid


def _seed_context_pruned_token_aware(
    logger: TraceLogger,
    harness_version: str,
    dropped: int,
    threshold_tokens: int,
    session_tokens: int,
) -> str:
    """Create a session with a token-aware ``context_pruned`` event (issue #1004).

    Token-aware sessions emit ``threshold_tokens`` instead of ``threshold``.
    """
    from foundry_x.observability.kpis import CONTEXT_PRUNED_KIND

    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        logger.record(
            sid,
            kind=CONTEXT_PRUNED_KIND,
            payload={
                "dropped": dropped,
                "threshold_tokens": threshold_tokens,
                "session_tokens": session_tokens,
            },
        )
    return sid


def test_context_efficiency_token_aware_session(tmp_path):
    """Issue #1004: token-aware sessions use threshold_tokens, not threshold.

    A token-aware session with dropped=100, threshold_tokens=8192 should contribute
    efficiency ≈ 1 - 100/(8192+100) ≈ 0.988, not 0.0.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_context_pruned_token_aware(
        logger, "v1", dropped=100, threshold_tokens=8192, session_tokens=8292
    )

    summary = compute_kpis(logger)

    assert summary.context_efficiency is not None
    expected = 1.0 - (100.0 / (8192.0 + 100.0))
    assert summary.context_efficiency == pytest.approx(expected)


def test_context_pruned_counted_per_session(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_context_pruned(logger, "v1", prune_count=2)
    s2 = _seed_context_pruned(logger, "v1", prune_count=1)
    _seed_context_pruned(logger, "v1", prune_count=0)

    summary = compute_kpis(logger)

    assert summary.context_pruned_count == {s1: 2, s2: 1}
    assert sum(summary.context_pruned_count.values()) == 3


def test_context_pruned_empty_when_no_events(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=0)

    summary = compute_kpis(logger)
    assert summary.context_pruned_count == {}


def test_context_pruned_respects_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=2)
    _seed_context_pruned(logger, "v2", prune_count=1)

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert list(summary_v1.context_pruned_count.values()) == [2]
    assert list(summary_v2.context_pruned_count.values()) == [1]


def test_context_efficiency_computed_when_prunes_present(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_context_pruned(logger, "v1", prune_count=1)
    _seed_context_pruned(logger, "v1", prune_count=1)

    summary = compute_kpis(logger)

    assert summary.context_efficiency is not None
    assert 0.0 <= summary.context_efficiency <= 1.0


def test_context_efficiency_includes_zero_prune_sessions_as_one(tmp_path):
    """Issue #979: a session with zero context_pruned events contributes 1.0."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=0)

    summary = compute_kpis(logger)

    assert summary.context_efficiency == 1.0


def test_context_efficiency_none_when_no_sessions(tmp_path):
    """No sessions at all → None (graceful degradation)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    summary = compute_kpis(logger)

    assert summary.context_efficiency is None


def test_context_efficiency_mixed_zero_and_nonzero_prune_sessions(tmp_path):
    """Issue #979: zero-pruning sessions included as 1.0 in the mean.

    One session prunes (dropped=1, threshold=200 → 1 - 1/201), one session
    never prunes (→ 1.0). Mean = (1 - 1/201 + 1.0) / 2.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=1)
    _seed_context_pruned(logger, "v1", prune_count=0)

    summary = compute_kpis(logger)

    assert summary.context_efficiency is not None
    pruned_efficiency = 1.0 - (1.0 / 201.0)
    expected = (pruned_efficiency + 1.0) / 2.0
    assert summary.context_efficiency == pytest.approx(expected)


def test_context_efficiency_respects_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=1)
    _seed_context_pruned(logger, "v2", prune_count=1)

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert summary_v1.context_efficiency is not None
    assert summary_v2.context_efficiency is not None
    assert summary_v1.context_efficiency == summary_v2.context_efficiency
