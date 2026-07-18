"""Benchmark: ``prune --vacuum`` reclaims SQLite WAL space (issue #896).

Acceptance criterion #3 for issue #896:

    A benchmark test creates 1000 sessions, prunes 900 with ``--vacuum``,
    asserts WAL file size < 2x main db size.

This is the deterministic gatekeeping benchmark the Critic runs to prove
the WAL reclaim path actually keeps ``logs/*.db-wal`` bounded. A
regression that re-introduces unbounded WAL growth (e.g. dropping the
``PRAGMA wal_checkpoint(TRUNCATE)`` call) fails here and blocks the
candidate harness from shipping. See ADR-0004 / ADR-0005 for the
benchmark-gate contract.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.trace.logger import TraceLogger

#: Acceptance threshold from issue #896. WAL must stay under 2x the live
#: database file after a vacuuming prune. The constant is named so a future
#: tightening (e.g. ``< 1.5x``) is a one-line change with a clear reason.
WAL_TO_DB_RATIO_LIMIT = 2

TASK = BenchmarkTask(
    name="prune_vacuum_reclaims_wal",
    description=(
        "TraceLogger.prune_sessions(vacuum=True) reclaims SQLite WAL pages so "
        "logs/*.db-wal stays under 2x the live database after a 90% prune "
        "(issue #896 acceptance criterion #3)."
    ),
    prompt=(
        "Inspect src/foundry_x/trace/logger.py: confirm prune_sessions() "
        "issues PRAGMA wal_checkpoint(TRUNCATE) when vacuum=True so the WAL "
        "sidecar does not grow unbounded across prune cycles."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After 1000 sessions + prune 900 with --vacuum, the WAL file is "
        "smaller than 2x the main database file."
    ),
    tags=["trace", "wal", "benchmark"],
)


def _wal_path(db_path: Path) -> Path:
    return db_path.with_suffix(db_path.suffix + "-wal")


@pytest.mark.benchmark
def test_prune_vacuum_keeps_wal_under_2x_db(benchmark_workspace: Path) -> None:
    """1000-session prune with ``--vacuum`` must keep WAL bounded (issue #896)."""
    db_path = benchmark_workspace / "traces.db"
    logger = TraceLogger(db_path, backend="sqlite")

    # 1000 sessions, each with enough payload to make the WAL accrue real
    # pages. Without vacuuming, successive prune cycles let this sidecar
    # grow to several times the live-data size.
    blob = "x" * 256
    all_sids: list[str] = []
    for _ in range(1000):
        with logger.session(harness_version="0.1.0") as sid:
            for _ in range(4):
                logger.record(sid, "tool_call", {"name": "read_file", "blob": blob})
            all_sids.append(sid)

    # Prune 900 (keep the 100 most recent) with vacuum enabled.
    deleted = logger.prune_sessions(all_sids[:900], vacuum=True)
    assert deleted == 900

    # Close so SQLite flushes and the on-disk sizes settle before we stat.
    logger.close()

    db_size = db_path.stat().st_size
    wal_size = _wal_path(db_path).stat().st_size if _wal_path(db_path).exists() else 0

    # Sanity: the database must actually contain the surviving sessions,
    # otherwise an empty-db regression would trivially satisfy the ratio.
    assert len(TraceLogger(db_path, backend="sqlite").list_sessions()) == 100
    assert db_size > 0
    assert wal_size < WAL_TO_DB_RATIO_LIMIT * db_size
