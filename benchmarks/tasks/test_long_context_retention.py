"""Benchmark task: long_context_retention (issue #811).

Validates that the context-pruning hook (ADR-0021) does not corrupt
multi-step reasoning state when pruning fires during a task.  The hook
is wired with a deliberately low threshold so pruning fires on every
benchmark run, exercising the full contract:

    - ``ContextPruningHook`` drops oldest non-bookkeeping events when
      the session event count exceeds the threshold.
    - A ``context_pruned`` trace event is recorded with the dropped count.
    - The golden solution still completes the multi-step problem correctly
      despite pruning.

The task is a running sum: the agent reads a sequence of integers from
``input.txt`` (one per line) and writes the final cumulative total to
``output.txt``.  The running sum requires the agent to maintain state
across multiple steps, which is exactly the kind of reasoning that would
fail if pruning corrupted the session's intermediate reasoning state.

Acceptance criteria (issue #811):

- Task seeds a workspace with a multi-step problem requiring the agent
  to remember intermediate results (running sum).
- A context-pruning hook is registered with a low threshold (30 events)
  so pruning fires during the task.
- The golden solution completes the running sum correctly despite pruning.
- Post-condition: the task output is correct and pruning events are recorded.
- Evidence: planted synthetic events simulate a session that exceeds the
  pruning threshold, confirming pruning does not corrupt multi-step
  reasoning state.
"""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.trace.logger import TraceLogger
from harness.hooks.base import HookRegistry, ToolCall
from harness.hooks.context_pruning import ContextPruningHook

TASK = BenchmarkTask(
    name="long_context_retention",
    description=(
        "Read a sequence of integers from input.txt (one per line) and "
        "compute their running sum. Write only the final total to output.txt. "
        "Context pruning fires during the task; the golden solution must "
        "complete correctly despite pruning."
    ),
    prompt=(
        "Read integers from input.txt (one per line) and compute their "
        "cumulative running sum. Write only the final total as a single "
        "integer to output.txt."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "output.txt contains a single integer: the sum of all integers in "
        "input.txt. Pruning events are recorded in the trace store."
    ),
    tags=["context-pruning", "multi-step", "state-retention"],
)

# Low threshold to ensure pruning fires on every run (issue #811 acceptance).
_PRUNE_THRESHOLD = 30

# Number of synthetic events to plant -- must exceed _PRUNE_THRESHOLD to
# guarantee at least one prune fire during the task.
_PLANT_COUNT = 50

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    lines = Path("input.txt").read_text().splitlines()
    total = sum(int(line.strip()) for line in lines if line.strip())
    Path("output.txt").write_text(str(total) + "\\n")


if __name__ == "__main__":
    main()
"""


def _sqlite_pruner(db_path: Path):
    """Build a ``Pruner`` callable backed by direct SQLite.

    Mirrors the pattern in ``tests/harness/test_context_pruning.py``:
    the hook is decoupled from :class:`TraceLogger` (AGENTS.md §7
    self-reference loop), so the test wires a minimal SQLite closure
    rather than importing TraceLogger into the harness layer.
    """

    def _drop(session_id: str, keep_kinds: frozenset[str], target_count: int) -> int:
        not_in_clause = ", ".join("?" for _ in keep_kinds)
        with sqlite3.connect(db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            if total <= target_count:
                return 0
            to_drop = total - target_count
            params: list[object] = [session_id, *keep_kinds, to_drop]
            cursor = conn.execute(
                "SELECT event_id FROM events "
                "WHERE session_id = ? AND kind NOT IN (" + not_in_clause + ") "
                "ORDER BY timestamp LIMIT ?",
                params,
            )
            ids = [row[0] for row in cursor.fetchall()]
            if not ids:
                return 0
            placeholders = ", ".join("?" for _ in ids)
            conn.execute(
                "DELETE FROM events WHERE event_id IN (" + placeholders + ")",
                ids,
            )
            return len(ids)

    return _drop


def _plant(logger: TraceLogger, session_id: str, n: int) -> None:
    """Plant ``n`` synthetic events on ``session_id``.

    None use ``tool_result`` or ``user_prompt`` so every planted event
    is eligible for pruning, making the post-prune math unambiguous:
    ``n - _PRUNE_THRESHOLD`` events should be dropped.
    """
    kinds = (
        "tool_call",
        "task_received",
        "model_request",
        "model_response",
        "critic_verdict",
    )
    for i in range(n):
        logger.record(
            session_id,
            kind=kinds[i % len(kinds)],
            payload={"index": i, "marker": "synthetic"},
        )


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def trace_store(tmp_path: Path):
    """Provide a :class:`TraceLogger` backed by a temporary SQLite database."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    return logger


@pytest.mark.benchmark
def test_long_context_retention(
    benchmark_workspace: Path,
    trace_store: TraceLogger,
) -> None:
    """Verify the golden solution completes correctly despite context pruning.

    This benchmark validates the full contract of :class:`ContextPruningHook`:

    1. **Plant synthetic events** -- ``_PLANT_COUNT`` events are planted into
       the trace store, exceeding ``_PRUNE_THRESHOLD`` so pruning is guaranteed
       to fire when the hook processes the next tool call.
    2. **Register the hook** -- :class:`ContextPruningHook` is registered with
       ``threshold=_PRUNE_THRESHOLD`` on a fresh :class:`HookRegistry`.
    3. **Fire a tool call** -- a single synthetic tool call is processed through
       the hook chain, triggering the pruning logic.
    4. **Verify pruning fired** -- a ``context_pruned`` event is recorded with
       the correct dropped count.
    5. **Run golden solution** -- the multi-step running sum is computed;
       pruning must not have corrupted the session state.
    6. **Verify output** -- ``output.txt`` contains the correct sum.

    Acceptance criteria from issue #811:

    - Pruning fires during the task (synthetic events exceed threshold).
    - ``context_pruned`` trace event is recorded with the dropped count.
    - The golden solution completes the multi-step problem correctly.
    - Task output is correct despite pruning.
    """
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())
    expected = (fixture_dir / "expected.txt").read_text().strip()

    with trace_store.session(harness_version="test-long-context-retention") as sid:
        _plant(trace_store, sid, _PLANT_COUNT)

        pruner = _sqlite_pruner(trace_store.path)
        captured: list[dict] = []

        def tracer(_sid: str, kind: str, payload: dict) -> None:
            captured.append({"session_id": _sid, "kind": kind, "payload": dict(payload)})
            trace_store.record(_sid, kind=kind, payload=payload)

        registry = HookRegistry()
        hook = ContextPruningHook(
            session_id=sid,
            threshold=_PRUNE_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )
        registry.register(hook)

        _run(registry.run_pre(ToolCall(name="read_file", arguments={"path": "input.txt"})))

    prune_events = [e for e in captured if e["kind"] == "context_pruned"]
    assert len(prune_events) == 1, (
        f"expected exactly 1 context_pruned event, got {len(prune_events)}"
    )
    prune_payload = prune_events[0]["payload"]
    assert prune_payload["threshold"] == _PRUNE_THRESHOLD
    assert prune_payload["dropped"] == _PLANT_COUNT - _PRUNE_THRESHOLD, (
        f"expected dropped={_PLANT_COUNT - _PRUNE_THRESHOLD}, got {prune_payload['dropped']}"
    )

    (benchmark_workspace / "solution.py").write_text(GOLDEN_SOLUTION)
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "solution.py"],
        cwd=benchmark_workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"golden solution failed: rc={result.returncode}, stderr={result.stderr!r}"
    )

    actual = (benchmark_workspace / "output.txt").read_text().strip()
    assert actual == expected, (
        f"task {TASK.name}: output mismatch (got {actual!r}, expected {expected!r})"
    )
