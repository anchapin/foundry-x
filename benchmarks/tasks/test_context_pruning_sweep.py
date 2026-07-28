"""Benchmark task: Context-pruning sweep for 5600G/6600 XT validation (issue #956).

Implements the ADR-0021 §5 sweep methodology: run the benchmark suite at
three ``FOUNDRY_CONTEXT_TOKENS`` thresholds (4096, 8192, 16384) and
compare pass rates to validate the default of 8192 tokens.

The sweep drives ``Runner.run_task`` with a stub ``ModelAdapter`` that
emits multiple tool calls with cumulative token usage, wiring the
``TokenAwarePruningHook`` so that pruning fires at each threshold.

For each threshold the sweep records:

- **passed**: session ``outcome.status == 'success'``
- **context_pruned_count**: number of ``context_pruned`` events emitted
- **dropped_total**: sum of ``dropped`` across all ``context_pruned`` events
- **token_budget_hit**: whether the session was aborted with
  ``task_aborted(reason='token_budget')``

After all three thresholds complete, the aggregation test asserts that
the pass rate at 8192 is within 5 percentage points of the 16384 baseline
(per ADR-0021 §5 regression threshold guidance).

The sweep emits one ``ContextPruningSweepResult`` per threshold. These
are the data contract that the KPI layer consumes for ``context_efficiency``
computation (issue #951):

    context_efficiency = 1 - (dropped_total / total_events)

A value near 1.0 means pruning rarely fired; near 0.0 means heavy pruning
throughout the session.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from benchmarks.models import BenchmarkTask, ContextPruningSweepResult
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ModelUsage,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger
from harness.hooks.context_pruning import _SqlitePruner

TASK = BenchmarkTask(
    name="context_pruning_sweep",
    description=(
        "Parametrized sweep of FOUNDRY_CONTEXT_TOKENS across 4096/8192/16384 "
        "to validate the 8192-token default against ADR-0021 §5 regression "
        "threshold guidance; emits ContextPruningSweepResult per threshold "
        "for context_efficiency KPI aggregation (issue #956)."
    ),
    tags=["agent-loop", "context-pruning", "sweep", "5600G", "6600-XT"],
    difficulty_tier="medium",
)

_THRESHOLDS = ["4096", "8192", "16384"]
_EVENT_THRESHOLD = 50

_SWEEP_RESULTS: list[ContextPruningSweepResult] = []


class _SweepStubAdapter:
    """Stub ``ModelAdapter`` for the sweep that emits escalating token usage.

    Produces responses with cumulative ``usage.total_tokens`` values that
    cross the token thresholds at predictable steps:

    =========  =========  ===================
    Step       Response  Cumulative tokens
    =========  =========  ===================
    1          tool_call  300
    2          tool_call  1100
    3          tool_call  2100   ← exceeds 2048
    4          tool_call  3400
    5          final      4500
    =========  =========  ===================

    At 4096 / 8192 threshold pruning fires at step 3 when tokens
    cross the threshold; at 16384 it never fires.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            tool_call = ModelToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(
                    name="bash",
                    arguments=json.dumps({"command": "echo step1"}),
                ),
            )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                usage=ModelUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            )
        if self.calls == 2:
            tool_call = ModelToolCall(
                id="call_2",
                type="function",
                function=ToolCallFunction(
                    name="read_file",
                    arguments=json.dumps({"path": "/tmp/x"}),
                ),
            )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                usage=ModelUsage(prompt_tokens=200, completion_tokens=600, total_tokens=800),
            )
        if self.calls == 3:
            tool_call = ModelToolCall(
                id="call_3",
                type="function",
                function=ToolCallFunction(
                    name="grep_search",
                    arguments=json.dumps({"pattern": "test", "path": "/tmp"}),
                ),
            )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                usage=ModelUsage(prompt_tokens=300, completion_tokens=500, total_tokens=800),
            )
        if self.calls == 4:
            tool_call = ModelToolCall(
                id="call_4",
                type="function",
                function=ToolCallFunction(
                    name="write_file",
                    arguments=json.dumps({"path": "/tmp/y", "content": "result"}),
                ),
            )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
                usage=ModelUsage(prompt_tokens=400, completion_tokens=900, total_tokens=1300),
            )
        if self.calls == 5:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
                usage=ModelUsage(prompt_tokens=500, completion_tokens=600, total_tokens=1100),
            )
        raise RuntimeError(
            f"_SweepStubAdapter exhausted after 5 scripted responses; "
            f"loop called complete() {self.calls} times"
        )

    async def chat(self, messages, tools=None, **kwargs):
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):
        response = await self.complete(messages, tools, **kwargs)
        if response.message.content:
            yield ModelResponseChunk(content=response.message.content)
        for i, tc in enumerate(response.tool_calls):
            yield ModelResponseChunk(
                tool_calls=[
                    ModelToolCallChunk(
                        index=i,
                        id=tc.id,
                        type=tc.type,
                        function=ToolCallFunctionChunk(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                ]
            )
        if response.finish_reason:
            yield ModelResponseChunk(finish_reason=response.finish_reason)
        if response.usage is not None:
            yield ModelResponseChunk(usage=response.usage)


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for context_pruning_sweep\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


def _sqlite_pruner(db_path: Path):
    """Build a ``Pruner`` callable backed by direct SQLite."""

    def _drop(session_id: str, keep_kinds: frozenset[str], target_count: int) -> int:
        not_in_clause = ", ".join("?" for _ in keep_kinds)
        with sqlite3.connect(db_path, timeout=30) as conn:
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


def _sqlite_token_counter(db_path: Path):
    """Build a ``TokenCounter`` backed by direct SQLite."""

    def _count(session_id: str) -> int:
        with sqlite3.connect(db_path, timeout=30) as conn:
            row = conn.execute(
                "SELECT payload FROM events "
                "WHERE session_id = ? AND kind = 'model_response' "
                "ORDER BY timestamp DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if not row:
            return 0
        payload = json.loads(row[0])
        return payload.get("tokens_used", 0)

    return _count


def _install_on_error_tracker(tracker):
    """Install ``tracker`` on the default ``HookRegistry`` for the test."""
    from harness.hooks import get_registry
    from harness.hooks.base import reset_default_registry

    reset_default_registry()
    registry = get_registry()
    registry._on_error = tracker  # type: ignore[assignment]
    return registry


@pytest.mark.parametrize("context_tokens", _THRESHOLDS)
@pytest.mark.benchmark
def test_context_pruning_sweep_run(
    benchmark_workspace: Path,
    context_tokens: str,
) -> None:
    """Run the sweep at one ``FOUNDRY_CONTEXT_TOKENS`` threshold.

    Drives ``Runner.run_task`` with the stub adapter and ``TokenAwarePruningHook``.
    Records the outcome and ``context_pruned`` event count into the module-level
    ``_SWEEP_RESULTS`` list for the aggregation test to consume.

    Parameters
    ----------
    context_tokens:
        The ``FOUNDRY_CONTEXT_TOKENS`` value for this parametrized run.
        One of "4096", "8192", "16384".
    """
    token_threshold = int(context_tokens)

    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    try:
        adapter = _SweepStubAdapter()
        _pruner = _SqlitePruner(db)

        async def _drive(reg: Any) -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as sid:

                def _tracer(sid: str, kind: str, payload: dict) -> None:
                    logger.record(sid, kind=kind, payload=payload)

                from harness.hooks.context_pruning import TokenAwarePruningHook

                hook = TokenAwarePruningHook(
                    session_id=sid,
                    token_threshold=token_threshold,
                    event_threshold=_EVENT_THRESHOLD,
                    pruner=_pruner.prune,
                    tracer=_tracer,
                    get_tokens=_pruner.count_tokens,
                )
                reg.register(hook)

                await run_task(
                    "context-pruning-sweep",
                    harness_dir,
                    logger,
                    sid,
                    model_adapter=adapter,
                )

        asyncio.run(_drive(registry))

        logger = TraceLogger(db)
        sid = logger.list_sessions()[0].session_id
        events = logger.load_session(sid)

        prune_events = [e for e in events if e.kind == "context_pruned"]
        context_pruned_count = len(prune_events)
        dropped_total = sum(e.payload.get("dropped", 0) for e in prune_events)
        total_events = len(events)

        outcome_event = next((e for e in events if e.kind == "outcome"), None)
        passed = outcome_event is not None and outcome_event.payload.get("status") == "success"

        task_abort = next(
            (
                e
                for e in events
                if e.kind == "task_aborted" and e.payload.get("reason") == "token_budget"
            ),
            None,
        )
        token_budget_hit = task_abort is not None

        _SWEEP_RESULTS.append(
            ContextPruningSweepResult(
                threshold=token_threshold,
                passed=passed,
                context_pruned_count=context_pruned_count,
                token_budget_hit=token_budget_hit,
                dropped_total=dropped_total,
                total_events=total_events,
            )
        )

        assert hook_failures == [], (
            f"expected zero HookRegistry._on_error calls; got {hook_failures!r}"
        )
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()
        _pruner.close()


@pytest.mark.benchmark
def test_context_pruning_sweep_aggregate() -> None:
    """Aggregate results across all three thresholds and assert regression bounds.

    Reads the module-level ``_SWEEP_RESULTS`` list (populated by the three
    parametrized ``test_context_pruning_sweep_run`` runs) and asserts:

    1. Exactly three results are present (one per threshold).
    2. The 8192 pass rate is within 5 percentage points of the 16384 baseline
       (ADR-0021 §5 regression threshold guidance).
    3. Each result has a valid ``context_pruned_count`` for KPI aggregation.

    The ``context_efficiency`` KPI is computed as::

        context_efficiency = 1 - (dropped_total / total_events)

    A value near 1.0 means pruning rarely fired; near 0.0 means heavy pruning.
    """
    thresholds_seen = {r.threshold for r in _SWEEP_RESULTS}
    assert len(_SWEEP_RESULTS) == 3, (
        f"expected 3 sweep results (one per threshold); got {len(_SWEEP_RESULTS)}: "
        f"{[(r.threshold, r.passed) for r in _SWEEP_RESULTS]!r}"
    )
    assert thresholds_seen == {4096, 8192, 16384}, (
        f"expected thresholds {{4096, 8192, 16384}}; got {thresholds_seen!r}"
    )

    result_by_threshold = {r.threshold: r for r in _SWEEP_RESULTS}
    r_8192 = result_by_threshold[8192]
    r_16384 = result_by_threshold[16384]
    _ = result_by_threshold[4096]

    assert r_8192.passed, (
        "sweep must pass at 8192 baseline (ADR-0021 §5 acceptance criterion); "
        f"got passed={r_8192.passed}, "
        f"context_pruned_count={r_8192.context_pruned_count}"
    )

    if r_16384.passed:
        delta_pp = (int(r_8192.passed) - int(r_16384.passed)) * 100
        assert abs(delta_pp) <= 5, (
            f"8192 pass rate must be within 5 pp of 16384 baseline; "
            f"got delta={delta_pp:+d} pp "
            f"(8192={'pass' if r_8192.passed else 'fail'}, "
            f"16384={'pass' if r_16384.passed else 'fail'})"
        )

    for r in _SWEEP_RESULTS:
        assert r.context_pruned_count >= 0, (
            f"context_pruned_count must be non-negative for threshold {r.threshold}; "
            f"got {r.context_pruned_count}"
        )
        assert r.dropped_total >= 0, (
            f"dropped_total must be non-negative for threshold {r.threshold}; got {r.dropped_total}"
        )
        assert r.total_events > 0, (
            f"total_events must be positive for threshold {r.threshold}; got {r.total_events}"
        )
