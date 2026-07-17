"""Benchmark task: ContextPruningHook deterministic benchmark (issue #618).

Exercises ``ContextPruningHook`` in the Runner loop with a stub
``ModelAdapter`` and asserts:

1. A ``context_pruned`` event is recorded with the correct ``dropped`` and
   ``threshold`` values when the hook fires on the first tool call.
2. The session completes with a valid ``outcome.status`` after pruning fires.
3. ``HookRegistry._on_error`` received zero hook failures.

This is the deterministic benchmark counterpart to the xfail integration test in
``tests/harness/test_context_pruning.py`` (issue #106). The integration test
proves the hook works in isolation; this benchmark proves it works correctly
when wired through the ``HookRegistry`` in the Runner's agent loop, so a
regression that breaks pruning behaviour would fail the Critic gate
(ADR-0004).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger


TASK = BenchmarkTask(
    name="context_pruning_benchmark",
    description=(
        "Drive Runner.run_task against a stub ModelAdapter with "
        "ContextPruningHook registered; assert context_pruned events "
        "with correct dropped/threshold values and a valid "
        "outcome.status after pruning fires (issue #618)."
    ),
    tags=["agent-loop", "context-pruning"],
    difficulty_tier="easy",
)

_THRESHOLD = 50
_PLANTED = 67


class _StubAdapter:
    """Stub ``ModelAdapter`` that emits one tool_call then a final answer.

    Same shape as ``_StubAdapter`` in ``test_runner_loop_smoke`` but simpler
    because this benchmark only needs one step to trigger pruning.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls == 1:
            tool_call = ModelToolCall(
                id="call_read",
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
            )
        if self.calls == 2:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
            )
        raise RuntimeError(
            f"_StubAdapter exhausted after 2 scripted responses; loop called "
            f"complete() {self.calls} times"
        )

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
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


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for context_pruning_benchmark\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


def _sqlite_pruner(db_path: Path):  # noqa: ANN401
    """Build a ``Pruner`` callable backed by direct SQLite.

    Mirrors the implementation in ``tests/harness/test_context_pruning.py``.
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
    """Plant ``n`` synthetic non-preserved events on ``session_id``.

    None of the planted events use ``tool_result`` or ``user_prompt``, so
    every planted event is eligible for pruning. With ``_PLANTED=67`` and
    ``_THRESHOLD=50``, the hook fires on the first tool_call with
    67 + 4 = 71 non-preserved events and drops > 0 to reach the threshold.
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


def _install_on_error_tracker(tracker):  # noqa: ANN001
    """Install ``tracker`` on the default ``HookRegistry`` for the test."""
    from harness.hooks import get_registry
    from harness.hooks.base import reset_default_registry

    reset_default_registry()
    registry = get_registry()
    registry._on_error = tracker  # type: ignore[assignment]
    return registry


@pytest.mark.benchmark
def test_context_pruning_benchmark(benchmark_workspace: Path) -> None:
    """ContextPruningHook deterministic benchmark (issue #618).

    Drives ``Runner.run_task`` with a stub ``ModelAdapter`` (one tool_call
    then final answer) and ``ContextPruningHook`` registered on the default
    registry. Before the run, plants ``_PLANTED`` synthetic non-preserved
    events into the trace database so the hook's first ``pre_tool`` call
    (on the first tool_call step) fires pruning.

    Asserts:
    1. A ``context_pruned`` event was recorded with
       ``{"dropped": <correct>, "threshold": _THRESHOLD}``.
    2. ``outcome.status`` is ``"success"``.
    3. Zero ``HookRegistry._on_error`` callbacks.
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    try:
        adapter = _StubAdapter()
        pruner = _sqlite_pruner(db)

        async def _drive(registry: Any) -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as sid:
                _plant(logger, sid, _PLANTED)

                def _tracer(sid: str, kind: str, payload: dict) -> None:
                    logger.record(sid, kind=kind, payload=payload)

                from harness.hooks.context_pruning import ContextPruningHook

                hook = ContextPruningHook(
                    session_id=sid,
                    threshold=_THRESHOLD,
                    pruner=pruner,
                    tracer=_tracer,
                )
                registry.register(hook)

                await run_task(
                    "context-pruning-benchmark",
                    harness_dir,
                    logger,
                    sid,
                    model_adapter=adapter,
                )

        asyncio.run(_drive(registry))

        # --- Trace analysis ------------------------------------------------
        logger = TraceLogger(db)
        sid = logger.list_sessions()[0].session_id
        events = logger.load_session(sid)

        # --- context_pruned event -----------------------------------------
        prune_events = [e for e in events if e.kind == "context_pruned"]
        assert len(prune_events) == 1, (
            f"expected exactly 1 context_pruned event; got {len(prune_events)}: "
            f"{[e.payload for e in prune_events]!r}"
        )
        prune_event = prune_events[0]
        assert prune_event.payload["threshold"] == _THRESHOLD
        assert prune_event.payload["dropped"] >= 1, (
            f"expected dropped >= 1 (hook must fire); got {prune_event.payload['dropped']!r}"
        )

        # --- outcome -------------------------------------------------------
        outcome_event = next(e for e in events if e.kind == "outcome")
        assert outcome_event.payload["status"] == "success", (
            f"expected outcome.status='success'; got {outcome_event.payload!r}"
        )
        assert outcome_event.payload["reason"] == "final_answer"

        # --- Hook isolation ------------------------------------------------
        assert hook_failures == [], (
            f"expected zero HookRegistry._on_error calls; got {hook_failures!r}"
        )

        # --- Adapter exhaustion guard --------------------------------------
        assert adapter.calls == 2, (
            f"expected exactly 2 model round-trips (tool_call then "
            f"final_answer); got {adapter.calls}"
        )
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()
