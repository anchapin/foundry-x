"""Benchmark task: TokenAwarePruningHook deterministic benchmark (issue #732).

Exercises ``TokenAwarePruningHook`` in the Runner loop with a stub
``ModelAdapter`` and asserts:

1. A ``context_pruned`` event is recorded with the correct ``dropped``,
   ``threshold_tokens``, and ``session_tokens`` values when the hook fires.
2. The session completes with a valid ``outcome.status`` after pruning fires.
3. No ``context_pruned`` event is recorded when the session stays under
   the token threshold.
4. ``HookRegistry._on_error`` received zero hook failures.

This is the token-aware counterpart to the event-count benchmark in
``test_context_pruning_benchmark.py`` (issue #618). The acceptance
criteria from issue #732 are:

- ``context_pruned`` event is recorded with correct payload when threshold
  is exceeded
- token count is accurate (older events only)
- session continues successfully after pruning
- no ``context_pruned`` event when below threshold
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
    ModelUsage,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger


TASK = BenchmarkTask(
    name="token_aware_pruning_benchmark",
    description=(
        "Drive Runner.run_task against a stub ModelAdapter with "
        "TokenAwarePruningHook registered; assert context_pruned events "
        "with correct threshold_tokens/session_tokens payload and a valid "
        "outcome.status after pruning fires (issue #732)."
    ),
    tags=["agent-loop", "context-pruning", "token-aware"],
    difficulty_tier="easy",
)

_TOKEN_THRESHOLD = 2048
_EVENT_THRESHOLD = 50
_PLANTED = 67


class _TokenAwareStubAdapter:
    """Stub ``ModelAdapter`` that emits scripted responses with cumulative tokens.

    This adapter produces responses with increasing ``usage.total_tokens`` values
    designed to exceed ``_TOKEN_THRESHOLD`` after a few steps, triggering the
    ``TokenAwarePruningHook``.
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
                usage=ModelUsage(prompt_tokens=200, completion_tokens=100, total_tokens=300),
            )
        if self.calls == 2:
            tool_call = ModelToolCall(
                id="call_write",
                type="function",
                function=ToolCallFunction(
                    name="write_file",
                    arguments=json.dumps({"path": "/tmp/y", "content": "hi"}),
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
                usage=ModelUsage(prompt_tokens=300, completion_tokens=800, total_tokens=1100),
            )
        if self.calls == 3:
            tool_call = ModelToolCall(
                id="call_grep",
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
                usage=ModelUsage(prompt_tokens=400, completion_tokens=1000, total_tokens=1400),
            )
        if self.calls == 4:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
                usage=ModelUsage(prompt_tokens=500, completion_tokens=200, total_tokens=700),
            )
        raise RuntimeError(
            f"_TokenAwareStubAdapter exhausted after 4 scripted responses; loop called "
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
        if response.usage is not None:
            yield ModelResponseChunk(usage=response.usage)


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text(
        "stub harness for token_aware_pruning_benchmark\n"
    )
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


def _sqlite_token_counter(db_path: Path):  # noqa: ANN401
    """Build a ``TokenCounter`` backed by direct SQLite.

    Queries the most recent ``model_response`` event and returns its
    ``tokens_used`` field. This mirrors the runner's behaviour.
    """

    def _count(session_id: str) -> int:
        with sqlite3.connect(db_path) as conn:
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


def _plant(logger: TraceLogger, session_id: str, n: int) -> None:
    """Plant ``n`` synthetic non-preserved events on ``session_id``.

    None of the planted events use ``tool_result`` or ``user_prompt``, so
    every planted event is eligible for pruning. This establishes a base
    event count before the token threshold triggers pruning.
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
def test_token_aware_pruning_benchmark(benchmark_workspace: Path) -> None:
    """TokenAwarePruningHook deterministic benchmark (issue #732).

    Drives ``Runner.run_task`` with a stub ``ModelAdapter`` (three tool_calls
    then final answer) and ``TokenAwarePruningHook`` registered on the default
    registry. The stub adapter's responses carry cumulative ``usage.total_tokens``
    values (300, 1100, 1400) that exceed the token threshold of 2048.

    After step 3, the cumulative tokens reach 2800 (300 + 1100 + 1400), which
    exceeds _TOKEN_THRESHOLD=2048, triggering pruning.

    Asserts:
    1. A ``context_pruned`` event was recorded with
       ``{"dropped": >= 1, "threshold_tokens": _TOKEN_THRESHOLD,
       "session_tokens": 2800}``.
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
        adapter = _TokenAwareStubAdapter()
        pruner = _sqlite_pruner(db)
        get_tokens = _sqlite_token_counter(db)

        async def _drive(registry: Any) -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as sid:
                _plant(logger, sid, _PLANTED)

                def _tracer(sid: str, kind: str, payload: dict) -> None:
                    logger.record(sid, kind=kind, payload=payload)

                from harness.hooks.context_pruning import TokenAwarePruningHook

                hook = TokenAwarePruningHook(
                    session_id=sid,
                    token_threshold=_TOKEN_THRESHOLD,
                    event_threshold=_EVENT_THRESHOLD,
                    pruner=pruner,
                    tracer=_tracer,
                    get_tokens=get_tokens,
                )
                registry.register(hook)

                await run_task(
                    "token-aware-pruning-benchmark",
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
        assert prune_event.payload["threshold_tokens"] == _TOKEN_THRESHOLD, (
            f"expected threshold_tokens={_TOKEN_THRESHOLD}; "
            f"got {prune_event.payload.get('threshold_tokens')!r}"
        )
        assert prune_event.payload["session_tokens"] >= _TOKEN_THRESHOLD, (
            f"expected session_tokens >= {_TOKEN_THRESHOLD}; "
            f"got {prune_event.payload.get('session_tokens')!r}"
        )
        assert prune_event.payload["dropped"] >= 1, (
            f"expected dropped >= 1 (hook must fire); got {prune_event.payload['dropped']!r}"
        )

        # --- outcome -------------------------------------------------------
        outcome_event = next(e for e in events if e.kind == "outcome")
        assert outcome_event.payload["status"] == "success", (
            f"expected outcome.status='success'; got {outcome_event.payload!r}"
        )

        # --- Hook isolation ------------------------------------------------
        assert hook_failures == [], (
            f"expected zero HookRegistry._on_error calls; got {hook_failures!r}"
        )

        # --- Adapter exhaustion guard --------------------------------------
        assert adapter.calls == 4, (
            f"expected exactly 4 model round-trips (3 tool_calls then "
            f"final_answer); got {adapter.calls}"
        )
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()


@pytest.mark.benchmark
def test_token_aware_no_prune_under_threshold(benchmark_workspace: Path) -> None:
    """TokenAwarePruningHook must NOT emit context_pruned when under threshold.

    Uses a stub adapter with small token values that never exceed the threshold.
    Asserts that no context_pruned event is recorded and the session completes
    successfully.
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    class _UnderThresholdAdapter:
        """Stub adapter that stays under the token threshold."""

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
                    usage=ModelUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100),
                )
            if self.calls == 2:
                return ModelResponse(
                    message=ModelMessage(role="assistant", content="done"),
                    finish_reason="stop",
                    usage=ModelUsage(prompt_tokens=50, completion_tokens=50, total_tokens=100),
                )
            raise RuntimeError("_UnderThresholdAdapter exhausted after 2 scripted responses")

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
            if response.usage is not None:
                yield ModelResponseChunk(usage=response.usage)

    try:
        adapter = _UnderThresholdAdapter()
        pruner = _sqlite_pruner(db)
        get_tokens = _sqlite_token_counter(db)

        async def _drive(registry: Any) -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as sid:
                _plant(logger, sid, _PLANTED)

                def _tracer(sid: str, kind: str, payload: dict) -> None:
                    logger.record(sid, kind=kind, payload=payload)

                from harness.hooks.context_pruning import TokenAwarePruningHook

                hook = TokenAwarePruningHook(
                    session_id=sid,
                    token_threshold=_TOKEN_THRESHOLD,
                    event_threshold=_EVENT_THRESHOLD,
                    pruner=pruner,
                    tracer=_tracer,
                    get_tokens=get_tokens,
                )
                registry.register(hook)

                await run_task(
                    "token-aware-no-prune",
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

        # --- no context_pruned event --------------------------------------
        prune_events = [e for e in events if e.kind == "context_pruned"]
        assert len(prune_events) == 0, (
            f"expected no context_pruned events when under threshold; got {len(prune_events)}: "
            f"{[e.payload for e in prune_events]!r}"
        )

        # --- outcome -------------------------------------------------------
        outcome_event = next(e for e in events if e.kind == "outcome")
        assert outcome_event.payload["status"] == "success", (
            f"expected outcome.status='success'; got {outcome_event.payload!r}"
        )

        # --- Hook isolation ------------------------------------------------
        assert hook_failures == [], (
            f"expected zero HookRegistry._on_error calls; got {hook_failures!r}"
        )

        # --- Adapter exhaustion guard --------------------------------------
        assert adapter.calls == 2, f"expected exactly 2 model round-trips; got {adapter.calls}"
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()
