"""Benchmark task: validate session trace coherence after aggressive pruning (issue #1285).

Exercises ``ContextPruningHook`` in the Runner loop and asserts the two
trace-integrity invariants that existing pruning benchmarks leave untested:

1. Every ``tool_call`` event has a surviving ``tool_result`` with the same
   ``call_id`` -- a ``tool_call`` whose result was pruned is an orphaned
   instrument that corrupts the Evolver's diagnostic signal.

2. ``user_prompt`` events are never lost -- the task prompt is the session's
   ground-truth input and losing it would silently bias the Digester's
   failure classification.

The two existing pruning benchmarks
(``test_context_pruning_benchmark.py`` #618,
``test_token_aware_pruning_benchmark.py`` #732) assert only that
``context_pruned`` fires and ``outcome.status == "success"``.  They never
check that the surviving trace is interpretable after pruning -- that the
model had enough context to produce a coherent next step.  A regression
that widened ``_PRESERVE_KINDS`` to include ``tool_call`` or narrowed it
to drop ``user_prompt`` would silently corrupt the trace store with no
benchmark catching it.  This task closes that gap (ADR-0004 -- Critic
gate gains a regression target for trace integrity; ADR-0005 --
pytest-based regression target).
"""

from __future__ import annotations

import asyncio
import json
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
from harness.hooks.context_pruning import _SqlitePruner

TASK = BenchmarkTask(
    name="pruning_preserves_trace_integrity",
    description=(
        "Drive Runner.run_task with ContextPruningHook and multi-step stub "
        "adapter; assert every tool_call has a surviving tool_result and "
        "user_prompt events are never lost after aggressive pruning (issue #1285)."
    ),
    tags=["agent-loop", "context-pruning", "trace-integrity"],
    difficulty_tier="easy",
)

# Hook fires when the session's event count exceeds ``_THRESHOLD``.
# The plant interleaves three slots per 3-event cycle:
# ``tool_result`` (preserved, orphaned-survivor check),
# ``user_prompt`` (preserved, loss check), and
# ``model_request`` (droppable noise, pruned).
#
# With ``_PLANTED=72`` that yields 24 tool_result, 24 user_prompt, and 24
# model_request events.  Combined with runner-emitted events (~10-15 per
# step), the session will exceed _THRESHOLD=50 and trigger pruning on the
# first pre_tool call.
_THRESHOLD = 50
_PLANTED = 72
_NOISE_KIND = "model_request"
_NOISE_MARKER = "noise"
_PRESERVE_MARKER = "preserve"


class _MultiStepAdapter:
    """Stub ``ModelAdapter`` that emits three tool_calls then a final answer.

    Three steps are enough to exercise multiple pruning cycles and ensure
    the trace integrity invariants are checked across the full session.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):
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
            )
        if self.calls == 4:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
            )
        raise RuntimeError(
            f"_MultiStepAdapter exhausted after 4 scripted responses; loop called "
            f"complete() {self.calls} times"
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


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text(
        "stub harness for pruning_preserves_trace_integrity\n"
    )
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


def _plant(logger: TraceLogger, session_id: str, n: int) -> None:
    """Plant ``n`` interleaved preserved + noise events on ``session_id``.

    Every 3-event cycle carries one ``tool_result`` (preserved),
    one ``user_prompt`` (preserved), and one ``model_request`` (droppable
    noise).  The preserved events occupy the OLDEST indices on purpose: a
    pruner that ignored ``_PRESERVE_KINDS`` would delete them first
    (oldest-first deletion order), so this layout makes the preservation
    assertion a genuine regression target.
    """
    for i in range(n):
        role = i % 3
        if role == 0:
            kind = "tool_result"
            payload: dict[str, object] = {"index": i, "marker": _PRESERVE_MARKER}
        elif role == 1:
            kind = "user_prompt"
            payload = {"index": i, "marker": _PRESERVE_MARKER}
        else:
            kind = _NOISE_KIND
            payload = {"index": i, "marker": _NOISE_MARKER}
        logger.record(session_id, kind=kind, payload=payload)


def _install_on_error_tracker(tracker):
    """Install ``tracker`` on the default ``HookRegistry`` for the test."""
    from harness.hooks import get_registry
    from harness.hooks.base import reset_default_registry

    reset_default_registry()
    registry = get_registry()
    registry._on_error = tracker  # type: ignore[assignment]
    return registry


@pytest.mark.benchmark
def test_pruning_preserves_trace_integrity(benchmark_workspace: Path) -> None:
    """Validate trace integrity invariants after aggressive context pruning.

    Drives ``Runner.run_task`` with a multi-step stub ``ModelAdapter`` (three
    tool_calls then final answer) and ``ContextPruningHook`` registered on the
    default registry.  Before the run, plants ``_PLANTED`` interleaved
    tool_result / user_prompt / model_request events into the trace database
    so the hook's first ``pre_tool`` call fires pruning.

    Asserts the three acceptance criteria from issue #1285:

    1. Every ``tool_call`` event emitted by the Runner has a surviving
       ``tool_result`` with the same ``call_id``.
    2. All planted ``user_prompt`` events survive pruning.
    3. ``context_pruned`` fires and ``outcome.status == "success"``.
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    try:
        adapter = _MultiStepAdapter()
        _pruner = _SqlitePruner(db)

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
                    pruner=_pruner.prune,
                    tracer=_tracer,
                )
                registry.register(hook)

                await run_task(
                    "pruning-preserves-trace-integrity",
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
        assert len(prune_events) >= 1, (
            f"expected at least 1 context_pruned event; got {len(prune_events)}"
        )
        dropped = prune_events[0].payload["dropped"]

        # Criterion 3: pruning actually fired.
        assert dropped >= 1, f"expected dropped >= 1 (hook must fire); got {dropped!r}"

        # --- Criterion 1: tool_call / tool_result integrity ---------------
        # Collect all tool_call call_ids emitted by the Runner (not planted).
        runner_tool_calls = {
            e.payload["call_id"]
            for e in events
            if e.kind == "tool_call" and "planted" not in str(e.payload.get("marker", ""))
        }
        # Collect all surviving tool_result call_ids (only real ones have call_id;
        # planted synthetic ones do not).
        surviving_results = {
            e.payload["call_id"]
            for e in events
            if e.kind == "tool_result" and "call_id" in e.payload
        }
        orphaned = runner_tool_calls - surviving_results
        assert not orphaned, (
            f"orphaned tool_call(s) without surviving tool_result: {orphaned}. "
            f"Runner emitted {len(runner_tool_calls)} tool_calls; "
            f"{len(surviving_results)} have surviving results. "
            f"This indicates pruning deleted a tool_result that its tool_call "
            f"depended on, corrupting the trace's interpretability."
        )

        # --- Criterion 2: user_prompt preservation ------------------------
        # All planted user_prompt events must survive.
        planted_up = [
            e
            for e in events
            if e.kind == "user_prompt" and e.payload.get("marker") == _PRESERVE_MARKER
        ]
        expected_planted_up = _PLANTED // 3  # 24
        assert len(planted_up) == expected_planted_up, (
            f"expected all {expected_planted_up} planted user_prompt events to "
            f"survive pruning; got {len(planted_up)}. "
            f"Lost user_prompt events would silently bias the Digester's "
            f"failure classification."
        )

        # --- outcome (sanity) ---------------------------------------------
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
        assert adapter.calls == 4, (
            f"expected exactly 4 model round-trips (3 tool_calls then "
            f"final_answer); got {adapter.calls}"
        )
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()
