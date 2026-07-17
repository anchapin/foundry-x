"""Benchmark task: Runner-driven token budget enforcement (issue #566).

This task verifies that ``Runner.run_task`` correctly aborts a task with
``task_aborted(reason='token_budget')`` when the running token total exceeds
the cap set via ``RunLimits.token_budget``.

The task drives ``Runner.run_task`` directly against a stub ``ModelAdapter``
that emits responses with scripted ``usage.total_tokens`` values designed to
exceed the configured ``token_budget``, then asserts:

1. A ``task_aborted`` event with ``reason='token_budget'`` is recorded.
2. The terminal ``outcome`` event reports ``status='failed'`` and
   ``reason='token_budget'``.
3. The ``tokens_used`` in the abort event equals or exceeds the budget.

It is registered as a :class:`BenchmarkTask` so the in-process registry
enumerates it alongside every other benchmark, and tagged with
``token-budget`` so the Critic (ADR-0004) can select it specifically.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

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
from foundry_x.execution.runner import RunLimits, run_task
from foundry_x.trace.logger import TraceLogger


_TASK_TOKEN_BUDGET = 150

TASK = BenchmarkTask(
    name="token_budget_enforcement",
    description=(
        f"Drive Runner.run_task with a stub ModelAdapter whose responses carry "
        f"usage.total_tokens values designed to exceed the configured token_budget "
        f"of {_TASK_TOKEN_BUDGET}; assert that the Runner emits task_aborted with "
        f"reason='token_budget' and outcome status='failed', reason='token_budget' "
        f"(issue #566)."
    ),
    tags=["token-budget"],
    difficulty_tier="easy",
    token_budget=_TASK_TOKEN_BUDGET,
)


class _ScriptedTokenBudgetAdapter:
    """Stub ``ModelAdapter`` that replays scripted responses with usage.

    The first two responses carry ``usage.total_tokens=100`` (each) and a
    ``tool_call`` so the loop iterates. After two steps the running total is
    200, which exceeds the budget of 150, triggering the abort.

    A third response is stashed -- if the loop ignores the abort and asks for
    another round-trip, the adapter raises loudly so the test surfaces the
    regression instead of silently passing.
    """

    def __init__(self) -> None:
        self.calls = 0

    def _step_response(self, step_index: int, total_tokens: int) -> ModelResponse:
        tool_call = ModelToolCall(
            id=f"call_step_{step_index}",
            type="function",
            function=ToolCallFunction(
                name="bash",
                arguments=json.dumps({"command": "true"}),
            ),
        )
        return ModelResponse(
            message=ModelMessage(role="assistant", content=None, tool_calls=[tool_call]),
            tool_calls=[tool_call],
            finish_reason="tool_calls",
            usage=ModelUsage(prompt_tokens=40, completion_tokens=60, total_tokens=total_tokens),
        )

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls == 1:
            return self._step_response(0, 100)
        if self.calls == 2:
            return self._step_response(1, 100)
        raise RuntimeError(
            f"_ScriptedTokenBudgetAdapter exhausted after {self.calls - 1} scripted responses; "
            f"loop called complete() {self.calls} times (possible runaway loop)"
        )

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        response = await self.complete(messages, tools, **kwargs)
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
    (harness_dir / "system_prompt.txt").write_text("stub harness for token_budget_enforcement\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


@pytest.mark.benchmark
def test_token_budget_enforcement(benchmark_workspace: Path) -> None:
    """Runner-driven token budget enforcement benchmark (issue #566).

    Drives ``Runner.run_task`` with a stub ``ModelAdapter`` whose responses
    carry ``usage.total_tokens=100`` per step, against a budget of 150.
    After two steps the running total is 200 (> 150), so the loop must emit
    ``task_aborted(reason='token_budget')`` and terminate with
    ``outcome.status='failed'``, ``outcome.reason='token_budget'``.
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    adapter = _ScriptedTokenBudgetAdapter()
    limits = RunLimits(token_budget=_TASK_TOKEN_BUDGET)

    async def noop_executor(name: str, arguments: dict) -> dict:
        return {"status": "ok"}

    async def _drive() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as session_id:
            await run_task(
                "token-budget-enforcement",
                harness_dir,
                logger,
                session_id,
                model_adapter=adapter,
                skill_executor=noop_executor,
                limits=limits,
            )

    asyncio.run(_drive())

    # --- Trace assertions ---------------------------------------------
    logger = TraceLogger(db)
    sessions = logger.list_sessions()
    assert sessions, "expected at least one session"
    session_id = sessions[0].session_id
    events = logger.load_session(session_id)

    # A token-budget abort must be recorded.
    aborted = [event for event in events if event.kind == "task_aborted"]
    assert len(aborted) == 1, f"expected exactly 1 task_aborted event; got {len(aborted)}"
    assert aborted[0].payload["reason"] == "token_budget", (
        f"expected reason='token_budget'; got {aborted[0].payload!r}"
    )
    # Running total after 2 steps is 200 (2 × 100).
    assert aborted[0].payload["tokens_used"] == 200, (
        f"expected tokens_used=200; got {aborted[0].payload['tokens_used']}"
    )
    assert aborted[0].payload["token_budget"] == _TASK_TOKEN_BUDGET

    # Terminal outcome must reflect the abort.
    outcome = next(event for event in events if event.kind == "outcome")
    assert outcome.payload["status"] == "failed", (
        f"expected outcome.status='failed'; got {outcome.payload!r}"
    )
    assert outcome.payload["reason"] == "token_budget", (
        f"expected outcome.reason='token_budget'; got {outcome.payload!r}"
    )
    assert outcome.payload["steps"] == 2, (
        f"expected outcome.steps=2; got {outcome.payload['steps']}"
    )
    assert outcome.payload["tokens_total"] == 200, (
        f"expected outcome.tokens_total=200; got {outcome.payload['tokens_total']}"
    )

    # The two model_response events preceding task_aborted carry the running
    # totals 100 and 200 -- observability is verified.
    model_responses = [event for event in events if event.kind == "model_response"]
    assert [event.payload["tokens_used"] for event in model_responses] == [100, 200], (
        "model_response events must carry ascending running totals"
    )

    # The loop must NOT have asked the adapter for a third round-trip.
    assert adapter.calls == 2, (
        f"expected exactly 2 model round-trips before abort; got {adapter.calls}"
    )
