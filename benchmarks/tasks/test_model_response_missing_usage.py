"""Benchmark task: model_response with usage=None path coverage (issue #1264).

When a model endpoint omits the wire-format ``usage`` object in its response,
``Runner.run_task`` emits a ``token_usage_missing`` event (issue #580) and
counts zero tokens for that step. The ``model_response`` event carries
``token_usage: null``.

This path is tested in ``tests/test_execution_token_budget.py`` but has no
benchmark coverage. A regression that silently drops ``token_usage_missing``
events would not be caught by the benchmark suite.

This task drives ``Runner.run_task`` directly against a stub ``ModelAdapter``
that emits ``usage=None`` on the first step then valid usage on subsequent
steps, and asserts:

1. A ``token_usage_missing`` event is recorded for the first step.
2. The ``model_response`` for that step carries ``token_usage: null``.
3. ``tokens_used`` is not incremented by the zero-token step.
4. Token budget abort fires correctly when subsequent steps exceed the budget.

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
    name="model_response_missing_usage",
    description=(
        "Drive Runner.run_task with a stub ModelAdapter whose first response "
        "carries usage=None (triggers token_usage_missing); assert that event "
        "is recorded, model_response.token_usage is null, tokens_used is not "
        "inflated, and budget abort still fires correctly across subsequent "
        "steps with valid usage (issue #1264)."
    ),
    tags=["token-budget"],
    difficulty_tier="easy",
    token_budget=_TASK_TOKEN_BUDGET,
)


class _MissingUsageAdapter:
    """Stub ``ModelAdapter`` that replays scripted responses with missing usage.

    Step 1: ``usage=None`` — triggers ``token_usage_missing``, zero tokens counted.
    Step 2: ``usage.total_tokens=100`` — counted normally.
    Step 3: ``usage.total_tokens=100`` — running total 200 (> 150 budget), aborts.

    A fourth response is stashed — if the loop ignores the abort and asks for
    another round-trip, the adapter raises loudly so the test surfaces the
    regression instead of silently passing.
    """

    def __init__(self) -> None:
        self.calls = 0

    def _step_response(
        self, step_index: int, total_tokens: int | None, content: str | None = None
    ) -> ModelResponse:
        tool_call = ModelToolCall(
            id=f"call_step_{step_index}",
            type="function",
            function=ToolCallFunction(
                name="bash",
                arguments=json.dumps({"command": "true"}),
            ),
        )
        return ModelResponse(
            message=ModelMessage(role="assistant", content=content, tool_calls=[tool_call]),
            tool_calls=[tool_call],
            finish_reason="tool_calls" if content is None else "stop",
            usage=ModelUsage(prompt_tokens=40, completion_tokens=60, total_tokens=total_tokens)
            if total_tokens is not None
            else None,
        )

    async def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            return self._step_response(0, None)
        if self.calls == 2:
            return self._step_response(1, 100)
        if self.calls == 3:
            return self._step_response(2, 100)
        raise RuntimeError(
            f"_MissingUsageAdapter exhausted after {self.calls - 1} scripted responses; "
            f"loop called complete() {self.calls} times (possible runaway loop)"
        )

    async def chat(self, messages, tools=None, **kwargs):
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):
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
    (harness_dir / "system_prompt.txt").write_text(
        "stub harness for model_response_missing_usage\n"
    )
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


@pytest.mark.benchmark
def test_model_response_missing_usage(benchmark_workspace: Path) -> None:
    """Benchmark: model_response with usage=None path coverage (issue #1264).

    Drives ``Runner.run_task`` with a stub ``ModelAdapter`` where:
    - Step 1: ``usage=None`` → triggers ``token_usage_missing``, zero tokens counted
    - Step 2: ``usage.total_tokens=100`` → running total 100
    - Step 3: ``usage.total_tokens=100`` → running total 200 > 150 budget, aborts

    Asserts:
    1. ``token_usage_missing`` event is recorded for step 1.
    2. ``model_response`` for step 1 carries ``token_usage: null``.
    3. ``tokens_used`` in step 1's model_response is 0 (zero tokens counted).
    4. ``tokens_used`` in step 2's model_response is 100 (not 0 + 100 = 100).
    5. ``tokens_used`` in step 3's model_response is 200.
    6. ``task_aborted`` fires with ``reason='token_budget'``.
    7. ``outcome`` reflects ``status='failed'``, ``reason='token_budget'``.
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    adapter = _MissingUsageAdapter()
    limits = RunLimits(token_budget=_TASK_TOKEN_BUDGET)

    async def noop_executor(name: str, arguments: dict) -> dict:
        return {"status": "ok"}

    async def _drive() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as session_id:
            await run_task(
                "model-response-missing-usage",
                harness_dir,
                logger,
                session_id,
                model_adapter=adapter,
                skill_executor=noop_executor,
                limits=limits,
            )

    asyncio.run(_drive())

    logger = TraceLogger(db)
    sessions = logger.list_sessions()
    assert sessions, "expected at least one session"
    session_id = sessions[0].session_id
    events = logger.load_session(session_id)

    # --- token_usage_missing event must fire for step 1 (usage=None) --------
    missing_events = [event for event in events if event.kind == "token_usage_missing"]
    assert len(missing_events) == 1, (
        f"expected exactly 1 token_usage_missing event; got {len(missing_events)}"
    )
    assert missing_events[0].payload["step"] == 0, (
        f"expected step=0; got {missing_events[0].payload['step']}"
    )
    assert "endpoint did not report token usage" in missing_events[0].payload["message"]

    # --- model_response assertions -------------------------------------------
    model_responses = [event for event in events if event.kind == "model_response"]
    assert len(model_responses) == 3, (
        f"expected 3 model_response events; got {len(model_responses)}"
    )

    # Step 0: usage=None → token_usage is null, tokens_used is 0
    assert model_responses[0].payload["token_usage"] is None, (
        f"step 0: expected token_usage=None; got {model_responses[0].payload['token_usage']!r}"
    )
    assert model_responses[0].payload["tokens_used"] == 0, (
        f"step 0: expected tokens_used=0; got {model_responses[0].payload['tokens_used']}"
    )

    # Step 1: usage=100 → token_usage is valid, tokens_used is 100 (not 0+100)
    assert model_responses[1].payload["token_usage"] is not None, (
        f"step 1: expected token_usage is not None; got {model_responses[1].payload['token_usage']!r}"
    )
    assert model_responses[1].payload["tokens_used"] == 100, (
        f"step 1: expected tokens_used=100; got {model_responses[1].payload['tokens_used']}"
    )

    # Step 2: usage=100 → running total 200, budget exceeded
    assert model_responses[2].payload["token_usage"] is not None
    assert model_responses[2].payload["tokens_used"] == 200, (
        f"step 2: expected tokens_used=200; got {model_responses[2].payload['tokens_used']}"
    )

    # --- task_aborted must fire ----------------------------------------------
    aborted = [event for event in events if event.kind == "task_aborted"]
    assert len(aborted) == 1, f"expected exactly 1 task_aborted event; got {len(aborted)}"
    assert aborted[0].payload["reason"] == "token_budget"
    assert aborted[0].payload["tokens_used"] == 200
    assert aborted[0].payload["token_budget"] == _TASK_TOKEN_BUDGET

    # --- outcome reflects abort ----------------------------------------------
    outcome = next(event for event in events if event.kind == "outcome")
    assert outcome.payload["status"] == "failed", (
        f"expected outcome.status='failed'; got {outcome.payload!r}"
    )
    assert outcome.payload["reason"] == "token_budget", (
        f"expected outcome.reason='token_budget'; got {outcome.payload!r}"
    )
    assert outcome.payload["steps"] == 3, (
        f"expected outcome.steps=3; got {outcome.payload['steps']}"
    )
    assert outcome.payload["tokens_total"] == 200, (
        f"expected outcome.tokens_total=200; got {outcome.payload['tokens_total']}"
    )

    # --- Adapter exhaustion guard --------------------------------------------
    assert adapter.calls == 3, (
        f"expected exactly 3 model round-trips before abort; got {adapter.calls}"
    )
