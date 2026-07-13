"""Token-budget enforcement tests for issue #197.

Issue #197 wires ``FOUNDRY_TOKEN_BUDGET`` (read into
:class:`RunLimits.token_budget`) into the agent loop, which today is a dead
field — it is plumbed into :class:`RunLimits` and recorded on
``task_aborted`` only as a hint, while the per-step usage is never counted.

This module exercises the full bookkeeping path the issue mandates:

1. ``ModelResponse`` carries a ``ModelUsage`` model parsed from the
   OpenAI-compatible ``usage`` object (``tests/test_model_adapter.py``
   already covered the wire-format json ; here we cover the model's own
   defaults and the Runner-side accumulation).
2. ``run_task`` stamps each ``model_response`` payload with the running
   total plus the per-step usage, and stamps ``tokens_total`` on the
   terminal ``outcome`` event so a KPI consumer can read what was spent.
3. When the running total exceeds ``RunLimits.token_budget``, the loop
   emits a ``task_aborted`` event with ``reason="token_budget"`` and exits
   with ``outcome.status="failed"``, ``outcome.reason="token_budget"`` —
   matching the SECURITY.md "Runaway detection" vocabulary that
   ``run_with_limits`` already uses for wall-clock aborts.

The acceptance scenario from issue #197 is reproduced end-to-end with a
scripted ``ModelAdapter`` whose responses carry monotonic usage.
"""

from __future__ import annotations

import json

import pytest

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


class _ScriptedAdapter:
    """Replay a fixed sequence of ``ModelResponse``s, each carrying usage.

    A test failure here is loud: each pop records the call number so the
    trace under a regression says which script index the adapter supplied
    before going silent — exactly what the loop under test needs to
    attribute a missing ``task_aborted``.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if not self._responses:
            raise RuntimeError(
                f"_ScriptedAdapter exhausted after {self.calls - 1} call(s);"
                " the loop called complete() more times than scripted"
            )
        return self._responses.pop(0)

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if not self._responses:
            raise RuntimeError(
                f"_ScriptedAdapter exhausted after {self.calls - 1} call(s);"
                " the loop called stream() more times than scripted"
            )
        response = self._responses.pop(0)
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


def _stub_harness(harness_dir) -> None:
    """Minimal harness layout so ``run_task`` can read ``system_prompt.txt``.

    ``tests/test_execution.py`` already validates the layout invariants;
    here we just need ``run_task`` to reach the adapter loop, which means
    a non-empty system prompt + a (possibly empty) ``skills/`` directory.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "_version.txt").write_text("0.1.0-test\n", encoding="utf-8")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def _final_response(content: str = "done") -> ModelResponse:
    return ModelResponse(
        message=ModelMessage(role="assistant", content=content),
        finish_reason="stop",
    )


def _tool_call_response(call_id: str) -> ModelResponse:
    tool_call = ModelToolCall(
        id=call_id,
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
    )


# --- ModelUsage model defaults (ADR-0006 boundary) --------------------------


def test_model_usage_defaults_to_zero_on_missing_fields():
    """An OpenAI-compatible endpoint that omits usage fields (some local
    servers do) still parses into a usable object: missing → 0 rather
    than raising. ADR-0006 keeps the boundary strict on type, lenient on
    missing integers so the runner can read ``usage.total_tokens`` without
    a guard.
    """
    usage = ModelUsage.model_validate({})
    assert usage.prompt_tokens == 0
    assert usage.completion_tokens == 0
    assert usage.total_tokens == 0


def test_model_usage_rejects_negative_token_counts():
    """A negative ``total_tokens`` is a misreport from the endpoint, not a
    fact about the model — pydantic refuses it so the runner can never
    accumulate a number that goes backwards under the budget check.
    """
    with pytest.raises(ValueError):
        ModelUsage(total_tokens=-1)


# --- Adapter-side usage flow-through ---------------------------------------


@pytest.mark.asyncio
async def test_run_task_records_usage_and_running_total_on_model_response(tmp_path):
    """Each ``model_response`` carries the latest ``ModelUsage`` reported by
    the adapter and the running total so a trace inspector can attribute
    token spend to a specific turn.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    tool_call = ModelToolCall(
        id="call_step_1",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "true"}),
        ),
    )
    responses = [
        ModelResponse(
            message=ModelMessage(role="assistant", content=None, tool_calls=[tool_call]),
            tool_calls=[tool_call],
            finish_reason="tool_calls",
            usage=ModelUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        ),
        ModelResponse(
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
            usage=ModelUsage(prompt_tokens=40, completion_tokens=60, total_tokens=100),
        ),
    ]
    adapter = _ScriptedAdapter(responses)

    async def noop_executor(name, arguments):  # noqa: ANN001, ARG001
        return {"status": "ok"}

    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as session_id:
        await run_task(
            "usage-flow",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=noop_executor,
        )

    events = logger.load_session(session_id)
    model_responses = [event for event in events if event.kind == "model_response"]
    assert len(model_responses) == 2

    first, second = model_responses
    assert first.payload["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }
    assert first.payload["tokens_used"] == 30

    assert second.payload["usage"] == {
        "prompt_tokens": 40,
        "completion_tokens": 60,
        "total_tokens": 100,
    }
    assert second.payload["tokens_used"] == 130

    outcome = next(event for event in events if event.kind == "outcome")
    assert outcome.payload["tokens_total"] == 130
    assert outcome.payload["status"] == "success"


# --- Token-budget abort behavior -------------------------------------------


@pytest.mark.asyncio
async def test_run_task_aborts_when_running_total_exceeds_token_budget(tmp_path):
    """Issue #197 acceptance: a scripted adapter whose responses carry
    monotonic usage triggers a ``task_aborted`` event with
    ``reason="token_budget"`` once the running total exceeds the cap.

    Budget=150, each step's response carries ``usage.total_tokens=100`` and
    a ``tool_call`` so the loop performs a second iteration. After step 2
    the running total is 200 (> 150) → the loop terminates without taking
    a third adapter call, ``outcome.status="failed"``,
    ``outcome.reason="token_budget"``, ``outcome.tokens_total=200``.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    def _step_response(step_index: int) -> ModelResponse:
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
            usage=ModelUsage(prompt_tokens=40, completion_tokens=60, total_tokens=100),
        )

    responses = [_step_response(0), _step_response(1)]
    # Plus one stash — if the loop ignores the abort and asks for a third
    # round-trip, the adapter raises loudly so the test surfaces the
    # regression instead of papering over it.
    responses.append(
        ModelResponse(
            message=ModelMessage(role="assistant", content="unreached"),
            finish_reason=None,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        )
    )
    adapter = _ScriptedAdapter(responses)
    limits = RunLimits(token_budget=150)

    async def noop_executor(name, arguments):  # noqa: ANN001, ARG001
        return {"status": "ok"}

    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as session_id:
        await run_task(
            "budget-exceeded",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=noop_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)

    aborted = [event for event in events if event.kind == "task_aborted"]
    assert len(aborted) == 1
    assert aborted[0].payload == {
        "reason": "token_budget",
        "tokens_used": 200,
        "token_budget": 150,
    }

    outcome = next(event for event in events if event.kind == "outcome")
    assert outcome.payload["status"] == "failed"
    assert outcome.payload["reason"] == "token_budget"
    assert outcome.payload["steps"] == 2
    assert outcome.payload["tokens_total"] == 200

    # The two ``model_response`` events preceding ``task_aborted`` carry
    # the running totals 100 and 200 — observability is the whole point of
    # the per-step stamp on the event payload.
    model_responses = [event for event in events if event.kind == "model_response"]
    assert [event.payload["tokens_used"] for event in model_responses] == [100, 200]

    # The loop must NOT have asked the adapter for a third round-trip.
    assert adapter.calls == 2


@pytest.mark.asyncio
async def test_run_task_within_budget_completes_normally(tmp_path):
    """When the running total stays under the cap, the loop runs to
    completion (``final_answer``) and no ``task_aborted`` is emitted.

    This is the regression guard for the budget check: a bug that
    aborts on the first response that *carries* usage (rather than the
    first response that *exceeds* the cap) would surface here as a
    spurious ``task_aborted``.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    responses = [
        ModelResponse(
            message=ModelMessage(role="assistant", content="step one"),
            finish_reason=None,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        ),
        _final_response("all done"),
    ]
    adapter = _ScriptedAdapter(responses)
    limits = RunLimits(token_budget=10_000)

    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as session_id:
        await run_task(
            "within-budget",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            limits=limits,
        )

    events = logger.load_session(session_id)
    kinds = [event.kind for event in events]
    assert "task_aborted" not in kinds

    outcome = next(event for event in events if event.kind == "outcome")
    assert outcome.payload["status"] == "success"
    assert outcome.payload["reason"] == "final_answer"
    assert outcome.payload["tokens_total"] == 20


@pytest.mark.asyncio
async def test_run_task_unset_budget_does_not_enforce(tmp_path):
    """``RunLimits(token_budget=None)`` (and the default ``RunLimits()``
    with no kwargs) leaves the loop permissive: usage is still recorded
    on every ``model_response`` and the terminal ``outcome``, but
    ``task_aborted`` is never emitted for the budget reason.

    This pins ADR-0006 discipline: ``None`` means "no enforcement", not
    "enforce zero tokens" — a value of ``0`` would abort on the first
    response with usage > 0, which is the opposite of what the env
    contract promises.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    responses = [
        ModelResponse(
            message=ModelMessage(role="assistant", content="hi"),
            finish_reason=None,
            usage=ModelUsage(prompt_tokens=10, completion_tokens=10, total_tokens=20),
        ),
        _final_response("done"),
    ]
    adapter = _ScriptedAdapter(responses)

    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as session_id:
        await run_task(
            "no-budget-cap",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            limits=None,
        )

    events = logger.load_session(session_id)
    kinds = [event.kind for event in events]
    assert "task_aborted" not in kinds

    outcome = next(event for event in events if event.kind == "outcome")
    assert outcome.payload["status"] == "success"
    assert outcome.payload["tokens_total"] == 20
