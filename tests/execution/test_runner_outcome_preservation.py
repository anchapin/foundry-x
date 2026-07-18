"""Regression tests for issue #791 — outcome_reason preservation.

The runner's ``finally`` block records the terminal ``outcome`` event. Every
abort path (``event_limit``, ``token_budget``, ``model_error``, ``max_steps``)
sets ``outcome_status`` and ``outcome_reason`` to specific values before
breaking out of the loop. The contract being verified here is that those
abort-path values are **preserved** when the outcome event is recorded, and
the runner does not silently reset them to ``"success"`` / ``"final_answer"``.

Acceptance criteria from issue #791:

- When ``max_events_per_session`` is exceeded mid-session, ``outcome.reason``
  must be ``"event_limit"`` (not ``"final_answer"``).
- When ``token_budget`` is exceeded mid-session, ``outcome.reason`` must be
  ``"token_budget"`` (not ``"final_answer"``).

These tests complement the existing per-issue tests
(``test_runner_event_limit.py`` for #708, ``test_execution_token_budget.py``
for #197) by exercising additional abort positions so a regression in any
one of them surfaces here rather than as a downstream KPI surprise.
"""

from __future__ import annotations

import json
from pathlib import Path

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


def _stub_harness(harness_dir: Path) -> None:
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def _tool_call_response(call_id: str, total_tokens: int | None = None) -> ModelResponse:
    tool_call = ModelToolCall(
        id=call_id,
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "true"}),
        ),
    )
    usage = (
        ModelUsage(
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens - total_tokens // 2,
            total_tokens=total_tokens,
        )
        if total_tokens is not None
        else None
    )
    return ModelResponse(
        message=ModelMessage(role="assistant", content=None, tool_calls=[tool_call]),
        tool_calls=[tool_call],
        finish_reason="tool_calls",
        usage=usage,
    )


def _final_response(total_tokens: int | None = None) -> ModelResponse:
    usage = (
        ModelUsage(
            prompt_tokens=total_tokens // 2,
            completion_tokens=total_tokens - total_tokens // 2,
            total_tokens=total_tokens,
        )
        if total_tokens is not None
        else None
    )
    return ModelResponse(
        message=ModelMessage(role="assistant", content="done"),
        finish_reason="stop",
        usage=usage,
    )


class _StreamingScriptedAdapter:
    """Adapter that yields chunks matching a sequence of ``ModelResponse``s.

    Each ``ModelResponse`` produces 1+ chunks: a tool_call chunk (if any),
    a finish_reason chunk, and an optional usage chunk. ``chunk_count`` is
    therefore the total number of SSE deltas, which the runner adds to
    ``event_count`` and then checks against ``max_events_per_session``.
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if not self._responses:
            raise RuntimeError(
                f"_StreamingScriptedAdapter exhausted after {self.calls - 1} call(s)"
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
                        type="function",
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

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        raise AssertionError("run_task must call stream()")

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        raise AssertionError("run_task must call stream()")


async def _noop_executor(name: str, arguments: dict) -> dict:  # noqa: ANN001, ARG001
    return {"status": "ok"}


def _outcome(events: list) -> dict:
    matches = [e for e in events if e.kind == "outcome"]
    assert len(matches) == 1, f"expected 1 outcome event, got {len(matches)}"
    return matches[0].payload


# --- event_limit abort paths ------------------------------------------------


@pytest.mark.asyncio
async def test_event_limit_after_model_request_preserves_outcome_reason(tmp_path):
    """When ``max_events_per_session`` is exceeded right after the
    ``model_request`` event is recorded (before any streaming), the abort
    path at ``runner.py:1597-1601`` sets ``outcome.reason="event_limit"``.
    The ``finally`` block must record this verbatim — not reset it to
    ``"final_answer"``."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    # Trace event accounting (using _record_and_count = +1 per recorded event):
    #   user_prompt → 1
    #   model_request → 2
    # So with max_events=2, the _check_event_limit() at the start of the
    # iteration fires, hits the early abort, and never streams a response.
    responses = [_tool_call_response("call_1"), _final_response()]
    adapter = _StreamingScriptedAdapter(responses)
    limits = RunLimits(max_events_per_session=2)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await run_task(
            "issue-791-event-limit-model-request",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=_noop_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 1
    assert aborted[0].payload["reason"] == "event_limit"

    outcome = _outcome(events)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "event_limit"
    # The adapter should not have been asked for a stream() — the abort
    # happens before _consume_model_stream is reached.
    assert adapter.calls == 0


@pytest.mark.asyncio
async def test_event_limit_after_streaming_chunks_preserves_outcome_reason(tmp_path):
    """When the cap is exceeded after streaming chunks are counted (the
    ``event_count += chunk_count`` path at ``runner.py:1624-1629``), the
    abort path sets ``outcome.reason="event_limit"``. The ``finally`` block
    must record this verbatim."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    # Stream yields 2 chunks (tool_call + finish_reason). With max_events=3:
    #   user_prompt → 1
    #   model_request → 2
    #   event_count += chunk_count(2) → 4  → triggers _check_event_limit
    responses = [_tool_call_response("call_1"), _final_response()]
    adapter = _StreamingScriptedAdapter(responses)
    limits = RunLimits(max_events_per_session=3)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await run_task(
            "issue-791-event-limit-chunks",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=_noop_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 1
    assert aborted[0].payload["reason"] == "event_limit"

    outcome = _outcome(events)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "event_limit"


@pytest.mark.asyncio
async def test_event_limit_after_tool_call_event_preserves_outcome_reason(tmp_path):
    """When the cap is exceeded after a ``tool_call`` event is recorded
    (the inner check at ``runner.py:1735-1739``), ``outcome.reason``
    must be ``"event_limit"``."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    # Tool-call turn emits 2 stream chunks. With max_events=6:
    #   user_prompt → 1
    #   model_request → 2
    #   event_count += chunk_count(2) → 4
    #   model_response → 5
    #   tool_call → 6  → triggers _check_event_limit
    responses = [_tool_call_response("call_1"), _final_response()]
    adapter = _StreamingScriptedAdapter(responses)
    limits = RunLimits(max_events_per_session=6)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await run_task(
            "issue-791-event-limit-tool-call",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=_noop_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 1
    assert aborted[0].payload["reason"] == "event_limit"

    outcome = _outcome(events)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "event_limit"


# --- token_budget abort path ------------------------------------------------


@pytest.mark.asyncio
async def test_token_budget_preserves_outcome_reason(tmp_path):
    """When ``RunLimits.token_budget`` is exceeded after ``model_response``,
    ``outcome.reason`` must be ``"token_budget"`` — not the default
    ``"final_answer"``."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    # Two tool-call turns, each reporting 100 tokens; budget=150. After step 2
    # the running total is 200 (> 150) → abort with token_budget.
    responses = [
        _tool_call_response("call_1", total_tokens=100),
        _tool_call_response("call_2", total_tokens=100),
        _final_response(total_tokens=10),  # unused — stash for safety
    ]
    adapter = _StreamingScriptedAdapter(responses)
    limits = RunLimits(token_budget=150)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await run_task(
            "issue-791-token-budget",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=_noop_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 1
    assert aborted[0].payload["reason"] == "token_budget"

    outcome = _outcome(events)
    assert outcome["status"] == "failed"
    assert outcome["reason"] == "token_budget"
    assert outcome["tokens_total"] == 200
    # Loop must NOT have made a third round-trip after the abort.
    assert adapter.calls == 2


# --- max_steps abort path ---------------------------------------------------


@pytest.mark.asyncio
async def test_max_steps_preserves_outcome_reason(tmp_path, monkeypatch):
    """When ``max_steps`` is reached at the end of a tool-call iteration
    (``runner.py:1801-1804``), ``outcome.reason`` must be ``"max_steps"``
    — not ``"final_answer"``."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    # Default max_steps=16, but we override via FOUNDRY_MAX_AGENT_STEPS.
    # With max_steps=1, the first step performs a tool_call. At the end of
    # step 0, step+1 >= max_steps (1 >= 1) AND response.tool_calls is set,
    # so outcome.status="truncated" and outcome.reason="max_steps".
    monkeypatch.setenv("FOUNDRY_MAX_AGENT_STEPS", "1")

    responses = [
        _tool_call_response("call_1"),
        _final_response(),  # unreachable
    ]
    adapter = _StreamingScriptedAdapter(responses)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await run_task(
            "issue-791-max-steps",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=_noop_executor,
        )
    events = logger.load_session(session_id)

    outcome = _outcome(events)
    assert outcome["status"] == "truncated"
    assert outcome["reason"] == "max_steps"


# --- default path (no abort) -----------------------------------------------


@pytest.mark.asyncio
async def test_default_outcome_when_no_abort_fires(tmp_path):
    """When no abort path fires (model returns a final answer with finish_reason
    'stop' and no tool_calls), the ``finally`` block applies the default
    ``outcome.status='success'`` and ``outcome.reason='final_answer'``.

    This guards against the refactor that moved defaults out of the
    pre-try-block: they must still be applied at the very end so a normal
    session has the right terminal event."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    responses = [_final_response()]
    adapter = _StreamingScriptedAdapter(responses)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await run_task(
            "issue-791-default",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=_noop_executor,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert aborted == []

    outcome = _outcome(events)
    assert outcome["status"] == "success"
    assert outcome["reason"] == "final_answer"


# --- abort-path invariant: outcome_reason is never silently reset ----------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "limits,expected_reason,expected_status",
    [
        (RunLimits(max_events_per_session=2), "event_limit", "failed"),
        (RunLimits(max_events_per_session=3), "event_limit", "failed"),
        (RunLimits(token_budget=10), "token_budget", "failed"),
    ],
)
async def test_outcome_reason_preserved_from_abort_path(
    tmp_path, limits, expected_reason, expected_status
):
    """Single parametrised invariant: regardless of which abort fires, the
    outcome event's ``reason`` field matches the abort reason. This is the
    issue #791 contract: ``outcome.reason`` MUST be the abort reason, never
    the default ``"final_answer"``."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    if limits.token_budget is not None:
        # Token-budget abort: stream two tool-call turns that together exceed
        # the budget. Each turn reports 8 tokens; budget=10 means the second
        # turn pushes the running total to 16 (> 10) → abort.
        responses = [
            _tool_call_response("call_1", total_tokens=8),
            _tool_call_response("call_2", total_tokens=8),
        ]
    else:
        responses = [_tool_call_response("call_1"), _final_response()]
    adapter = _StreamingScriptedAdapter(responses)

    label = (
        f"max-events-{limits.max_events_per_session}"
        if limits.max_events_per_session is not None
        else f"token-budget-{limits.token_budget}"
    )

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await run_task(
            f"issue-791-invariant-{label}",
            harness_dir,
            logger,
            session_id,
            model_adapter=adapter,
            skill_executor=_noop_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 1
    assert aborted[0].payload["reason"] == expected_reason

    outcome = _outcome(events)
    assert outcome["reason"] != "final_answer", (
        f"BUG: outcome.reason reset to default 'final_answer' for "
        f"abort={expected_reason} limits={limits}"
    )
    assert outcome["reason"] == expected_reason
    assert outcome["status"] == expected_status
