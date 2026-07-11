"""Integration tests for the asyncio agent loop in ``run_task`` (issue #89).

Companion to ``tests/test_execution.py`` which by design avoids mutating
``src/foundry_x/execution/runner.py``. Issue #89 / ADR-0010 lands a real
agent loop in that file (it replaces the single-turn stub), so this file
isolates the new behaviors:

* the harness :class:`ToolDefinition` surface is data-driven from
  ``harness/skills/*.json`` (issue #104, #105)
* ``run_task`` brackets each model round-trip with ``model_request`` /
  ``model_response`` trace events
* on a response carrying ``tool_calls``, each call records ``tool_call``
  and ``tool_result`` events and fans through ``HookRegistry.run_pre``
  / ``run_post`` (SECURITY.md "Prompt-input firewall")
* the loop terminates on a final assistant message, ``max_steps``, or a
  model error; every terminal path records an ``outcome`` event with
  ``status`` + ``reason`` + ``steps``

The tests drive ``main()`` (not :func:`run_task` directly) per the
acceptance criterion. The real ``harness/`` directory is used so the
runner imports ``harness.hooks`` and finds the prompt-input firewall
that is supposed to run by default — the stub skill outputs are chosen
to be benign so the firewall passes them through unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import foundry_x.execution.runner as runner_mod
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import main
from foundry_x.trace.logger import TraceLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_HARNESS_DIR = REPO_ROOT / "harness"


class _ScriptedAdapter:
    """Stub ``ModelAdapter`` that replays a fixed response sequence (issue #89).

    Each ``complete()`` call pops the next scripted response off the queue;
    tests populate the queue with whatever (``tool_calls``-bearing or
    final-answer) the loop path under test requires. Raises ``RuntimeError``
    with a descriptive message if the queue is exhausted — silent fall-through
    to a default response would mask a loop that calls the adapter too many
    times, which is the exact regression the ``max_steps`` bound guards
    against (SECURITY.md "Runaway detection").
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls: list[list[ModelMessage]] = []
        self.tool_surfaces: list[list[object]] = []

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls.append([ModelMessage.model_validate(m) for m in messages])
        self.tool_surfaces.append(list(tools) if tools else [])
        if not self._responses:
            raise RuntimeError(
                "_ScriptedAdapter exhausted; the loop called complete() more times than scripted"
            )
        return self._responses.pop(0)

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls.append([ModelMessage.model_validate(m) for m in messages])
        self.tool_surfaces.append(list(tools) if tools else [])
        if not self._responses:
            raise RuntimeError(
                "_ScriptedAdapter exhausted; the loop called stream() more times than scripted"
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


def _argv(task: str, trace_path: Path, harness_dir: Path) -> list[str]:
    return [
        "fx-runner",
        "--task",
        task,
        "--harness-dir",
        str(harness_dir),
        "--trace-path",
        str(trace_path),
    ]


def _reset_default_registry() -> None:
    """Drop every hook from the harness ``HookRegistry`` after each test.

    ``harness/hooks/__init__`` self-registers the prompt-input firewall on
    import. The integration tests below rely on the firewall passing benign
    stub outputs through unchanged; resetting between tests prevents one
    test's hook installation from leaking into the next. The fixture-style
    ``autouse`` ensures every test in this file starts from a known-empty
    registry.
    """
    try:
        from harness.hooks import reset_default_registry
    except ImportError:
        return
    reset_default_registry()


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _reset_default_registry()


def test_agent_loop_records_full_event_sequence(tmp_path, monkeypatch):
    """Issue #89 acceptance: ``main()`` driving the agent loop with a stub
    ``ModelAdapter`` that emits one ``tool_calls`` response followed by a
    final assistant message produces a trace containing
    ``user_prompt``, ``tool_call``, ``tool_result``, ``outcome`` events
    in that order, plus the bracketing ``model_request`` /
    ``model_response`` round-trips and ``task_received`` /
    ``task_completed`` lifecycle markers from ``main()``.
    """
    db = tmp_path / "traces.db"
    responses = [
        ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ModelToolCall(
                        id="call_bash_1",
                        type="function",
                        function=ToolCallFunction(
                            name="bash",
                            arguments=json.dumps({"command": "echo hello", "cwd": str(tmp_path)}),
                        ),
                    )
                ],
            ),
            tool_calls=[
                ModelToolCall(
                    id="call_bash_1",
                    type="function",
                    function=ToolCallFunction(
                        name="bash",
                        arguments=json.dumps({"command": "echo hello", "cwd": str(tmp_path)}),
                    ),
                )
            ],
            finish_reason="tool_calls",
        ),
        ModelResponse(
            message=ModelMessage(
                role="assistant",
                content="done",
            ),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("loop-task", db, REPO_HARNESS_DIR))

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    kinds = [event.kind for event in events]

    user_prompt_idx = kinds.index("user_prompt")
    tool_call_idx = kinds.index("tool_call")
    tool_result_idx = kinds.index("tool_result")
    outcome_idx = kinds.index("outcome")
    assert (
        user_prompt_idx < tool_call_idx < tool_result_idx < outcome_idx
    ), f"events out of order: {kinds!r}"

    tool_call_event = events[tool_call_idx]
    assert tool_call_event.payload["name"] == "bash"
    assert tool_call_event.payload["call_id"] == "call_bash_1"
    assert tool_call_event.payload["arguments"] == {
        "command": "echo hello",
        "cwd": str(tmp_path),
    }

    tool_result_event = events[tool_result_idx]
    assert tool_result_event.payload["name"] == "bash"
    assert tool_result_event.payload["call_id"] == "call_bash_1"
    assert tool_result_event.payload["duration_ms"] >= 0
    assert tool_result_event.payload["error"] is None
    assert tool_result_event.payload["output"]["status"] == "ok"
    assert tool_result_event.payload["output"]["skill"] == "bash"

    outcome_event = events[outcome_idx]
    assert outcome_event.payload["status"] == "success"
    assert outcome_event.payload["reason"] == "final_answer"
    assert outcome_event.payload["steps"] == 2
    assert outcome_event.payload["ttft_ms"] is not None
    assert outcome_event.payload["ttft_ms"] >= 0
    assert outcome_event.payload["tokens_total"] == 0

    # Issue #199: per-chunk trace events emitted between model_request and
    # model_response, with delta_index, content_so_far, and chunk_duration_ms.
    assert kinds.count("model_response_chunk") > 0
    chunk_events = [e for e in events if e.kind == "model_response_chunk"]
    first_chunk = chunk_events[0]
    assert first_chunk.payload["delta_index"] == 0
    assert "content_so_far" in first_chunk.payload
    assert "chunk_duration_ms" in first_chunk.payload

    # model_response gains time_to_first_token_ms and chunk_count (#199).
    model_response_events = [e for e in events if e.kind == "model_response"]
    for mr_event in model_response_events:
        assert "time_to_first_token_ms" in mr_event.payload
        assert "chunk_count" in mr_event.payload
        assert mr_event.payload["chunk_count"] > 0

    # Chunk events fall between their model_request and model_response (#199).
    req_idxs = [i for i, k in enumerate(kinds) if k == "model_request"]
    resp_idxs = [i for i, k in enumerate(kinds) if k == "model_response"]
    chunk_idxs = [i for i, k in enumerate(kinds) if k == "model_response_chunk"]
    for ci in chunk_idxs:
        req_idx = max((i for i in req_idxs if i < ci), default=-1)
        resp_idx = min((i for i in resp_idxs if i > ci), default=len(kinds))
        assert (
            req_idx >= 0 and ci < resp_idx
        ), f"chunk event {ci} not bracketed by request {req_idx} and response {resp_idx}"
    # Each step's chunks share its model_request/model_response bracket.
    assert len(chunk_idxs) >= len(req_idxs), (
        f"expected at least one chunk per step; got {len(chunk_idxs)} chunks "
        f"for {len(req_idxs)} requests"
    )

    assert kinds.count("model_request") == 2
    assert kinds.count("model_response") == 2
    assert kinds.count("tool_call") == 1
    assert kinds.count("tool_result") == 1
    assert kinds[0] == "task_received"
    assert kinds[-1] == "task_completed"


def test_agent_loop_terminates_with_truncated_outcome_on_max_steps(tmp_path, monkeypatch):
    """When the loop exhausts the step cap without a final assistant message,
    the runner records ``outcome.status="truncated"`` and
    ``outcome.reason="max_steps"`` so the Phase 2 Digester can distinguish a
    runaway session from a successful one.
    """
    db = tmp_path / "traces.db"

    def _tool_response(call_id: str) -> ModelResponse:
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=None,
                tool_calls=[
                    ModelToolCall(
                        id=call_id,
                        type="function",
                        function=ToolCallFunction(
                            name="bash",
                            arguments='{"command": "echo loop"}',
                        ),
                    )
                ],
            ),
            tool_calls=[
                ModelToolCall(
                    id=call_id,
                    type="function",
                    function=ToolCallFunction(
                        name="bash",
                        arguments='{"command": "echo loop"}',
                    ),
                )
            ],
            finish_reason="tool_calls",
        )

    responses = [_tool_response(f"call_step_{i}") for i in range(3)]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("runaway", db, REPO_HARNESS_DIR))
    monkeypatch.setenv("FOUNDRY_MAX_AGENT_STEPS", "2")

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "truncated"
    assert outcome_event.payload["reason"] == "max_steps"
    assert outcome_event.payload["steps"] == 2

    tool_calls = [event for event in events if event.kind == "tool_call"]
    assert len(tool_calls) == 2, (
        "cap at max_steps=2 must produce exactly two ``tool_call`` events; the third adapter "
        "call should not happen because the loop exits before it"
    )


def test_agent_loop_executor_errors_surface_as_failed_tool_results(tmp_path, monkeypatch):
    """A skill executor that raises still records a ``tool_result`` event
    with ``error`` populated (AGENTS.md §2 — never silently swallow). The
    loop then continues so a transient failure in one skill does not abort
    the entire agent run; the model sees the error in the tool channel and
    can branch on it.
    """
    db = tmp_path / "traces.db"

    call_count = {"n": 0}

    async def failing_executor(name, arguments):  # noqa: ANN001
        call_count["n"] += 1
        raise RuntimeError(f"boom from skill {name}")

    tool_call = ModelToolCall(
        id="call_fail",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments='{"command": "true"}',
        ),
    )
    responses = [
        ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=None,
                tool_calls=[tool_call],
            ),
            tool_calls=[tool_call],
            finish_reason="tool_calls",
        ),
        ModelResponse(
            message=ModelMessage(role="assistant", content="errored"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("errored-loop", db, REPO_HARNESS_DIR))

    async def _run_with_executor(task, harness_dir, log, session_id, **kwargs):  # noqa: ANN001
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
            skill_executor=failing_executor,
        )

    main(run_task_fn=_run_with_executor)

    assert call_count["n"] == 1, "failing executor should be called exactly once"
    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    tool_result_event = next(event for event in events if event.kind == "tool_result")
    assert tool_result_event.payload["error"] is not None
    assert "boom from skill bash" in tool_result_event.payload["error"]
    assert tool_result_event.payload["output"] is None
