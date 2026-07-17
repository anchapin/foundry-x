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
        from harness.hooks.base import reset_default_registry
        from harness.hooks.injection_firewall import InjectionFirewallHook
        from harness.hooks import register_hook
    except ImportError:
        return
    reset_default_registry()
    register_hook(InjectionFirewallHook())


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
    assert user_prompt_idx < tool_call_idx < tool_result_idx < outcome_idx, (
        f"events out of order: {kinds!r}"
    )

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
    # Issue #258: bash skill now uses real subprocess-backed executor
    assert "hello" in tool_result_event.payload["output"]["stdout"]
    assert tool_result_event.payload["output"]["exit_code"] == 0
    assert tool_result_event.payload["output"]["truncated"] is False

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
        assert req_idx >= 0 and ci < resp_idx, (
            f"chunk event {ci} not bracketed by request {req_idx} and response {resp_idx}"
        )
    # Each step's chunks share its model_request/model_response bracket.
    assert len(chunk_idxs) >= len(req_idxs), (
        f"expected at least one chunk per step; got {len(chunk_idxs)} chunks "
        f"for {len(req_idxs)} requests"
    )

    assert kinds.count("model_request") == 2
    assert kinds.count("model_response") == 2
    # develop emits 2 tool_call events per tool: before execution (duration_ms=0)
    # via _execute_skill and after execution with real duration (#258)
    assert kinds.count("tool_call") == 2
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
    # develop emits 2 tool_call events per tool: before and after skill execution (#258)
    # max_steps=2 means 2 steps with 1 tool each → 4 events
    assert len(tool_calls) == 4, (
        "cap at max_steps=2 must produce exactly four ``tool_call`` events; the third adapter "
        "call should not happen because the loop exits before it"
    )


def test_agent_loop_dynamic_max_steps_re_reads_env_on_each_iteration(tmp_path, monkeypatch):
    """When FOUNDRY_MAX_AGENT_STEPS_DYNAMIC=1, the step cap is re-evaluated
    on each loop iteration so operators can adjust the bound mid-session without
    restarting the process (useful for integration tests that verify the
    max_steps boundary).
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

    responses = [_tool_response(f"call_step_{i}") for i in range(10)]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("runaway", db, REPO_HARNESS_DIR))
    monkeypatch.setenv("FOUNDRY_MAX_AGENT_STEPS", "10")
    monkeypatch.setenv("FOUNDRY_MAX_AGENT_STEPS_DYNAMIC", "1")

    call_count = {"n": 0}
    original_resolve_max_steps = runner_mod._resolve_max_steps

    def _counting_resolve_max_steps(env=None):
        call_count["n"] += 1
        return original_resolve_max_steps(env)

    monkeypatch.setattr(runner_mod, "_resolve_max_steps", _counting_resolve_max_steps)

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "truncated"
    assert outcome_event.payload["reason"] == "max_steps"
    assert outcome_event.payload["steps"] == 10

    assert call_count["n"] >= 10, (
        "with dynamic mode, _resolve_max_steps must be called at least once per iteration"
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


def test_agent_loop_records_parse_error_for_malformed_arguments(tmp_path, monkeypatch):
    """Issue #261 acceptance: when a model emits malformed tool-call
    arguments, the runner records a ``tool_argument_parse_error`` trace
    event carrying the raw string and error message, yet the tool call
    still proceeds with empty arguments (resilience contract, ADR-0010).

    The trace event is purely additive — the ``tool_call`` and
    ``tool_result`` events still fire with coerced empty arguments so the
    Digester can correlate the failure to the step that produced it.
    """
    db = tmp_path / "traces.db"
    malformed = '{"command": "echo oops"'  # truncated JSON — no closing brace
    tool_call = ModelToolCall(
        id="call_bad_args",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=malformed,
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
            message=ModelMessage(role="assistant", content="recovered"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("malformed-args-loop", db, REPO_HARNESS_DIR))

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    kinds = [event.kind for event in events]

    # The parse-error event exists and precedes the corresponding tool_call.
    assert "tool_argument_parse_error" in kinds, f"missing parse-error event: {kinds!r}"
    err_idx = kinds.index("tool_argument_parse_error")
    tool_call_idx = kinds.index("tool_call")
    assert err_idx < tool_call_idx, (
        f"parse-error must precede tool_call: err={err_idx}, call={tool_call_idx}"
    )

    parse_error_event = events[err_idx]
    assert parse_error_event.payload["call_id"] == "call_bad_args"
    assert parse_error_event.payload["name"] == "bash"
    assert parse_error_event.payload["step"] == 0
    assert parse_error_event.payload["raw"] == malformed
    assert "JSONDecodeError" in parse_error_event.payload["error"]

    # No behavior change: the tool_call still fires with coerced empty args.
    tool_call_event = events[tool_call_idx]
    assert tool_call_event.payload["arguments"] == {}
    assert tool_call_event.payload["call_id"] == "call_bad_args"

    # The tool_result and outcome still fire — the loop did not abort.
    assert "tool_result" in kinds
    assert kinds[-1] == "task_completed"
    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "success"


def test_agent_loop_records_parse_error_for_non_dict_arguments(tmp_path, monkeypatch):
    """Issue #261: a JSON value that parses but is not an object (e.g. an
    array) is also reported via ``tool_argument_parse_error`` while the
    call proceeds with empty arguments.
    """
    db = tmp_path / "traces.db"
    tool_call = ModelToolCall(
        id="call_array_args",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments='["echo", "array"]',
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
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("array-args-loop", db, REPO_HARNESS_DIR))

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    parse_error_event = next(event for event in events if event.kind == "tool_argument_parse_error")
    assert parse_error_event.payload["call_id"] == "call_array_args"
    assert "expected JSON object" in parse_error_event.payload["error"]
    tool_call_event = next(event for event in events if event.kind == "tool_call")
    assert tool_call_event.payload["arguments"] == {}


def test_agent_loop_run_pre_argument_mutation(tmp_path, monkeypatch):
    """Issue #615: a pre-tool hook that mutates ``call.arguments`` must have
    the mutated value reach the skill executor. ``run_task`` calls
    ``registry.run_pre(call)`` and then passes the returned (possibly
    modified) call to ``_execute_skill`` — no test exercised this contract
    until now.

    Issue #739: the ``tool_call`` trace event must also record the mutated
    ``call.arguments`` so KPI pipelines that consume the trace see the actual
    arguments the skill received (not the raw model output).
    """
    from harness.hooks.base import ToolCall

    db = tmp_path / "traces.db"
    received_arguments: dict = {}

    async def capturing_executor(name, arguments):
        received_arguments[name] = arguments
        return {"stdout": "", "stderr": "", "exit_code": 0, "truncated": False}

    class DoublerHook:
        async def pre_tool(self, call: ToolCall) -> ToolCall:
            if "x" in call.arguments and isinstance(call.arguments["x"], (int, float)):
                call.arguments["x"] = call.arguments["x"] * 2
            return call

        async def post_tool(self, call, result):
            return result

    tool_call = ModelToolCall(
        id="call_dbl",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments='{"command": "echo test", "x": 21}',
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
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("mutate-loop", db, REPO_HARNESS_DIR))

    from harness.hooks import register_hook

    hook = DoublerHook()
    register_hook(hook)

    async def _run_with_executor(task, harness_dir, log, session_id, **kwargs):
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
            skill_executor=capturing_executor,
        )

    try:
        main(run_task_fn=_run_with_executor)

        assert "bash" in received_arguments
        assert received_arguments["bash"]["x"] == 42, (
            "pre-tool hook must double the x argument before executor receives it"
        )

        events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
        tool_call_event = next(event for event in events if event.kind == "tool_call")
        assert tool_call_event.payload["arguments"]["x"] == 42, (
            "pre-tool hook must double the x argument before trace event is recorded (#739)"
        )
    finally:
        _reset_default_registry()


def test_agent_loop_run_pre_argument_reassignment_in_trace(tmp_path, monkeypatch):
    """Issue #739: when a pre-tool hook REASSIGNS ``call.arguments`` entirely
    (not just mutates it in-place), both the skill executor and the
    ``tool_call`` trace event must see the new dict.

    Regression: runner.py used the pre-hook ``arguments`` local variable instead
    of ``call.arguments``, so a hook that does ``call.arguments = {...}``
    (reassignment) would leave the trace event recording the original dict
    while the executor received the new one.
    """
    from harness.hooks.base import ToolCall

    db = tmp_path / "traces.db"
    received_arguments: dict = {}

    async def capturing_executor(name, arguments):
        received_arguments[name] = arguments
        return {"stdout": "", "stderr": "", "exit_code": 0, "truncated": False}

    class ReassignerHook:
        async def pre_tool(self, call: ToolCall) -> ToolCall:
            call.arguments = {"command": "echo reassigned", "x": 99}
            return call

        async def post_tool(self, call, result):
            return result

    tool_call = ModelToolCall(
        id="call_reassign",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments='{"command": "echo original", "x": 21}',
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
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("reassign-loop", db, REPO_HARNESS_DIR))

    from harness.hooks import register_hook

    hook = ReassignerHook()
    register_hook(hook)

    async def _run_with_executor(task, harness_dir, log, session_id, **kwargs):
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
            skill_executor=capturing_executor,
        )

    try:
        main(run_task_fn=_run_with_executor)

        assert "bash" in received_arguments
        assert received_arguments["bash"]["x"] == 99, (
            "pre-tool hook must replace arguments before executor receives them"
        )
        assert received_arguments["bash"]["command"] == "echo reassigned"

        events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
        tool_call_event = next(event for event in events if event.kind == "tool_call")
        assert tool_call_event.payload["arguments"]["x"] == 99, (
            "pre-tool hook must replace arguments before trace event is recorded (#739)"
        )
        assert tool_call_event.payload["arguments"]["command"] == "echo reassigned"
    finally:
        _reset_default_registry()


def test_agent_loop_emits_no_parse_error_for_valid_arguments(tmp_path, monkeypatch):
    """Issue #261 acceptance: when arguments parse successfully, no
    ``tool_argument_parse_error`` event is emitted — the event is purely a
    failure-mode signal.
    """
    db = tmp_path / "traces.db"
    tool_call = ModelToolCall(
        id="call_good",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments='{"command": "echo hello"}',
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
            message=ModelMessage(role="assistant", content="ok"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("good-args-loop", db, REPO_HARNESS_DIR))

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    kinds = [event.kind for event in events]
    assert "tool_argument_parse_error" not in kinds, (
        f"no parse-error event should fire for valid args: {kinds!r}"
    )
    tool_call_event = next(event for event in events if event.kind == "tool_call")
    assert tool_call_event.payload["arguments"] == {"command": "echo hello"}


def test_parse_tool_arguments_unit():
    """Unit coverage for ``_parse_tool_arguments`` (issue #261): every
    coercion path populates ``error`` and every clean parse returns
    ``error=None`` with the decoded dict.
    """
    clean = runner_mod._parse_tool_arguments('{"a": 1}')
    assert clean.arguments == {"a": 1}
    assert clean.error is None

    empty = runner_mod._parse_tool_arguments("")
    assert empty.arguments == {}
    assert empty.error is None

    bad = runner_mod._parse_tool_arguments("{not json")
    assert bad.arguments == {}
    assert "JSONDecodeError" in (bad.error or "")

    non_dict = runner_mod._parse_tool_arguments("[1, 2, 3]")
    assert non_dict.arguments == {}
    assert "expected JSON object" in (non_dict.error or "")


def test_agent_loop_records_two_tool_results_for_parallel_tool_calls(tmp_path, monkeypatch):
    """Issue #611 acceptance: when a single ModelResponse carries two
    ``tool_calls``, the loop records two distinct ``tool_result`` events
    with different ``call_id`` values, then terminates with
    ``outcome.status='success'`` and ``outcome.reason='final_answer'``
    after processing both.

    This guards against a regression that drops or reorders parallel tool
    results — the most common real-world agent turn shape (e.g. "read file
    X then edit file Y").
    """
    db = tmp_path / "traces.db"

    tool_call_a = ModelToolCall(
        id="call_read_1",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "echo read", "cwd": str(tmp_path)}),
        ),
    )
    tool_call_b = ModelToolCall(
        id="call_edit_2",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "echo edit", "cwd": str(tmp_path)}),
        ),
    )
    responses = [
        ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=None,
                tool_calls=[tool_call_a, tool_call_b],
            ),
            tool_calls=[tool_call_a, tool_call_b],
            finish_reason="tool_calls",
        ),
        ModelResponse(
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("parallel-tools", db, REPO_HARNESS_DIR))

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    tool_result_events = [e for e in events if e.kind == "tool_result"]

    assert len(tool_result_events) == 2, (
        f"expected exactly two tool_result events, got {len(tool_result_events)}: "
        f"{[(e.payload.get('call_id'), e.payload.get('name')) for e in tool_result_events]!r}"
    )

    call_ids = {e.payload["call_id"] for e in tool_result_events}
    assert call_ids == {"call_read_1", "call_edit_2"}, (
        f"expected distinct call_ids {{'call_read_1', 'call_edit_2'}}, got {call_ids!r}"
    )

    for tr in tool_result_events:
        assert tr.payload["error"] is None, (
            f"unexpected error for {tr.payload['call_id']}: {tr.payload['error']}"
        )

    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "success", outcome_event.payload
    assert outcome_event.payload["reason"] == "final_answer", outcome_event.payload
    assert outcome_event.payload["steps"] == 2


def test_agent_loop_emits_hook_registry_error_when_get_registry_raises(tmp_path, monkeypatch):
    """Issue #260: when ``harness.hooks.get_registry()`` raises after a
    successful lazy import, ``run_task`` must record a
    ``hook_registry_error`` trace event (with ``error_type`` and
    ``message``) and continue in degraded mode rather than silently
    disabling every hook — including the ``InjectionFirewallHook`` — for
    the whole session (AGENTS.md §2 — never silently swallow an
    exception).

    The session must still complete: a final-answer turn produces a
    ``success`` ``outcome`` event, proving the failure is a degraded-mode
    signal, not a hard stop.
    """
    import harness.hooks as harness_hooks

    db = tmp_path / "traces.db"
    responses = [
        ModelResponse(
            message=ModelMessage(role="assistant", content="degraded-ok"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("degraded-task", db, REPO_HARNESS_DIR))

    def _boom() -> None:
        raise RuntimeError("registry blew up")

    monkeypatch.setattr(harness_hooks, "get_registry", _boom)

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    kinds = [event.kind for event in events]
    assert "hook_registry_error" in kinds, (
        f"expected a hook_registry_error event when get_registry() raises; kinds={kinds!r}"
    )

    err_event = next(event for event in events if event.kind == "hook_registry_error")
    assert err_event.payload["error_type"] == "RuntimeError"
    assert err_event.payload["message"] == "registry blew up"

    # Degraded mode: the session still completes successfully.
    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "success"


@pytest.mark.parametrize("finish_reason", ["length", "content_filter"])
def test_agent_loop_truncates_on_non_stop_finish_reason_without_tool_calls(
    tmp_path, monkeypatch, finish_reason
):
    """Issue #750: when the model returns no tool_calls and finish_reason is
    not 'stop' (e.g., 'length' for context limit, 'content_filter'), the runner
    must emit outcome.status='truncated' and outcome.reason=<finish_reason>,
    not outcome.status='success' with outcome.reason='final_answer'.
    """
    db = tmp_path / "traces.db"
    responses = [
        ModelResponse(
            message=ModelMessage(role="assistant", content="partial answer before cutoff"),
            finish_reason=finish_reason,
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("simple-task", db, REPO_HARNESS_DIR))

    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "truncated", outcome_event.payload
    assert outcome_event.payload["reason"] == finish_reason, outcome_event.payload
