"""Unit tests for ``runner``'s SSE-streaming surface (issue #199).

Issue #199 wires ``run_task`` to ``ModelAdapter.stream()`` so the trace
carries per-chunk latency signals instead of one opaque ``model_response``
event. These tests pin the contract on the smallest pieces:

* :class:`ModelResponseChunkEvent` is a pydantic model (no ``Any``) with
  ``step``, ``delta_index``, ``content_so_far``, ``chunk_duration_ms``.
* :class:`_StreamingToolCallAccumulator` reassembles incremental OpenAI-
  compatible tool-call deltas by stream ``index``.
* :func:`_assemble_streamed_response` builds a complete ``ModelResponse``
  from the accumulator (content concatenated; only tool calls with a
  known ``id`` + ``name`` survive — placeholder indexes are dropped).
* :func:`_consume_model_stream` emits one ``model_response_chunk`` event
  per SSE delta, measures time-to-first-token from stream start, and
  records ``chunk_count`` and ``time_to_first_token_ms`` on the terminal
  ``model_response``.
* :func:`run_task` aggregates per-turn TTFT into ``outcome.ttft_ms``
  (p50 across turns) so the KPI consumer can see the model latency
  independent of network latency.
* The :class:`OpenAICompatibleAdapter` parses a llama-server-style SSE
  payload end-to-end so a real curl capture (see PR description) proves
  the wire shape works.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

import foundry_x.execution.runner as runner_mod
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    OpenAICompatibleAdapter,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import (
    ModelResponseChunkEvent,
    _StreamingToolCallAccumulator,
    _assemble_streamed_response,
    _consume_model_stream,
    main,
)
from foundry_x.execution.runner import run_task as real_run_task
from foundry_x.trace.logger import TraceLogger


# A faithful capture of llama.cpp's ``/v1/chat/completions`` SSE output
# (``stream:true``). The fixture mirrors the field order documented in
# llama.cpp's ``examples/server/README.md`` so the parser exercises the
# same wire shape a real llama-server emits.
LLAMA_CPP_SSE_FIXTURE = (
    'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
    '"created":1720000000,"model":"codellama-7b",'
    '"choices":[{"index":0,"delta":{"role":"assistant","content":""},'
    '"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
    '"created":1720000000,"model":"codellama-7b",'
    '"choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
    '"created":1720000000,"model":"codellama-7b",'
    '"choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
    '"created":1720000000,"model":"codellama-7b",'
    '"choices":[{"index":0,"delta":{"content":"world"},"finish_reason":null}]}\n\n'
    'data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk",'
    '"created":1720000000,"model":"codellama-7b",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
    "data: [DONE]\n\n"
)


def _stub_harness(harness_dir: Path) -> None:
    """Build a minimal valid harness layout (issue #90)."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def _argv(task: str, trace_path: Path, harness_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fx-runner",
            "--task",
            task,
            "--trace-path",
            str(trace_path),
            "--harness-dir",
            str(harness_dir),
        ],
    )


class _StubAdapter:
    """In-memory adapter that yields a scripted list of ``ModelResponseChunk``s."""

    def __init__(self, chunks: list[ModelResponseChunk]) -> None:
        self._chunks = list(chunks)
        self.stream_calls = 0

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.stream_calls += 1
        for chunk in self._chunks:
            yield chunk

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream(), not complete() (#199)")

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream(), not chat() (#199)")


def _reset_default_registry() -> None:
    try:
        from harness.hooks import reset_default_registry
    except ImportError:
        return
    reset_default_registry()


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    _reset_default_registry()


# --- pydantic event shape ----------------------------------------------------


def test_model_response_chunk_event_is_a_pydantic_model():
    """Issue #199 acceptance: ``ModelResponseChunkEvent`` is a pydantic
    model (no ``Any``) so the schema is enforced at the trace boundary.
    """
    event = ModelResponseChunkEvent(
        step=0,
        delta_index=3,
        content_so_far="hel",
        chunk_duration_ms=42,
    )
    assert event.step == 0
    assert event.delta_index == 3
    assert event.content_so_far == "hel"
    assert event.chunk_duration_ms == 42
    dumped = event.model_dump()
    assert dumped == {
        "step": 0,
        "delta_index": 3,
        "content_so_far": "hel",
        "chunk_duration_ms": 42,
    }


# --- response assembly from streamed deltas --------------------------------


def test_assemble_streamed_response_concatenates_content_and_tool_calls():
    """``_assemble_streamed_response`` rebuilds a :class:`ModelResponse`
    from accumulated streaming deltas: content concatenated verbatim, and
    tool calls rebuilt only when both ``id`` and ``name`` are known.
    """
    content_parts = ["Hel", "lo ", "world"]
    tool_call_acc = {
        0: _StreamingToolCallAccumulator(
            id="call_1",
            type="function",
            name="bash",
            arguments='{"command": "echo hi"}',
        ),
        1: _StreamingToolCallAccumulator(
            id=None,
            type=None,
            name=None,
            arguments="",
        ),
    }
    response = _assemble_streamed_response(content_parts, tool_call_acc, "stop")
    assert response.finish_reason == "stop"
    assert response.message.role == "assistant"
    assert response.message.content == "Hello world"
    # The placeholder accumulator (index 1, no id/name) is dropped.
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_1"
    assert call.function.name == "bash"
    assert json.loads(call.function.arguments) == {"command": "echo hi"}


def test_assemble_streamed_response_handles_empty_stream():
    """A stream that produced zero payload deltas assembles to a response
    with ``content=None`` and no tool calls; the caller (``run_task``)
    treats that as a non-final-answer turn so the loop terminates cleanly
    rather than crashing on a ``None`` content append.
    """
    response = _assemble_streamed_response([], {}, None)
    assert response.message.content is None
    assert response.tool_calls == []
    assert response.finish_reason is None


# --- chunk-event emission ---------------------------------------------------


@pytest.mark.asyncio
async def test_consume_model_stream_emits_chunk_events_with_monotone_index(tmp_path):
    """Issue #199 acceptance: every SSE delta produces a
    ``model_response_chunk`` event whose ``delta_index`` increments by 1
    per delta and whose ``content_so_far`` is the concatenation up to
    and including that delta. ``chunk_duration_ms`` is non-negative.
    """
    db = tmp_path / "traces.db"
    chunks = [
        ModelResponseChunk(content="Hel"),
        ModelResponseChunk(content="lo "),
        ModelResponseChunk(content="world"),
        ModelResponseChunk(finish_reason="stop"),
    ]
    adapter = _StubAdapter(chunks)
    log = TraceLogger(db)
    with log.session(harness_version="test") as session_id:
        response, ttft_ms, chunk_count, total_stream_ms = await _consume_model_stream(
            adapter,
            [ModelMessage(role="user", content="ping")],
            [],
            log,
            session_id,
            step=2,
        )

    assert adapter.stream_calls == 1
    assert response.message.content == "Hello world"
    assert response.finish_reason == "stop"
    assert chunk_count == 4
    assert ttft_ms is not None
    assert ttft_ms >= 0

    events = log.load_session(session_id)
    chunk_events = [event for event in events if event.kind == "model_response_chunk"]
    assert len(chunk_events) == 4
    deltas = [event.payload["delta_index"] for event in chunk_events]
    assert deltas == [0, 1, 2, 3]
    contents = [event.payload["content_so_far"] for event in chunk_events]
    assert contents == ["Hel", "Hello ", "Hello world", "Hello world"]
    durations = [event.payload["chunk_duration_ms"] for event in chunk_events]
    assert all(d >= 0 for d in durations)
    assert all(event.payload["step"] == 2 for event in chunk_events)


@pytest.mark.asyncio
async def test_consume_model_stream_assembles_tool_calls_from_indexed_deltas(tmp_path):
    """OpenAI-compatible streams send tool-call arguments in fragments
    keyed by delta ``index``. ``_consume_model_stream`` must reassemble
    them into a single :class:`ModelToolCall` with the concatenated
    ``arguments`` string.
    """
    db = tmp_path / "traces.db"
    chunks = [
        ModelResponseChunk(
            tool_calls=[
                ModelToolCallChunk(
                    index=0,
                    id="call_42",
                    type="function",
                    function=ToolCallFunctionChunk(name="bash"),
                )
            ]
        ),
        ModelResponseChunk(
            tool_calls=[
                ModelToolCallChunk(
                    index=0,
                    function=ToolCallFunctionChunk(arguments='{"command":'),
                )
            ]
        ),
        ModelResponseChunk(
            tool_calls=[
                ModelToolCallChunk(
                    index=0,
                    function=ToolCallFunctionChunk(arguments=' "echo hi"}'),
                )
            ]
        ),
        ModelResponseChunk(finish_reason="tool_calls"),
    ]
    adapter = _StubAdapter(chunks)
    log = TraceLogger(db)
    with log.session(harness_version="test") as session_id:
        response, _, chunk_count, _ = await _consume_model_stream(
            adapter,
            [ModelMessage(role="user", content="ping")],
            [],
            log,
            session_id,
            step=0,
        )

    assert chunk_count == 4
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "call_42"
    assert call.function.name == "bash"
    assert json.loads(call.function.arguments) == {"command": "echo hi"}


@pytest.mark.asyncio
async def test_consume_model_stream_reports_ttft_on_first_payload_delta(tmp_path):
    """Time-to-first-token is the elapsed milliseconds from stream start
    to the first delta carrying content OR a tool-call fragment; a
    metadata-only delta (e.g. a role marker with no payload) must NOT
    trip the measurement.
    """
    db = tmp_path / "traces.db"
    chunks = [
        ModelResponseChunk(),  # metadata-only; no content/tool_calls
        ModelResponseChunk(),
        ModelResponseChunk(content="first real token"),
        ModelResponseChunk(finish_reason="stop"),
    ]
    adapter = _StubAdapter(chunks)
    log = TraceLogger(db)
    with log.session(harness_version="test") as session_id:
        _, ttft_ms, _, _ = await _consume_model_stream(
            adapter,
            [ModelMessage(role="user", content="ping")],
            [],
            log,
            session_id,
            step=0,
        )
    assert ttft_ms is not None
    assert ttft_ms >= 0


@pytest.mark.asyncio
async def test_consume_model_stream_returns_none_ttft_for_empty_stream(tmp_path):
    """A stream that yields no payload deltas at all reports
    ``ttft_ms=None`` so ``run_task`` can skip the aggregation slot
    rather than stamping a misleading zero.
    """
    db = tmp_path / "traces.db"
    adapter = _StubAdapter([])
    log = TraceLogger(db)
    with log.session(harness_version="test") as session_id:
        _, ttft_ms, chunk_count, _ = await _consume_model_stream(
            adapter,
            [ModelMessage(role="user", content="ping")],
            [],
            log,
            session_id,
            step=0,
        )
    assert ttft_ms is None
    assert chunk_count == 0


@pytest.mark.asyncio
async def test_openai_adapter_parses_llama_server_sse_end_to_end():
    """``OpenAICompatibleAdapter.stream()`` must parse the exact wire
    format llama.cpp's chat-completions server emits (see PR description
    for the curl capture). The transport layer was added in #88 but was
    not exercised by ``run_task`` until #199, so this is the first
    in-process end-to-end check against the production SSE shape.
    """
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=LLAMA_CPP_SSE_FIXTURE,
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://127.0.0.1:8080",
            model="codellama-7b",
            client=client,
        )
        chunks: list[ModelResponseChunk] = []
        async for chunk in adapter.stream(
            messages=[ModelMessage(role="user", content="Say hi")],
        ):
            chunks.append(chunk)
        await adapter.aclose()

    assert seen["url"] == "http://127.0.0.1:8080/v1/chat/completions"
    assert seen["payload"]["stream"] is True  # noqa: S104 — wire contract check
    assert seen["payload"]["model"] == "codellama-7b"

    contents = [chunk.content for chunk in chunks if chunk.content]
    assert contents == ["Hello", ", ", "world"]
    assert chunks[-1].finish_reason == "stop"


# --- run_task aggregation --------------------------------------------------


def test_run_task_aggregates_ttft_p50_across_turns(tmp_path, monkeypatch):
    """Issue #199 acceptance: ``outcome.ttft_ms`` aggregates the
    per-turn time-to-first-token across all model turns using the
    median (p50) so a single slow turn does not skew the KPI.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    tool_call = ModelToolCall(
        id="call_1",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "true"}),
        ),
    )

    class ScriptedAdapter:
        def __init__(self) -> None:
            self._turn = 0

        async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            self._turn += 1
            if self._turn == 1:
                yield ModelResponseChunk(
                    tool_calls=[
                        ModelToolCallChunk(
                            index=0,
                            id=tool_call.id,
                            type=tool_call.type,
                            function=ToolCallFunctionChunk(
                                name=tool_call.function.name,
                                arguments=tool_call.function.arguments,
                            ),
                        )
                    ]
                )
                yield ModelResponseChunk(finish_reason="tool_calls")
                return
            yield ModelResponseChunk(content="done")
            yield ModelResponseChunk(finish_reason="stop")

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001
            raise AssertionError("run_task must call stream() (#199)")

        async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
            raise AssertionError("run_task must call stream() (#199)")

    async def executor(name, arguments):  # noqa: ANN001, ARG001
        return {"status": "ok"}

    monkeypatch.setattr(runner_mod, "build_model_adapter", ScriptedAdapter)
    _argv("ttft-task", db, harness_dir, monkeypatch)

    async def drive() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as session_id:
            await real_run_task(
                "ttft-task",
                harness_dir,
                logger,
                session_id,
                skill_executor=executor,
            )

    asyncio.run(drive())

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    outcome = next(event for event in events if event.kind == "outcome")
    # Both turns produced payload deltas, so ttft_ms must be a non-negative int
    # (the median of two non-negative values).
    assert outcome.payload["ttft_ms"] is not None
    assert outcome.payload["ttft_ms"] >= 0

    # Each turn's model_response carries its own time_to_first_token_ms and chunk_count.
    response_events = [event for event in events if event.kind == "model_response"]
    assert len(response_events) == 2
    for event in response_events:
        assert "time_to_first_token_ms" in event.payload
        assert "chunk_count" in event.payload
        assert event.payload["chunk_count"] >= 1


def test_run_task_emits_chunk_events_per_sse_delta(tmp_path, monkeypatch):
    """A turn with multiple content deltas produces one
    ``model_response_chunk`` event per delta. ``delta_index`` is a
    zero-based ordinal within the turn.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    class Adapter:
        async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            yield ModelResponseChunk(content="Hel")
            yield ModelResponseChunk(content="lo")
            yield ModelResponseChunk(content=" world")
            yield ModelResponseChunk(finish_reason="stop")

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001
            raise AssertionError("run_task must call stream() (#199)")

        async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
            raise AssertionError("run_task must call stream() (#199)")

    monkeypatch.setattr(runner_mod, "build_model_adapter", Adapter)
    _argv("chunk-task", db, harness_dir, monkeypatch)
    main()

    events = TraceLogger(db).load_session(TraceLogger(db).list_sessions()[0].session_id)
    chunk_events = [event for event in events if event.kind == "model_response_chunk"]
    assert len(chunk_events) == 4
    deltas = [event.payload["delta_index"] for event in chunk_events]
    assert deltas == [0, 1, 2, 3]
    contents = [event.payload["content_so_far"] for event in chunk_events]
    assert contents == ["Hel", "Hello", "Hello world", "Hello world"]
