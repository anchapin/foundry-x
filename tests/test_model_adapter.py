from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from foundry_x.execution.model_adapter import (
    ModelAdapter,
    ModelAdapterError,
    ModelAdapterHTTPError,
    ModelAdapterResponseError,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCallChunk,
    OpenAICompatibleAdapter,
    ToolCallFunctionChunk,
    ToolDefinition,
    ToolFunctionSchema,
)
from foundry_x.execution.runner import build_model_adapter, run_task
from foundry_x.trace.logger import TraceLogger


class FakeAdapter:
    def __init__(self) -> None:
        self.messages: list[ModelMessage] = []
        self.tools: list[ToolDefinition] | None = None

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.messages = [ModelMessage.model_validate(message) for message in messages]
        self.tools = [ToolDefinition.model_validate(tool) for tool in tools] if tools else []
        return ModelResponse(
            message=ModelMessage(role="assistant", content="runner response"),
            finish_reason="stop",
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


@pytest.mark.asyncio
async def test_build_model_adapter_uses_openai_compatible_env():
    adapter = build_model_adapter(
        {
            "OPENCODE_SERVER_URL": "http://model.test/v1",
            "FOUNDRY_MODEL_ID": "foundry-test",
            "FOUNDRY_MODEL_API_KEY": "test-token",
        }
    )
    try:
        assert adapter.base_url == "http://model.test/v1"
        assert adapter.model == "foundry-test"
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_run_task_calls_injected_adapter_and_records_trace(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("system rules", encoding="utf-8")
    logger = TraceLogger(tmp_path / "traces.db")
    adapter = FakeAdapter()

    with logger.session(harness_version="test-0.0") as session_id:
        await run_task("do the task", tmp_path, logger, session_id, model_adapter=adapter)
        events = logger.load_session(session_id)

    assert adapter.messages == [
        ModelMessage(role="system", content="system rules"),
        ModelMessage(role="user", content="do the task"),
    ]
    assert adapter.tools == []
    kinds = [event.kind for event in events]
    # Issue #89 / ADR-0009: run_task now brackets the round-trip with a
    # ``user_prompt`` marker at entry and an ``outcome`` marker at exit;
    # the round-trip event kinds (model_request/model_response) keep their
    # position in the middle. Issue #199 inserts at least one
    # ``model_response_chunk`` event between the request and the terminal
    # response (the stub adapter yields one content chunk + one
    # finish_reason chunk per round-trip).
    assert kinds[0] == "user_prompt"
    assert kinds[-1] == "outcome"
    assert kinds.count("model_request") == 1
    assert kinds.count("model_response") == 1
    chunk_idxs = [i for i, k in enumerate(kinds) if k == "model_response_chunk"]
    assert len(chunk_idxs) >= 1
    request_idx = kinds.index("model_request")
    response_idx = kinds.index("model_response")
    for ci in chunk_idxs:
        assert request_idx < ci < response_idx, (
            f"chunk event at {ci} must be between model_request {request_idx} "
            f"and model_response {response_idx}"
        )
    response_event = next(event for event in events if event.kind == "model_response")
    assert response_event.payload["message"]["content"] == "runner response"
    # Issue #199: model_response carries time_to_first_token_ms and chunk_count.
    assert "time_to_first_token_ms" in response_event.payload
    assert "chunk_count" in response_event.payload
    assert response_event.payload["chunk_count"] >= 1
    chunk_event = next(event for event in events if event.kind == "model_response_chunk")
    assert chunk_event.payload["delta_index"] == 0
    assert chunk_event.payload["content_so_far"] == "runner response"
    assert chunk_event.payload["chunk_duration_ms"] >= 0
    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "success"
    assert outcome_event.payload["reason"] == "final_answer"
    assert outcome_event.payload["steps"] == 1
    # Issue #199: outcome event also carries ttft_ms (p50 across turns).
    assert "ttft_ms" in outcome_event.payload
    assert outcome_event.payload["ttft_ms"] >= 0
    assert "chunk_count" in response_event.payload
    assert response_event.payload["chunk_count"] >= 1
    chunk_event = next(event for event in events if event.kind == "model_response_chunk")
    assert chunk_event.payload["delta_index"] == 0
    assert chunk_event.payload["content_so_far"] == "runner response"
    assert chunk_event.payload["chunk_duration_ms"] >= 0
    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "success"
    assert outcome_event.payload["reason"] == "final_answer"
    assert outcome_event.payload["steps"] == 1
    # Issue #199: outcome event also carries ttft_ms (p50 across turns).
    assert "ttft_ms" in outcome_event.payload
    assert outcome_event.payload["ttft_ms"] >= 0


@pytest.mark.asyncio
async def test_openai_adapter_conforms_to_protocol():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
        )
        assert isinstance(adapter, ModelAdapter)


def test_request_models_validate_required_fields():
    with pytest.raises(ValidationError):
        ModelRequest(model="", messages=[ModelMessage(role="user", content="hello")])

    with pytest.raises(ValidationError):
        ModelRequest(model="foundry-test", messages=[])


def test_tool_schema_serializes_to_openai_shape():
    tool = ToolDefinition(
        function=ToolFunctionSchema(
            name="read_file",
            description="Read a file from the workspace",
            parameters={
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        )
    )

    payload = ModelRequest(
        model="foundry-test",
        messages=[ModelMessage(role="user", content="Read README.md")],
        tools=[tool],
    ).to_openai_payload()

    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a file from the workspace",
                "parameters": {
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                },
            },
        }
    ]


@pytest.mark.asyncio
async def test_complete_posts_request_and_parses_response():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization")
        seen["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test/v1",
            model="foundry-test",
            api_key="test-token",
            client=client,
        )
        response = await adapter.complete(
            messages=[ModelMessage(role="user", content="hello")],
            tools=[],
            temperature=0,
        )

    assert seen["url"] == "http://model.test/v1/chat/completions"
    assert seen["authorization"] == " ".join(["Bearer", "test-token"])
    assert seen["payload"] == {
        "model": "foundry-test",
        "messages": [{"role": "user", "content": "hello"}],
        "tools": [],
        "stream": False,
        "temperature": 0,
    }
    assert response.message.content == "done"
    assert response.tool_calls == []
    assert response.finish_reason == "stop"


@pytest.mark.asyncio
async def test_complete_parses_tool_calls():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "read_file",
                                        "arguments": '{"path":"README.md"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ]
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
        )
        response = await adapter.chat(messages=[{"role": "user", "content": "read"}], tools=[])

    assert response.finish_reason == "tool_calls"
    assert response.tool_calls[0].id == "call_1"
    assert response.tool_calls[0].function.name == "read_file"


@pytest.mark.asyncio
async def test_http_error_raises_adapter_error_with_status_code():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(503, text="offline"))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
        )
        with pytest.raises(ModelAdapterHTTPError) as exc_info:
            await adapter.complete(messages=[ModelMessage(role="user", content="hello")], tools=[])

    assert exc_info.value.status_code == 503
    assert "offline" in exc_info.value.response_body


@pytest.mark.asyncio
async def test_network_error_is_wrapped_and_chained():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
        )
        with pytest.raises(ModelAdapterError) as exc_info:
            await adapter.complete(messages=[ModelMessage(role="user", content="hello")], tools=[])

    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


@pytest.mark.asyncio
async def test_invalid_response_shape_raises_response_error():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"choices": []}))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
        )
        with pytest.raises(ModelAdapterResponseError):
            await adapter.complete(messages=[ModelMessage(role="user", content="hello")], tools=[])
