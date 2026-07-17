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
    ModelRetryEvent,
    ModelToolCallChunk,
    OpenAICompatibleAdapter,
    ToolCallFunctionChunk,
    ToolDefinition,
    ToolFunctionSchema,
)
from foundry_x.execution.runner import (
    _DEFAULT_REQUEST_TIMEOUT_S,
    _resolve_request_timeout,
    build_model_adapter,
    run_task,
)
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
    # Issue #197: model_response carries token_usage and tokens_used.
    assert "token_usage" in response_event.payload
    assert "tokens_used" in response_event.payload
    assert response_event.payload["tokens_used"] == 0
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
    # Issue #197: outcome event also carries tokens_total.
    assert outcome_event.payload["tokens_total"] == 0


@pytest.mark.asyncio
async def test_run_task_raises_TypeError_for_non_protocol_model_adapter(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("system rules", encoding="utf-8")
    logger = TraceLogger(tmp_path / "traces.db")

    class NotAnAdapter:
        pass

    with logger.session(harness_version="test-0.0") as session_id:
        with pytest.raises(TypeError, match="ModelAdapter"):
            await run_task(
                "do the task", tmp_path, logger, session_id, model_adapter=NotAnAdapter()
            )


@pytest.mark.asyncio
async def test_run_task_raises_TypeError_for_dict_model_adapter(tmp_path):
    (tmp_path / "system_prompt.txt").write_text("system rules", encoding="utf-8")
    logger = TraceLogger(tmp_path / "traces.db")

    with logger.session(harness_version="test-0.0") as session_id:
        with pytest.raises(TypeError, match="ModelAdapter"):
            await run_task(
                "do the task", tmp_path, logger, session_id, model_adapter={"wrong": "keys"}
            )


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
async def test_complete_parses_usage_block():
    """OpenAI-compatible chat-completion responses (and the llama-server
    ``/v1/chat/completions`` route that mirrors them) carry a top-level
    ``usage`` object — issue #197 wires it through :class:`ModelUsage` so
    ``run_task`` can read ``usage.total_tokens`` against
    ``FOUNDRY_TOKEN_BUDGET``. The canonical wire shape has
    ``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``; the
    adapter must accept all three and forward them verbatim.
    """
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-test",
                "object": "chat.completion",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 25,
                    "completion_tokens": 12,
                    "total_tokens": 37,
                },
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test/v1",
            model="foundry-test",
            client=client,
        )
        response = await adapter.complete(
            messages=[ModelMessage(role="user", content="hi")],
            tools=[],
        )

    assert response.usage is not None
    assert response.usage.prompt_tokens == 25
    assert response.usage.completion_tokens == 12
    assert response.usage.total_tokens == 37


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


# --- per-request httpx timeout (issue #201) --------------------------------


def _env_with_timeout(value: str | None) -> dict[str, str]:
    """Build the env dict ``build_model_adapter`` needs, optionally setting the timeout."""
    env = {
        "OPENCODE_SERVER_URL": "http://model.test/v1",
        "FOUNDRY_MODEL_ID": "foundry-test",
    }
    if value is not None:
        env["FOUNDRY_REQUEST_TIMEOUT_S"] = value
    return env


@pytest.mark.asyncio
async def test_build_model_adapter_applies_default_request_timeout():
    """Issue #201: when ``FOUNDRY_REQUEST_TIMEOUT_S`` is unset the owned
    httpx client is built with the default cap so a stuck endpoint still
    aborts within the documented budget.
    """
    adapter = build_model_adapter(_env_with_timeout(value=None))
    try:
        assert adapter._client.timeout.read == _DEFAULT_REQUEST_TIMEOUT_S
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_build_model_adapter_applies_env_request_timeout():
    """Issue #201: an explicit ``FOUNDRY_REQUEST_TIMEOUT_S`` is plumbed into
    the owned httpx client as the per-request cap.
    """
    adapter = build_model_adapter(_env_with_timeout(value="5.5"))
    try:
        assert adapter._client.timeout.read == 5.5
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_build_model_adapter_non_positive_request_timeout_falls_back():
    """Issue #201: a non-positive value falls back to the default rather than
    disabling the per-step guard (a stuck step would otherwise hang the
    session until the wall-clock cap fires).
    """
    for raw in ("0", "-1", "0.0"):
        adapter = build_model_adapter(_env_with_timeout(value=raw))
        try:
            assert adapter._client.timeout.read == _DEFAULT_REQUEST_TIMEOUT_S, raw
        finally:
            await adapter.aclose()


def test_resolve_request_timeout_non_numeric_raises():
    """Issue #201: a non-numeric value raises ValueError at process start so a
    typo in ``.env`` surfaces immediately (AGENTS.md §2).
    """
    with pytest.raises(ValueError):
        _resolve_request_timeout(env={"FOUNDRY_REQUEST_TIMEOUT_S": "not-a-number"})


@pytest.mark.parametrize("raw", ["inf", "-inf", "nan", "Infinity", "NaN"])
def test_resolve_request_timeout_non_finite_raises(raw):
    """Issue #201: a non-finite value raises ValueError at process start; an
    infinite/NaN cap would never fire and silently disable the guard.
    """
    with pytest.raises(ValueError):
        _resolve_request_timeout(env={"FOUNDRY_REQUEST_TIMEOUT_S": raw})


@pytest.mark.asyncio
async def test_adapter_request_timeout_wraps_read_timeout():
    """Issue #201: when the per-request cap fires the adapter wraps the
    httpx timeout as :class:`ModelAdapterError` with ``__cause__`` carrying
    the original :class:`httpx.ReadTimeout`, so ``run_task`` records a
    ``model_error`` outcome instead of hanging.

    ``httpx.MockTransport`` bypasses the network-layer timeout enforcement
    (the cap is applied around real socket I/O), so the handler emulates the
    read timeout a real transport would raise once the configured budget is
    exceeded.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("read timed out", request=request)

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        timeout=0.01,
    ) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
        )
        with pytest.raises(ModelAdapterError) as exc_info:
            await adapter.complete(messages=[ModelMessage(role="user", content="hello")], tools=[])
        assert isinstance(exc_info.value.__cause__, httpx.ReadTimeout)


# ---------------------------------------------------------------------------
# Issue #200 — bounded retry on transient ModelAdapter failures
# ---------------------------------------------------------------------------

_SUCCESS_BODY = {
    "choices": [
        {
            "message": {"role": "assistant", "content": "done"},
            "finish_reason": "stop",
        }
    ]
}


async def _no_sleep(_seconds: float) -> None:
    """No-op replacement for ``asyncio.sleep`` in retry tests."""


def _make_status_handler(statuses: list[int]):
    """Return a MockTransport handler that replays *statuses* then 200."""
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        idx = calls["count"]
        calls["count"] += 1
        if idx < len(statuses):
            return httpx.Response(statuses[idx], text="transient")
        return httpx.Response(200, json=_SUCCESS_BODY)

    return handler


@pytest.mark.asyncio
async def test_post_json_retries_503_then_succeeds(monkeypatch):
    """503 → 503 → 200 yields a ModelResponse with two model_retry events."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    retries: list[ModelRetryEvent] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_make_status_handler([503, 503]))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test/v1",
            model="foundry-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        response = await adapter.complete(
            messages=[ModelMessage(role="user", content="hello")],
        )

    assert response.message.content == "done"
    assert len(retries) == 2
    assert retries[0].attempt == 1
    assert retries[0].error_type == "HTTPStatusError"
    assert retries[1].attempt == 2
    assert retries[1].error_type == "HTTPStatusError"
    assert all(r.backoff_ms >= 0 for r in retries)


@pytest.mark.asyncio
async def test_retries_on_connect_error(monkeypatch):
    """httpx.ConnectError is retried, then succeeds on the third attempt."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    retries: list[ModelRetryEvent] = []
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] <= 2:
            raise httpx.ConnectError("boom", request=request)
        return httpx.Response(200, json=_SUCCESS_BODY)

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        response = await adapter.complete(
            messages=[ModelMessage(role="user", content="hello")],
        )

    assert response.message.content == "done"
    assert len(retries) == 2
    assert retries[0].error_type == "ConnectError"


@pytest.mark.asyncio
async def test_no_retry_on_400_client_error():
    """HTTP 400 is not retryable — adapter raises immediately."""
    retries: list[ModelRetryEvent] = []
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(400, text="bad request")

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        with pytest.raises(ModelAdapterHTTPError) as exc_info:
            await adapter.complete(
                messages=[ModelMessage(role="user", content="hello")],
            )

    assert exc_info.value.status_code == 400
    assert calls["count"] == 1
    assert retries == []


@pytest.mark.asyncio
async def test_no_retry_on_response_parse_error():
    """ModelAdapterResponseError is not retried (issue #200)."""
    retries: list[ModelRetryEvent] = []
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        return httpx.Response(200, json={"choices": []})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        with pytest.raises(ModelAdapterResponseError):
            await adapter.complete(
                messages=[ModelMessage(role="user", content="hello")],
            )

    assert calls["count"] == 1
    assert retries == []


@pytest.mark.asyncio
async def test_retries_exhausted_raises_http_error(monkeypatch):
    """All retries exhausted → ModelAdapterHTTPError on final attempt."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    retries: list[ModelRetryEvent] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda r: httpx.Response(503, text="down"))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        with pytest.raises(ModelAdapterHTTPError) as exc_info:
            await adapter.complete(
                messages=[ModelMessage(role="user", content="hello")],
            )

    assert exc_info.value.status_code == 503
    assert len(retries) == 2


@pytest.mark.asyncio
async def test_retry_429_is_retryable(monkeypatch):
    """HTTP 429 (rate-limit) is in the retryable set."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    retries: list[ModelRetryEvent] = []
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(_make_status_handler([429]))
    ) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        response = await adapter.complete(
            messages=[ModelMessage(role="user", content="hello")],
        )

    assert response.message.content == "done"
    assert len(retries) == 1
    assert retries[0].error_type == "HTTPStatusError"


@pytest.mark.asyncio
async def test_stream_retries_503_then_succeeds(monkeypatch):
    """The SSE stream retries a 503 connection error, then yields chunks."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] <= 2:
            return httpx.Response(503, text="down")
        body = (
            "data: "
            + json.dumps({"choices": [{"delta": {"content": "hi"}, "finish_reason": "stop"}]})
            + "\n\ndata: [DONE]\n\n"
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    retries: list[ModelRetryEvent] = []
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAICompatibleAdapter(
            base_url="http://model.test",
            model="foundry-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        chunks = []
        async for chunk in adapter.stream(
            messages=[ModelMessage(role="user", content="hello")],
        ):
            chunks.append(chunk)

    assert calls["count"] == 3
    assert len(retries) == 2
    assert retries[0].error_type == "HTTPStatusError"
    assert len(chunks) == 1
    assert chunks[0].content == "hi"


@pytest.mark.asyncio
async def test_resolve_adapter_max_retries_from_env():
    from foundry_x.execution.model_adapter import resolve_adapter_max_retries

    assert resolve_adapter_max_retries({}) == 2
    assert resolve_adapter_max_retries({"FOUNDRY_ADAPTER_MAX_RETRIES": "5"}) == 5
    assert resolve_adapter_max_retries({"FOUNDRY_ADAPTER_MAX_RETRIES": "0"}) == 0
    assert resolve_adapter_max_retries({"FOUNDRY_ADAPTER_MAX_RETRIES": "  3 "}) == 3

    with pytest.raises(ValueError):
        resolve_adapter_max_retries({"FOUNDRY_ADAPTER_MAX_RETRIES": "abc"})
