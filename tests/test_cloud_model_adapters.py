"""Tests for the cloud-native model adapters (issue #1041, ADR-0029).

Covers:

- `CloudModelAdapter` ABC contract (provider override points).
- `AnthropicAdapter` — ``/v1/messages`` request shape, ``event:``/``data:``
  SSE framing, ``anthropic-ratelimit-*`` header parsing, per-token pricing.
- `OpenAINativeAdapter` — native error envelope, ``x-ratelimit-*`` header
  parsing, reuse of the OpenAI chat-completion wire body.
- `resolve_model_adapter` prefix routing (``anthropic/`` / ``openai/`` / fallthrough).
- `ModelCostEvent` / `ModelRateLimitInfo` callback emission.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import AsyncIterator

import httpx
import pytest

from foundry_x.execution.model_adapter import (
    AnthropicAdapter,
    CloudModelAdapter,
    ModelAdapter,
    ModelCostEvent,
    ModelRateLimitInfo,
    ModelRetryEvent,
    OpenAICompatibleAdapter,
    OpenAINativeAdapter,
    resolve_model_adapter,
)
from foundry_x.execution.runner import build_model_adapter_with_overrides

# ---------------------------------------------------------------------------
# AnthropicAdapter
# ---------------------------------------------------------------------------


def test_anthropic_adapter_requires_api_key():
    with pytest.raises(ValueError, match="api_key"):
        AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key=None,
        )


def test_anthropic_adapter_resolves_via_prefix():
    adapter = resolve_model_adapter(
        "anthropic/claude-3-5-sonnet-20241022",
        api_key="sk-ant-test",
    )
    try:
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter.model == "claude-3-5-sonnet-20241022"
        assert adapter.base_url == "https://api.anthropic.com"
        assert isinstance(adapter, ModelAdapter)
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


@pytest.mark.asyncio
async def test_anthropic_complete_posts_messages_endpoint():
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["x_api_key"] = request.headers.get("x-api-key")
        seen["anthropic_version"] = request.headers.get("anthropic-version")
        seen["payload"] = json.loads(request.content.decode())
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "hello"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
        )
        response = await adapter.complete(
            messages=[
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "hi"},
            ],
        )

    assert seen["url"] == "https://api.anthropic.com/v1/messages"
    assert seen["x_api_key"] == "sk-ant-test"
    assert seen["anthropic_version"] == "2023-06-01"
    payload = seen["payload"]
    assert payload["model"] == "claude-3-5-sonnet-20241022"
    assert payload["system"] == "be brief"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]
    assert payload["stream"] is False

    assert response.message.content == "hello"
    assert response.finish_reason == "end_turn"
    assert response.usage is not None
    assert response.usage.prompt_tokens == 10
    assert response.usage.completion_tokens == 5
    assert response.usage.total_tokens == 15


async def _no_sleep(_seconds: float) -> None:
    """No-op replacement for ``asyncio.sleep`` in retry tests."""


@pytest.mark.asyncio
async def test_anthropic_complete_retries_503_then_succeeds(monkeypatch):
    """503 → 503 → 200 yields a ModelResponse with two model_retry events (issue #1168)."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    retries: list[ModelRetryEvent] = []
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        idx = calls["count"]
        calls["count"] += 1
        if idx < 2:
            return httpx.Response(503, text="transient")
        return httpx.Response(
            200,
            json={
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [{"type": "text", "text": "done"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 10, "output_tokens": 5},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        response = await adapter.complete(
            messages=[{"role": "user", "content": "hello"}],
        )

    assert response.message.content == "done"
    assert len(retries) == 2
    assert retries[0].attempt == 1
    assert retries[0].error_type == "HTTPStatusError"
    assert retries[1].attempt == 2
    assert retries[1].error_type == "HTTPStatusError"
    assert all(r.backoff_ms >= 0 for r in retries)


@pytest.mark.asyncio
async def test_anthropic_complete_parses_tool_use_blocks():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "read_file",
                        "input": {"path": "README.md"},
                    }
                ],
                "stop_reason": "tool_use",
                "usage": {"input_tokens": 1, "output_tokens": 2},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
        )
        response = await adapter.complete(messages=[{"role": "user", "content": "read"}])

    assert response.finish_reason == "tool_use"
    assert len(response.tool_calls) == 1
    call = response.tool_calls[0]
    assert call.id == "toolu_1"
    assert call.function.name == "read_file"
    assert json.loads(call.function.arguments) == {"path": "README.md"}


@pytest.mark.asyncio
async def test_anthropic_stream_parses_event_framing():
    body = (
        "event: message_start\n"
        'data: {"type":"message_start","message":{"usage":{"input_tokens":8}}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n\n'
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},'
        '"usage":{"output_tokens":3}}\n\n'
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
        )
        chunks = []
        async for chunk in adapter.stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    contents = [c.content for c in chunks if c.content]
    assert contents == ["Hel", "lo"]
    finish_chunks = [c for c in chunks if c.finish_reason == "end_turn"]
    assert len(finish_chunks) == 1
    usage_chunks = [c for c in chunks if c.usage is not None]
    assert len(usage_chunks) >= 1


@pytest.mark.asyncio
async def test_anthropic_stream_preserves_tool_call_name():
    """content_block_start carries tool name; content_block_delta must preserve it (issue #1275)."""
    body = (
        "event: content_block_start\n"
        'data: {"type":"content_block_start","index":0,'
        '"content_block":{"type":"tool_use","id":"toolu_1","name":"read_file"}}\n\n'
        "event: content_block_delta\n"
        'data: {"type":"content_block_delta","index":0,'
        '"delta":{"type":"tool_use","input_json":"{\\"path\\": \\"README.md\\"}"}}\n\n'
        "event: message_delta\n"
        'data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},'
        '"usage":{"output_tokens":3}}\n\n'
        "event: message_stop\n"
        'data: {"type":"message_stop"}\n\n'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=body,
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
        )
        chunks = []
        async for chunk in adapter.stream(messages=[{"role": "user", "content": "read"}]):
            chunks.append(chunk)

    tool_chunks = [c for c in chunks if c.tool_calls]
    assert len(tool_chunks) == 2
    start_chunk = tool_chunks[0]
    assert start_chunk.tool_calls[0].function.name == "read_file"
    delta_chunk = tool_chunks[1]
    assert delta_chunk.tool_calls[0].function.name == "read_file"
    assert delta_chunk.tool_calls[0].function.arguments == '{"path": "README.md"}'


@pytest.mark.asyncio
async def test_anthropic_complete_emits_cost_and_rate_limit_callbacks():
    cost_events: list[ModelCostEvent] = []
    rate_events: list[ModelRateLimitInfo] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
            headers={
                "anthropic-ratelimit-requests-remaining": "42",
                "anthropic-ratelimit-tokens-remaining": "8000",
                "anthropic-ratelimit-requests-reset": "1m",
                "anthropic-ratelimit-tokens-reset": "2m",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            on_cost=cost_events.append,
            on_rate_limit=rate_events.append,
        )
        await adapter.complete(messages=[{"role": "user", "content": "hi"}])

    assert len(cost_events) == 1
    cost = cost_events[0]
    assert cost.provider == "anthropic"
    assert cost.model == "claude-3-5-sonnet-20241022"
    assert cost.prompt_tokens == 1000
    assert cost.completion_tokens == 500
    # claude-3-5-sonnet: $3.00/1M input, $15.00/1M output
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000.0
    assert abs(cost.estimated_cost_usd - round(expected, 8)) < 1e-9

    assert len(rate_events) == 1
    info = rate_events[0]
    assert info.requests_remaining == 42
    assert info.tokens_remaining == 8000
    assert info.requests_reset_seconds == 60.0
    assert info.tokens_reset_seconds == 120.0


@pytest.mark.asyncio
async def test_anthropic_complete_no_callbacks_when_usage_missing():
    cost_events: list[ModelCostEvent] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            on_cost=cost_events.append,
        )
        await adapter.complete(messages=[{"role": "user", "content": "hi"}])

    assert cost_events == []


# ---------------------------------------------------------------------------
# OpenAINativeAdapter
# ---------------------------------------------------------------------------


def test_openai_native_adapter_requires_api_key():
    with pytest.raises(ValueError, match="api_key"):
        OpenAINativeAdapter(
            model="gpt-4o",
            base_url="https://api.openai.com",
            api_key=None,
        )


def test_openai_native_adapter_resolves_via_prefix():
    adapter = resolve_model_adapter("openai/gpt-4o", api_key="sk-openai-test")
    try:
        assert isinstance(adapter, OpenAINativeAdapter)
        assert adapter.model == "gpt-4o"
        assert adapter.base_url == "https://api.openai.com"
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


@pytest.mark.asyncio
async def test_openai_native_complete_posts_chat_completions():
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
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAINativeAdapter(
            model="gpt-4o",
            base_url="https://api.openai.com",
            api_key="sk-openai-test",
            client=client,
        )
        response = await adapter.complete(messages=[{"role": "user", "content": "hi"}])

    assert seen["url"] == "https://api.openai.com/v1/chat/completions"
    assert seen["authorization"] == "Bearer sk-openai-test"
    assert response.message.content == "done"
    assert response.usage.total_tokens == 8


@pytest.mark.asyncio
async def test_openai_native_complete_raises_response_error_on_error_envelope():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={"error": {"message": "invalid_api_key", "type": "invalid_request_error"}},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAINativeAdapter(
            model="gpt-4o",
            base_url="https://api.openai.com",
            api_key="sk-openai-test",
            client=client,
        )
        with pytest.raises(Exception, match="invalid_api_key"):
            await adapter.complete(messages=[{"role": "user", "content": "hi"}])


@pytest.mark.asyncio
async def test_openai_native_complete_parses_rate_limit_headers():
    rate_events: list[ModelRateLimitInfo] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            },
            headers={
                "x-ratelimit-remaining-requests": "1000",
                "x-ratelimit-remaining-tokens": "50000",
                "x-ratelimit-reset-requests": "6m0s",
                "x-ratelimit-reset-tokens": "5m0s",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAINativeAdapter(
            model="gpt-4o",
            base_url="https://api.openai.com",
            api_key="sk-openai-test",
            client=client,
            on_rate_limit=rate_events.append,
        )
        await adapter.complete(messages=[{"role": "user", "content": "hi"}])

    assert len(rate_events) == 1
    info = rate_events[0]
    assert info.requests_remaining == 1000
    assert info.tokens_remaining == 50000


# ---------------------------------------------------------------------------
# resolve_model_adapter
# ---------------------------------------------------------------------------


def test_resolve_model_adapter_falls_back_to_openai_compatible():
    adapter = resolve_model_adapter(
        "local-model",
        api_key="test",
        base_url="http://localhost:8080",
    )
    try:
        from foundry_x.execution.model_adapter import OpenAICompatibleAdapter

        assert isinstance(adapter, OpenAICompatibleAdapter)
        assert adapter.model == "local-model"
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


def test_resolve_model_adapter_compatible_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        resolve_model_adapter("local-model", api_key="test")


@pytest.mark.asyncio
async def test_build_model_adapter_with_overrides_routes_anthropic():
    adapter = build_model_adapter_with_overrides(
        "anthropic/claude-3-5-haiku-20241022",
        quantization=None,
        path_or_endpoint="https://api.anthropic.com",
        env={"ANTHROPIC_API_KEY": "sk-ant-test"},
    )
    try:
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter.model == "claude-3-5-haiku-20241022"
        assert adapter.base_url == "https://api.anthropic.com"
    finally:
        await adapter.aclose()


@pytest.mark.asyncio
async def test_build_model_adapter_with_overrides_routes_openai():
    adapter = build_model_adapter_with_overrides(
        "openai/gpt-4o-mini",
        quantization=None,
        path_or_endpoint=None,
        env={"OPENAI_API_KEY": "sk-openai-test"},
    )
    try:
        assert isinstance(adapter, OpenAINativeAdapter)
        assert adapter.model == "gpt-4o-mini"
        assert adapter.base_url == "https://api.openai.com"
    finally:
        await adapter.aclose()


def test_resolve_model_adapter_env_override_compatible(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_ADAPTER", "OpenAICompatibleAdapter")
    adapter = resolve_model_adapter(
        "anthropic/claude-3-5-sonnet-20241022",
        api_key="test",
        base_url="http://localhost:8080",
    )
    try:
        assert isinstance(adapter, OpenAICompatibleAdapter)
        assert adapter.model == "anthropic/claude-3-5-sonnet-20241022"
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


def test_resolve_model_adapter_env_override_unknown(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_ADAPTER", "NonExistentAdapter")
    with pytest.raises(ValueError, match="Unknown FOUNDRY_MODEL_ADAPTER"):
        resolve_model_adapter("test-model", api_key="test")


def test_resolve_model_adapter_env_override_compatible_requires_base_url(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_ADAPTER", "OpenAICompatibleAdapter")
    with pytest.raises(ValueError, match="base_url"):
        resolve_model_adapter("anthropic/claude-3-5-sonnet-20241022", api_key="test")


def test_resolve_model_adapter_env_override_anthropic(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_ADAPTER", "AnthropicAdapter")
    adapter = resolve_model_adapter(
        "anthropic/claude-3-5-sonnet-20241022",
        api_key="sk-ant-test",
        base_url="https://api.anthropic.com",
    )
    try:
        assert isinstance(adapter, AnthropicAdapter)
        assert adapter.model == "claude-3-5-sonnet-20241022"
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


def test_resolve_model_adapter_env_override_openai(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_ADAPTER", "OpenAINativeAdapter")
    adapter = resolve_model_adapter(
        "openai/gpt-4o-mini",
        api_key="sk-openai-test",
    )
    try:
        assert isinstance(adapter, OpenAINativeAdapter)
        assert adapter.model == "gpt-4o-mini"
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


# ---------------------------------------------------------------------------
# CloudModelAdapter ABC contract
# ---------------------------------------------------------------------------


def test_cloud_model_adapter_is_abstract():
    with pytest.raises(TypeError):
        CloudModelAdapter(  # type: ignore[abstract]
            model="x",
            base_url="https://example.com",
            api_key="x",
        )


async def _no_sleep(_seconds: float) -> None:
    """No-op replacement for ``asyncio.sleep`` in retry tests."""


class _FailingLinesResponse(httpx.Response):
    """Response whose ``aiter_lines()`` yields partial SSE then raises."""

    def __init__(self, *args: object, fail_after: int = 1, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._fail_after = fail_after

    async def aiter_lines(self) -> AsyncIterator[str]:  # type: ignore[override]
        n = 0
        async for line in super().aiter_lines():  # type: ignore[misc]
            yield line
            n += 1
            if n >= self._fail_after:
                raise OSError("simulated mid-stream connection drop")


# ---------------------------------------------------------------------------
# Mid-stream retry boundary — issue #200 / #1164 / #1278
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_adapter_stream_retries_transport_error(monkeypatch):
    """Transport errors (ConnectError) are retried during streaming."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    calls: dict[str, int] = {"count": 0}
    retries: list[ModelRetryEvent] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] <= 2:
            raise httpx.ConnectError("connection refused", request=request)
        body = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n'
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n'
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        chunks = []
        async for chunk in adapter.stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    assert calls["count"] == 3, "ConnectError should be retried twice then succeed"
    assert len(retries) == 2
    assert retries[0].error_type == "ConnectError"
    assert retries[1].error_type == "ConnectError"
    contents = [c.content for c in chunks if c.content]
    assert contents == ["Hi"]


@pytest.mark.asyncio
async def test_anthropic_adapter_stream_retries_429(monkeypatch):
    """HTTP 429 (rate-limit) is retried during streaming."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    calls: dict[str, int] = {"count": 0}
    retries: list[ModelRetryEvent] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] <= 2:
            return httpx.Response(429, text="rate limited")
        body = (
            "event: message_start\n"
            'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n'
            "event: content_block_delta\n"
            'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n'
            "event: message_stop\n"
            'data: {"type":"message_stop"}\n\n'
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        chunks = []
        async for chunk in adapter.stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    assert calls["count"] == 3, "429 should be retried twice then succeed"
    assert len(retries) == 2
    assert retries[0].error_type == "HTTPStatusError"
    assert retries[1].error_type == "HTTPStatusError"
    contents = [c.content for c in chunks if c.content]
    assert contents == ["Hi"]


@pytest.mark.asyncio
async def test_openai_native_adapter_stream_retries_503(monkeypatch):
    """HTTP 503 is retried during streaming for OpenAINativeAdapter."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    calls: dict[str, int] = {"count": 0}
    retries: list[ModelRetryEvent] = []

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

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = OpenAINativeAdapter(
            model="gpt-4o",
            base_url="https://api.openai.com",
            api_key="sk-openai-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        chunks = []
        async for chunk in adapter.stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    assert calls["count"] == 3, "503 should be retried twice then succeed"
    assert len(retries) == 2
    assert retries[0].error_type == "HTTPStatusError"
    assert retries[1].error_type == "HTTPStatusError"
    contents = [c.content for c in chunks if c.content]
    assert contents == ["hi"]


@pytest.mark.asyncio
async def test_anthropic_stream_does_not_retry_mid_stream_failure(monkeypatch):
    """Connection errors are retried; mid-stream errors are NOT retried.

    This mirrors the OpenAICompatibleAdapter contract from issue #200.
    Verifies that after a successful connection + 200 response, an error
    during SSE iteration propagates immediately without triggering a retry.
    """
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    calls: dict[str, int] = {"count": 0}
    retries: list[object] = []

    def handler(request: httpx.Request) -> _FailingLinesResponse:
        calls["count"] += 1
        return _FailingLinesResponse(
            200,
            content=(
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta",'
                '"delta":{"type":"text_delta","text":"Hi"}}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
            fail_after=1,  # fail on the second line
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        with pytest.raises(OSError, match="simulated mid-stream"):
            async for _chunk in adapter.stream(messages=[{"role": "user", "content": "hi"}]):
                pass

    assert calls["count"] == 1, "handler should only be called once (no mid-stream retry)"
    assert retries == [], "no retry events for mid-stream failure"


@pytest.mark.asyncio
async def test_anthropic_stream_retries_connection_failure(monkeypatch):
    """Connection-establishment errors ARE retried (issue #200 / #1164)."""
    monkeypatch.setattr("foundry_x.execution.model_adapter.asyncio.sleep", _no_sleep)

    calls: dict[str, int] = {"count": 0}
    retries: list[object] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if calls["count"] <= 2:
            return httpx.Response(503, text="unavailable")
        return httpx.Response(
            200,
            content=(
                "event: message_start\n"
                'data: {"type":"message_start","message":{"usage":{"input_tokens":5}}}\n\n'
                "event: content_block_delta\n"
                'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi"}}\n\n'
                "event: message_stop\n"
                'data: {"type":"message_stop"}\n\n'
            ),
            headers={"content-type": "text/event-stream"},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            max_retries=2,
            on_retry=retries.append,
        )
        chunks = []
        async for chunk in adapter.stream(messages=[{"role": "user", "content": "hi"}]):
            chunks.append(chunk)

    assert calls["count"] == 3, "503 should be retried twice then succeed"
    assert len(retries) == 2
    contents = [c.content for c in chunks if c.content]
    assert contents == ["Hi"]


def test_unknown_model_pricing_returns_zero():
    adapter = resolve_model_adapter(
        "anthropic/unknown-model",
        api_key="sk-test",
    )
    try:
        assert adapter.token_pricing() == (0.0, 0.0)
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


def test_pricing_env_var_override_anthropic(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_PRICING_CLAUDE_3_5_SONNET_20241022", "5.0,25.0")
    adapter = AnthropicAdapter(
        model="claude-3-5-sonnet-20241022",
        base_url="https://api.anthropic.com",
        api_key="sk-test",
    )
    try:
        assert adapter.token_pricing() == (5.0, 25.0)
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


def test_pricing_env_var_override_openai(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_PRICING_GPT_4O", "10.0,40.0")
    adapter = OpenAINativeAdapter(
        model="gpt-4o",
        base_url="https://api.openai.com",
        api_key="sk-test",
    )
    try:
        assert adapter.token_pricing() == (10.0, 40.0)
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


def test_pricing_env_var_invalid_format_falls_back_to_hardcoded(monkeypatch):
    monkeypatch.setenv("FOUNDRY_MODEL_PRICING_CLAUDE_3_5_SONNET_20241022", "not-a-number")
    adapter = AnthropicAdapter(
        model="claude-3-5-sonnet-20241022",
        base_url="https://api.anthropic.com",
        api_key="sk-test",
    )
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            pricing = adapter.token_pricing()
            assert pricing == (3.0, 15.0)
            assert len(w) == 1
            assert "Invalid pricing" in str(w[0].message)
            assert "RuntimeWarning" in str(w[0].category)
    finally:
        import asyncio

        asyncio.run(adapter.aclose())


@pytest.mark.asyncio
async def test_anthropic_cost_known_model_marks_pricing_known():
    """CloudModelAdapter stamps pricing_known=True for a catalogued model (issue #1465)."""
    cost_events: list[ModelCostEvent] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-3-5-sonnet-20241022",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            on_cost=cost_events.append,
        )
        await adapter.complete(messages=[{"role": "user", "content": "hi"}])

    assert len(cost_events) == 1
    event = cost_events[0]
    assert event.pricing_known is True
    expected = (1000 * 3.0 + 500 * 15.0) / 1_000_000.0
    assert abs(event.estimated_cost_usd - round(expected, 8)) < 1e-9


@pytest.mark.asyncio
async def test_anthropic_cost_unknown_model_marks_pricing_unknown():
    """CloudModelAdapter stamps pricing_known=False, cost 0.0 for an unknown model (issue #1465)."""
    cost_events: list[ModelCostEvent] = []

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "role": "assistant",
                "content": [{"type": "text", "text": "ok"}],
                "stop_reason": "end_turn",
                "usage": {"input_tokens": 1000, "output_tokens": 500},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        adapter = AnthropicAdapter(
            model="claude-future-unreleased",
            base_url="https://api.anthropic.com",
            api_key="sk-ant-test",
            client=client,
            on_cost=cost_events.append,
        )
        await adapter.complete(messages=[{"role": "user", "content": "hi"}])

    assert len(cost_events) == 1
    event = cost_events[0]
    assert event.pricing_known is False
    assert event.estimated_cost_usd == 0.0
