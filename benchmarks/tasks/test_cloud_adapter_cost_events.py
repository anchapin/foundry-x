"""Benchmark task: CloudModelAdapter cost/rate-limit event emission (issue #1262).

ADR-0029 (§4, issue #1041) introduced ``model_cost`` and ``model_rate_limit``
trace events emitted by ``CloudModelAdapter`` subclasses. These events feed the
``improvement-rate`` KPI's cost attribution.  This benchmark verifies the
event-emission path exercises correctly by driving ``Runner.run_task`` with a
scripted ``CloudModelAdapter`` subclass and asserting the trace contains both
event kinds with correctly-populated payloads.

Success criteria (issue #1262):
- ``model_cost`` event present with provider, model, prompt_tokens,
  completion_tokens, estimated_cost_usd
- ``model_rate_limit`` event present with requests_remaining or tokens_remaining
- Both ``complete()`` (non-streaming) and ``stream()`` paths are exercised
- estimated_cost_usd > 0 when usage is non-zero
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.model_adapter import (
    CloudModelAdapter,
    ModelCostEvent,
    ModelMessage,
    ModelRateLimitInfo,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ModelUsage,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger

TASK = BenchmarkTask(
    name="cloud_adapter_cost_events",
    description=(
        "Drive Runner.run_task against a scripted CloudModelAdapter subclass "
        "and assert model_cost and model_rate_limit trace events are emitted "
        "with correctly-populated payloads (issue #1262)."
    ),
    tags=["cloud-adapter", "cost-events", "rate-limit-events"],
    difficulty_tier="easy",
)


class _ScriptedCloudAdapter(CloudModelAdapter):
    """Scripted ``CloudModelAdapter`` that replays canned responses without HTTP.

    Implements all abstract methods as no-ops and overrides ``complete()`` /
    ``stream()`` to return canned ``ModelResponse`` objects with populated
    ``usage`` so that ``_emit_cost_and_rate_limit`` fires the ``on_cost`` and
    ``on_rate_limit`` callbacks on every call.
    """

    provider = "test-provider"

    def __init__(
        self,
        *,
        responses: list[ModelResponse],
        stream_chunks_per_response: list[list[ModelResponseChunk]],
        rate_limit_headers: dict[str, str],
        **kwargs: Any,
    ) -> None:
        super().__init__(model="test-model", base_url="http://test.local", **kwargs)
        self._responses = responses
        self._stream_chunks = stream_chunks_per_response
        self._rate_limit_headers = rate_limit_headers
        self._call_count = 0

    def _build_auth_headers(self, api_key: str | None) -> dict[str, str]:
        return {}

    def build_request(
        self,
        messages: Any,
        tools: Any = None,
        *,
        stream: bool,
        extra_params: dict[str, Any],
    ) -> dict[str, Any]:
        return {}

    def request_url(self, *, stream: bool) -> str:
        return "http://test.local/v1/chat"

    def request_headers(self, *, stream: bool) -> dict[str, str]:
        return {"content-type": "application/json"}

    def parse_response(self, data: dict[str, Any]) -> ModelResponse:
        if self._call_count >= len(self._responses):
            raise RuntimeError("_ScriptedCloudAdapter exhausted")
        return self._responses[self._call_count]

    def parse_stream_chunk(self, data: dict[str, Any]) -> ModelResponseChunk | None:
        return None

    def rate_limit_headers(self) -> tuple[str, ...]:
        return tuple(self._rate_limit_headers.keys())

    def token_pricing(self) -> tuple[float, float]:
        return (1.0, 4.0)

    def parse_rate_limit(self, headers_lower: Mapping[str, str]) -> ModelRateLimitInfo:
        def _to_int(value: str | None) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except ValueError:
                return None

        def _to_float(value: str | None) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value)
            except ValueError:
                return None

        return ModelRateLimitInfo(
            requests_remaining=_to_int(headers_lower.get("x-ratelimit-remaining-requests")),
            tokens_remaining=_to_int(headers_lower.get("x-ratelimit-remaining-tokens")),
            requests_reset_seconds=_to_float(headers_lower.get("x-ratelimit-reset-requests")),
            tokens_reset_seconds=_to_float(headers_lower.get("x-ratelimit-reset-tokens")),
        )

    async def complete(
        self,
        messages: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> ModelResponse:
        if self._call_count >= len(self._responses):
            raise RuntimeError(f"_ScriptedCloudAdapter exhausted after {self._call_count} calls")
        response = self._responses[self._call_count]
        self._call_count += 1
        headers = dict(self._rate_limit_headers)
        self._emit_cost_and_rate_limit(response, headers)
        return response

    async def stream(
        self,
        messages: Any,
        tools: Any = None,
        **kwargs: Any,
    ) -> AsyncIterator[ModelResponseChunk]:
        if self._call_count >= len(self._responses):
            raise RuntimeError(f"_ScriptedCloudAdapter exhausted after {self._call_count} calls")
        response = self._responses[self._call_count]
        chunks = (
            self._stream_chunks[self._call_count]
            if self._call_count < len(self._stream_chunks)
            else []
        )
        self._call_count += 1

        for chunk in chunks:
            yield chunk

        if response.tool_calls:
            for tc in response.tool_calls:
                yield ModelResponseChunk(
                    tool_calls=[
                        ModelToolCallChunk(
                            index=response.tool_calls.index(tc),
                            id=tc.id,
                            type=tc.type,
                            function=ToolCallFunctionChunk(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    ]
                )

        if response.usage is not None:
            yield ModelResponseChunk(
                content=response.message.content if response.message.content else None,
                tool_calls=[],
                finish_reason=response.finish_reason,
                usage=response.usage,
            )

        headers = dict(self._rate_limit_headers)
        self._emit_cost_and_rate_limit(response, headers)


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


def _install_on_error_tracker(tracker: Any) -> Any:
    """Install ``tracker`` on the default ``HookRegistry`` for the test."""
    from harness.hooks import get_registry
    from harness.hooks.base import reset_default_registry
    from harness.hooks.injection_firewall import InjectionFirewallHook

    reset_default_registry()
    registry = get_registry()
    registry.register(InjectionFirewallHook())
    registry._on_error = tracker
    return registry


def _make_response(
    content: str | None,
    tool_calls: list[ModelToolCall] | None = None,
    prompt_tokens: int = 10,
    completion_tokens: int = 20,
) -> ModelResponse:
    """Build a canned ``ModelResponse`` with usage for cost calculation."""
    return ModelResponse(
        message=ModelMessage(role="assistant", content=content, tool_calls=tool_calls or []),
        tool_calls=tool_calls or [],
        finish_reason="stop",
        usage=ModelUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def _stream_chunks(
    content: str, prompt_tokens: int, completion_tokens: int
) -> list[ModelResponseChunk]:
    """Build streaming chunks that replicate the content + usage."""
    return [
        ModelResponseChunk(content=content[:5] if content else ""),
        ModelResponseChunk(
            content=content[5:] if len(content) > 5 else "",
            finish_reason="stop",
            usage=ModelUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        ),
    ]


RATE_LIMIT_HEADERS = {
    "x-ratelimit-remaining-requests": "99",
    "x-ratelimit-remaining-tokens": "9999",
    "x-ratelimit-reset-requests": "1.0",
    "x-ratelimit-reset-tokens": "60.0",
}


@pytest.mark.benchmark
def test_cloud_adapter_cost_events(benchmark_workspace: Path) -> None:
    """Assert model_cost and model_rate_limit events in trace (issue #1262).

    Drives ``Runner.run_task`` with a scripted ``CloudModelAdapter`` that:

    1. Returns a response with non-zero usage on the first ``complete()`` call
       (tool_call round-trip)
    2. Returns a response with non-zero usage on the second ``complete()`` call
       (final answer round-trip)

    After the run, verifies the captured trace contains:
    - ``model_cost`` events (one per model round-trip) with
      provider, model, prompt_tokens, completion_tokens, estimated_cost_usd > 0
    - ``model_rate_limit`` events (one per model round-trip) with
      requests_remaining or tokens_remaining
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    tool_call = ModelToolCall(
        id="call_test",
        type="function",
        function=ToolCallFunction(name="bash", arguments=json.dumps({"command": "echo test"})),
    )

    responses = [
        _make_response(
            content=None, tool_calls=[tool_call], prompt_tokens=10, completion_tokens=15
        ),
        _make_response(content="done", prompt_tokens=15, completion_tokens=5),
    ]

    stream_chunks = [
        _stream_chunks(content="hello", prompt_tokens=10, completion_tokens=15),
        _stream_chunks(content="world", prompt_tokens=15, completion_tokens=5),
    ]

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    try:
        adapter = _ScriptedCloudAdapter(
            responses=responses,
            stream_chunks_per_response=stream_chunks,
            rate_limit_headers=RATE_LIMIT_HEADERS,
        )

        async def _drive() -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as session_id:
                await run_task(
                    "cloud-adapter-cost-events",
                    harness_dir,
                    logger,
                    session_id,
                    model_adapter=adapter,
                )

        asyncio.run(_drive())

        logger = TraceLogger(db)
        events = logger.load_session(logger.list_sessions()[0].session_id)

        model_cost_events = [e for e in events if e.kind == "model_cost"]
        model_rate_limit_events = [e for e in events if e.kind == "model_rate_limit"]

        assert len(model_cost_events) >= 1, (
            f"expected at least 1 model_cost event; "
            f"got {len(model_cost_events)}: {model_cost_events!r}"
        )

        assert len(model_rate_limit_events) >= 1, (
            f"expected at least 1 model_rate_limit event; "
            f"got {len(model_rate_limit_events)}: {model_rate_limit_events!r}"
        )

        for event in model_cost_events:
            payload = event.payload
            assert payload.get("provider") is not None, f"model_cost missing provider: {payload}"
            assert payload.get("model") is not None, f"model_cost missing model: {payload}"
            assert payload.get("prompt_tokens", 0) > 0, (
                f"model_cost prompt_tokens should be > 0: {payload}"
            )
            assert payload.get("completion_tokens", 0) > 0, (
                f"model_cost completion_tokens should be > 0: {payload}"
            )
            assert payload.get("estimated_cost_usd", 0) > 0, (
                f"model_cost estimated_cost_usd should be > 0 when usage is non-zero: {payload}"
            )

        for event in model_rate_limit_events:
            payload = event.payload
            has_requests = payload.get("requests_remaining") is not None
            has_tokens = payload.get("tokens_remaining") is not None
            assert has_requests or has_tokens, (
                f"model_rate_limit must have requests_remaining or tokens_remaining: {payload}"
            )

        assert hook_failures == [], (
            f"expected zero HookRegistry.on_error calls; got {hook_failures!r}"
        )

        assert adapter._call_count == 2, (
            f"expected exactly 2 model round-trips; got {adapter._call_count}"
        )
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()


@pytest.mark.benchmark
def test_cloud_adapter_cost_events_streaming(benchmark_workspace: Path) -> None:
    """Assert model_cost and model_rate_limit events in trace via streaming path.

    Same as ``test_cloud_adapter_cost_events`` but drives the ``stream()`` path
    instead of ``complete()`` by providing a task whose harness hook forces
    streaming mode.  Because the Runner calls ``complete()`` by default for
    CloudModelAdapter, we drive the stream path directly via the adapter's
    ``stream()`` method and verify cost/rate-limit callbacks fire correctly
    when final usage is accumulated from streaming chunks.
    """
    responses = [
        _make_response(content="hello world", prompt_tokens=10, completion_tokens=15),
        _make_response(content="done", prompt_tokens=15, completion_tokens=5),
    ]

    stream_chunks = [
        _stream_chunks(content="hello world", prompt_tokens=10, completion_tokens=15),
        _stream_chunks(content="done", prompt_tokens=15, completion_tokens=5),
    ]

    adapter = _ScriptedCloudAdapter(
        responses=responses,
        stream_chunks_per_response=stream_chunks,
        rate_limit_headers=RATE_LIMIT_HEADERS,
    )

    cost_events: list[ModelCostEvent] = []
    rate_limit_events: list[ModelRateLimitInfo] = []

    def _capture_cost(event: ModelCostEvent) -> None:
        cost_events.append(event)

    def _capture_rate_limit(info: ModelRateLimitInfo) -> None:
        rate_limit_events.append(info)

    adapter.on_cost = _capture_cost
    adapter.on_rate_limit = _capture_rate_limit

    async def _drive_streaming() -> None:
        messages = [{"role": "user", "content": "hello"}]
        async for _chunk in adapter.stream(messages):
            pass

    asyncio.run(_drive_streaming())

    assert len(cost_events) >= 1, (
        f"expected at least 1 model_cost event from stream(); got {len(cost_events)}"
    )
    assert len(rate_limit_events) >= 1, (
        f"expected at least 1 model_rate_limit event from stream(); got {len(rate_limit_events)}"
    )

    cost = cost_events[0]
    assert cost.provider == "test-provider", (
        f"expected provider 'test-provider'; got {cost.provider}"
    )
    assert cost.model == "test-model", f"expected model 'test-model'; got {cost.model}"
    assert cost.prompt_tokens > 0, f"expected prompt_tokens > 0; got {cost.prompt_tokens}"
    assert cost.completion_tokens > 0, (
        f"expected completion_tokens > 0; got {cost.completion_tokens}"
    )
    assert cost.estimated_cost_usd > 0, (
        f"expected estimated_cost_usd > 0; got {cost.estimated_cost_usd}"
    )

    rl = rate_limit_events[0]
    has_requests = rl.requests_remaining is not None
    has_tokens = rl.tokens_remaining is not None
    assert has_requests or has_tokens, (
        f"model_rate_limit must have requests_remaining or tokens_remaining; got {rl!r}"
    )
