"""Model-agnostic test fixtures supporting both mock and real model adapters.

``MockModelAdapter`` provides deterministic, offline test responses with
configurable message content and tool calls. ``RealModelAdapter`` wraps
``OpenAICompatibleAdapter`` for integration testing against a live endpoint.

The ``TEST_MODEL_MODE`` environment variable switches between modes:

- ``mock`` (default): ``MockModelAdapter`` — no network, deterministic responses
- ``real``: ``RealModelAdapter`` — live endpoint via ``build_model_adapter``

Usage in tests::

    async def test_something(model_adapter):
        await run_task("do the task", harness_dir, logger, session_id, model_adapter=model_adapter)

Usage in benchmark tasks::

    @pytest.mark.benchmark
    def test_my_task(benchmark_workspace, model_adapter):
        # ... setup ...
        await run_task("do the task", harness_dir, logger, session_id, model_adapter=model_adapter)
"""

from __future__ import annotations

import os
from collections.abc import AsyncIterator, Sequence

from pydantic import BaseModel

from foundry_x.execution.model_adapter import (
    JsonValue,
    MessageInput,
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ToolCallFunctionChunk,
    ToolInput,
)


class MockModelAdapterConfig(BaseModel):
    """Configuration for a single ``MockModelAdapter`` turn."""

    response_content: str | None = None
    finish_reason: str = "stop"
    tool_calls: list[ModelToolCall] = []


class MockModelAdapter:
    """Deterministic mock ``ModelAdapter`` for offline test runs.

    Each ``complete`` / ``chat`` call returns the configured response.
    ``stream`` yields one content chunk followed by tool-call chunks and
    a finish-reason chunk.

    Example::

        adapter = MockModelAdapter()
        adapter.set_response(MockModelAdapterConfig(
            response_content="hello",
            tool_calls=[...],
        ))
    """

    def __init__(self) -> None:
        self._config = MockModelAdapterConfig()

    def set_response(self, config: MockModelAdapterConfig) -> None:
        self._config = config

    def set_responses(self, configs: list[MockModelAdapterConfig]) -> None:
        self._responses = configs
        self._index = 0

    async def complete(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse:
        return ModelResponse(
            message=ModelMessage(
                role="assistant",
                content=self._config.response_content,
            ),
            tool_calls=self._config.tool_calls,
            finish_reason=self._config.finish_reason,
        )

    async def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse:
        return await self.complete(messages, tools, **kwargs)

    async def stream(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> AsyncIterator[ModelResponseChunk]:
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


def build_model_adapter_from_env(
    env: dict[str, str] | None = None,
):
    """Build a real ``OpenAICompatibleAdapter`` from environment variables.

    Reads ``OPENCODE_SERVER_URL`` / ``LLAMACPP_HOST`` for the endpoint and
    ``FOUNDRY_MODEL_ID`` / ``FOUNDRY_MODEL_API_KEY`` for credentials.
    Raises ``ValueError`` if no endpoint is configured.
    """
    from foundry_x.execution.runner import build_model_adapter

    source = env if env is not None else os.environ
    base_url = (
        source.get("OPENCODE_SERVER_URL", "").strip() or source.get("LLAMACPP_HOST", "").strip()
    )
    if not base_url:
        raise ValueError(
            "TEST_MODEL_MODE=real requires OPENCODE_SERVER_URL or LLAMACPP_HOST to be set"
        )
    return build_model_adapter(source)


def create_model_adapter(
    mode: str | None = None,
    env: dict[str, str] | None = None,
):
    """Create a model adapter based on ``TEST_MODEL_MODE``.

    Args:
        mode: Adapter mode. ``mock`` returns ``MockModelAdapter``.
            ``real`` returns a real adapter via ``build_model_adapter``.
            Defaults to ``TEST_MODEL_MODE`` env var or ``mock``.
        env: Environment dict for real adapter resolution.
    """
    if mode is None:
        mode = os.environ.get("TEST_MODEL_MODE", "mock").strip().lower()

    if mode == "real":
        return build_model_adapter_from_env(env)
    return MockModelAdapter()
