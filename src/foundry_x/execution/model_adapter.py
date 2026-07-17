from __future__ import annotations

import asyncio
import json
import os
import random
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from typing import Literal, Protocol, Self, TypeAlias, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter

JsonObject: TypeAlias = dict[str, JsonValue]

_JSON_OBJECT_ADAPTER = TypeAdapter(JsonObject)
_RESERVED_REQUEST_KEYS = frozenset({"model", "messages", "tools", "stream"})

# Bounded retry on transient failures (issue #200). A single llama-server
# hiccup (5xx, connect failure, read timeout) must not abort a multi-step
# agent loop. Retries are bounded by FOUNDRY_ADAPTER_MAX_RETRIES (default 2)
# and fire only on explicitly transient conditions — never on 4xx client
# errors (other than 408/429), never on response-parsing failures, and
# never on exceptions from an injected skill executor.
_ADAPTER_MAX_RETRIES_ENV = "FOUNDRY_ADAPTER_MAX_RETRIES"
_DEFAULT_ADAPTER_MAX_RETRIES = 2
_RETRYABLE_STATUS_CODES: frozenset[int] = frozenset({408, 429, 500, 502, 503, 504})
_RETRYABLE_TRANSPORT_ERRORS: tuple[type[Exception], ...] = (
    httpx.ConnectError,
    httpx.ReadTimeout,
    httpx.RemoteProtocolError,
)
_BASE_BACKOFF_MS = 500
_MAX_BACKOFF_MS = 8000


class ToolCallFunction(BaseModel):
    """Function-call payload returned by an OpenAI-compatible model."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    arguments: str = ""


class ModelToolCall(BaseModel):
    """A complete tool call emitted by a chat-completion response."""

    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=1)
    type: Literal["function"] = "function"
    function: ToolCallFunction


class ModelMessage(BaseModel):
    """Chat message exchanged with a model adapter (ADR-0006 boundary)."""

    model_config = ConfigDict(extra="ignore")

    role: str = Field(min_length=1)
    content: str | None = None
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ModelToolCall] | None = None


class ToolFunctionSchema(BaseModel):
    """OpenAI-compatible function schema for a callable tool."""

    model_config = ConfigDict(extra="ignore")

    name: str = Field(min_length=1)
    description: str | None = None
    parameters: JsonObject = Field(default_factory=dict)


class ToolDefinition(BaseModel):
    """OpenAI-compatible tool definition serialized into the request body."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["function"] = "function"
    function: ToolFunctionSchema


class ModelRequest(BaseModel):
    """Validated OpenAI-compatible chat-completion request."""

    model: str = Field(min_length=1)
    messages: list[ModelMessage] = Field(min_length=1)
    tools: list[ToolDefinition] | None = None
    stream: bool = False
    extra_params: JsonObject = Field(default_factory=dict)

    def to_openai_payload(self) -> JsonObject:
        """Return the wire-format JSON body for `/chat/completions`."""
        payload = self.model_dump(
            mode="json",
            exclude_none=True,
            exclude={"extra_params"},
        )
        payload.update(self.extra_params)
        return _JSON_OBJECT_ADAPTER.validate_python(payload)


class ModelUsage(BaseModel):
    """Token-usage accounting carried in a chat-completion response (issue #197).

    All three counters default to ``0`` rather than being required so an
    OpenAI-compatible endpoint that omits (for example) ``prompt_tokens``
    still parses into a usable object — the runner only reads
    ``usage.total_tokens`` to enforce ``FOUNDRY_TOKEN_BUDGET``, and a
    missing token field is conservatively counted as zero (ADR-0006
    pydantic discipline: a real ``0`` is preferable to a ``None`` that
    would force every consumer to gate on ``is None``).

    The shape mirrors the OpenAI-compatible ``usage`` object
    (``prompt_tokens`` / ``completion_tokens`` / ``total_tokens``) so
    wire-format JSON can be parsed without reshaping.
    """

    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)


class ModelResponse(BaseModel):
    """Normalized non-streaming response returned by a ModelAdapter."""

    message: ModelMessage
    tool_calls: list[ModelToolCall] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: ModelUsage | None = None


class ToolCallFunctionChunk(BaseModel):
    """Partial function-call payload from a streaming delta."""

    model_config = ConfigDict(extra="ignore")

    name: str | None = None
    arguments: str | None = None


class ModelToolCallChunk(BaseModel):
    """Partial tool call emitted by an OpenAI-compatible streaming delta."""

    model_config = ConfigDict(extra="ignore")

    index: int | None = None
    id: str | None = None
    type: str | None = None
    function: ToolCallFunctionChunk | None = None


class ModelResponseChunk(BaseModel):
    """Normalized streaming response chunk returned by a ModelAdapter."""

    content: str | None = None
    tool_calls: list[ModelToolCallChunk] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: ModelUsage | None = None


class _OpenAIChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: ModelMessage | None = None
    finish_reason: str | None = None


class _OpenAIChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choices: list[_OpenAIChoice] = Field(min_length=1)
    usage: ModelUsage | None = None


class _OpenAIStreamDelta(BaseModel):
    model_config = ConfigDict(extra="ignore")

    content: str | None = None
    tool_calls: list[ModelToolCallChunk] | None = None


class _OpenAIStreamChoice(BaseModel):
    model_config = ConfigDict(extra="ignore")

    delta: _OpenAIStreamDelta | None = None
    finish_reason: str | None = None


class _OpenAIChatCompletionChunk(BaseModel):
    model_config = ConfigDict(extra="ignore")

    choices: list[_OpenAIStreamChoice] = Field(default_factory=list)
    usage: ModelUsage | None = None


class ModelAdapterError(RuntimeError):
    """Base error for model adapter failures."""


class ModelAdapterHTTPError(ModelAdapterError):
    """Raised when the OpenAI-compatible endpoint returns a non-2xx response."""

    def __init__(self, status_code: int, response_body: str) -> None:
        self.status_code = status_code
        self.response_body = response_body
        super().__init__(f"model endpoint returned HTTP {status_code}: {response_body}")


class ModelAdapterResponseError(ModelAdapterError):
    """Raised when the endpoint response cannot be parsed as a model response."""


class ModelRetryEvent(BaseModel):
    """Emitted when the adapter retries a transient failure (issue #200).

    Each retry during ``_post_json`` or the SSE ``stream`` path produces one
    event. The payload is fully typed (no ``Any``) so the trace store and the
    Phase 2 Digester can reason about retry behaviour without schema drift.
    """

    attempt: int = Field(ge=1, description="1-based index of the failed attempt.")
    error_type: str = Field(min_length=1, description="Class name of the retried exception.")
    backoff_ms: int = Field(ge=0, description="Jittered backoff (ms) before the next attempt.")


RetryCallback: TypeAlias = Callable[[ModelRetryEvent], None]


MessageInput: TypeAlias = ModelMessage | dict[str, JsonValue]
ToolInput: TypeAlias = ToolDefinition | dict[str, JsonValue]


@runtime_checkable
class ModelAdapter(Protocol):
    """Protocol for model-agnostic chat completion backends."""

    async def complete(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse:
        """Return one full response for the provided chat messages."""

    async def stream(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> AsyncIterator[ModelResponseChunk]:
        """Yield normalized chunks from a streaming chat completion."""

    async def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse:
        """Compatibility alias for callers that use chat terminology."""


class OpenAICompatibleAdapter(ModelAdapter):
    """ModelAdapter backed by an OpenAI-compatible chat-completions API."""

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        chat_completions_path: str | None = None,
        max_retries: int = _DEFAULT_ADAPTER_MAX_RETRIES,
        on_retry: RetryCallback | None = None,
    ) -> None:
        base = base_url.strip().rstrip("/")
        if not base:
            raise ValueError("base_url must be a non-empty OpenAI-compatible endpoint URL")
        model_name = model.strip()
        if not model_name:
            raise ValueError("model must be a non-empty chat-completion model name")

        self.base_url = base
        self.model = model_name
        self.chat_completions_path = chat_completions_path or _default_chat_completions_path(base)
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._headers = _auth_headers(api_key)
        self.max_retries = max_retries
        self.on_retry = on_retry

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the owned HTTP client, leaving injected clients to their owner."""
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse:
        request = _build_request(self.model, messages, tools, stream=False, extra_params=kwargs)
        response = await self._post_json(request.to_openai_payload())
        return _parse_completion_response(response)

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
        request = _build_request(self.model, messages, tools, stream=True, extra_params=kwargs)
        payload = request.to_openai_payload()

        for attempt in range(self.max_retries + 1):
            cm = self._client.stream(
                "POST",
                self._chat_completions_url,
                json=payload,
                headers=self._headers,
            )
            # Phase 1 — establish connection and check status (retryable).
            try:
                response = await cm.__aenter__()
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise ModelAdapterError(
                        f"model endpoint request failed: {exc}",
                    ) from exc
                backoff_ms = _compute_backoff_ms(attempt)
                self._emit_retry(attempt + 1, exc, backoff_ms)
                await asyncio.sleep(backoff_ms / 1000)
                continue
            except httpx.HTTPError as exc:
                raise ModelAdapterError(
                    f"model endpoint request failed: {exc}",
                ) from exc

            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                await cm.__aexit__(type(exc), exc, exc.__traceback__)
                status = exc.response.status_code
                if not _is_retryable_status(status) or attempt >= self.max_retries:
                    raise ModelAdapterHTTPError(
                        status_code=status,
                        response_body=exc.response.text,
                    ) from exc
                backoff_ms = _compute_backoff_ms(attempt)
                self._emit_retry(attempt + 1, exc, backoff_ms)
                await asyncio.sleep(backoff_ms / 1000)
                continue

            # Phase 2 — stream the body.  Mid-stream failures are NOT
            # retried; issue #200 explicitly excludes partially-received
            # SSE from the retry boundary.
            try:
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue
                    yield chunk
            finally:
                await cm.__aexit__(None, None, None)
            return

    def _emit_retry(self, attempt: int, exc: Exception, backoff_ms: int) -> None:
        """Invoke the ``on_retry`` callback if one is wired (issue #200)."""
        if self.on_retry is None:
            return
        self.on_retry(
            ModelRetryEvent(
                attempt=attempt,
                error_type=type(exc).__name__,
                backoff_ms=backoff_ms,
            )
        )

    async def _post_json(self, payload: JsonObject) -> JsonObject:
        """POST *payload* with bounded retry on transient failures (issue #200).

        Retries fire only on ``httpx.ConnectError``, ``httpx.ReadTimeout``,
        ``httpx.RemoteProtocolError``, and HTTP 408 / 429 / 5xx — never on
        other 4xx, ``ModelAdapterResponseError``, or any non-HTTP exception.
        Each retry invokes ``on_retry`` (if wired) with a
        :class:`ModelRetryEvent` before the jittered backoff sleep.
        """
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    self._chat_completions_url,
                    json=payload,
                    headers=self._headers,
                )
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if not _is_retryable_status(status) or attempt >= self.max_retries:
                    raise ModelAdapterHTTPError(
                        status_code=status,
                        response_body=exc.response.text,
                    ) from exc
                backoff_ms = _compute_backoff_ms(attempt)
                self._emit_retry(attempt + 1, exc, backoff_ms)
                await asyncio.sleep(backoff_ms / 1000)
                continue
            except _RETRYABLE_TRANSPORT_ERRORS as exc:
                if attempt >= self.max_retries:
                    raise ModelAdapterError(
                        f"model endpoint request failed: {exc}",
                    ) from exc
                backoff_ms = _compute_backoff_ms(attempt)
                self._emit_retry(attempt + 1, exc, backoff_ms)
                await asyncio.sleep(backoff_ms / 1000)
                continue
            except httpx.HTTPError as exc:
                raise ModelAdapterError(
                    f"model endpoint request failed: {exc}",
                ) from exc

            try:
                data = response.json()
            except json.JSONDecodeError as exc:
                raise ModelAdapterResponseError("model endpoint returned invalid JSON") from exc
            return _JSON_OBJECT_ADAPTER.validate_python(data)

        raise ModelAdapterError("model endpoint request failed: retries exhausted")

    @property
    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/{self.chat_completions_path.lstrip('/')}"


def _auth_headers(api_key: str | None) -> dict[str, str]:
    if api_key is None or not api_key.strip():
        return {}
    token = api_key.strip()
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}


def _default_chat_completions_path(base_url: str) -> str:
    path = httpx.URL(base_url).path.rstrip("/")
    if path.endswith("/v1") or path == "v1":
        return "/chat/completions"
    return "/v1/chat/completions"


def resolve_adapter_max_retries(env: Mapping[str, str] | None = None) -> int:
    """Resolve the adapter retry cap from ``FOUNDRY_ADAPTER_MAX_RETRIES``.

    An empty / absent value yields :data:`_DEFAULT_ADAPTER_MAX_RETRIES`
    (``2``). A non-negative integer overrides it; ``0`` disables retries
    entirely (equivalent to the pre-issue-#200 single-shot behaviour).
    A non-integer value propagates :class:`ValueError` so a typo in
    ``.env`` surfaces at startup (AGENTS.md §2).
    """
    source = env if env is not None else os.environ
    raw = source.get(_ADAPTER_MAX_RETRIES_ENV, "").strip()
    if not raw:
        return _DEFAULT_ADAPTER_MAX_RETRIES
    value = int(raw)
    return max(0, value)


def _compute_backoff_ms(attempt: int) -> int:
    """Exponential backoff with full jitter for retry attempt *attempt*.

    ``attempt`` is 0-based (the first retry is attempt 0). The ceiling
    doubles each attempt starting from :data:`_BASE_BACKOFF_MS`, capped at
    :data:`_MAX_BACKOFF_MS`. The actual sleep is a uniform random value in
    ``[0, ceiling]`` (full-jitter strategy, AWS Architecture Blog) so
    concurrent retry bursts do not synchronise.
    """
    ceiling = min(_BASE_BACKOFF_MS * (2**attempt), _MAX_BACKOFF_MS)
    return random.randint(0, ceiling)  # noqa: S311 — jitter, not crypto


def _is_retryable_status(status_code: int) -> bool:
    """Return ``True`` for HTTP status codes that warrant a retry."""
    return status_code in _RETRYABLE_STATUS_CODES


def _build_request(
    model: str,
    messages: Sequence[MessageInput],
    tools: Sequence[ToolInput] | None,
    *,
    stream: bool,
    extra_params: dict[str, JsonValue],
) -> ModelRequest:
    validated_extra = _JSON_OBJECT_ADAPTER.validate_python(dict(extra_params))
    conflicts = sorted(_RESERVED_REQUEST_KEYS.intersection(validated_extra))
    if conflicts:
        joined = ", ".join(conflicts)
        raise ValueError(f"extra request parameters conflict with reserved keys: {joined}")

    return ModelRequest(
        model=model,
        messages=[ModelMessage.model_validate(message) for message in messages],
        tools=[ToolDefinition.model_validate(tool) for tool in tools]
        if tools is not None
        else None,
        stream=stream,
        extra_params=validated_extra,
    )


def _parse_completion_response(data: JsonObject) -> ModelResponse:
    try:
        parsed = _OpenAIChatCompletionResponse.model_validate(data)
    except ValueError as exc:
        raise ModelAdapterResponseError(
            "model endpoint response did not match chat schema"
        ) from exc
    choice = parsed.choices[0]
    if choice.message is None:
        raise ModelAdapterResponseError("model endpoint response did not include a message")
    return ModelResponse(
        message=choice.message,
        tool_calls=choice.message.tool_calls or [],
        finish_reason=choice.finish_reason,
        usage=parsed.usage,
    )


def _parse_sse_line(line: str) -> ModelResponseChunk | None:
    if not line or not line.startswith("data:"):
        return None
    payload = line.removeprefix("data:").strip()
    if payload == "[DONE]":
        return None
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ModelAdapterResponseError("stream chunk was not valid JSON") from exc
    try:
        parsed = _OpenAIChatCompletionChunk.model_validate(data)
    except ValueError as exc:
        raise ModelAdapterResponseError("stream chunk did not match chat schema") from exc
    if parsed.choices:
        choice = parsed.choices[0]
        delta = choice.delta or _OpenAIStreamDelta()
        return ModelResponseChunk(
            content=delta.content,
            tool_calls=delta.tool_calls or [],
            finish_reason=choice.finish_reason,
            usage=parsed.usage,
        )
    # Usage-only chunk (empty ``choices``): OpenAI-compatible servers emit a
    # terminal chunk carrying just the top-level ``usage`` object when
    # ``stream_options.include_usage`` is set. Forward it so the runner can
    # accumulate ``total_tokens`` against ``FOUNDRY_TOKEN_BUDGET`` (issue #197).
    if parsed.usage is not None:
        return ModelResponseChunk(usage=parsed.usage)
    return None
