from __future__ import annotations

# trivial change to trigger fresh CI
import asyncio
import json
import os
import random
import re
import warnings
from abc import ABC, abstractmethod
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

    _RATE_LIMIT_HEADERS = (
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    )

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
        on_cost: CostCallback | None = None,
        on_rate_limit: RateLimitCallback | None = None,
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
        self.on_cost = on_cost
        self.on_rate_limit = on_rate_limit

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
        response_data, headers = await self._post_json(request.to_openai_payload())
        response = _parse_completion_response(response_data)
        self._emit_cost_and_rate_limit(response, headers)
        return response

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
            final_usage: ModelUsage | None = None
            last_headers: httpx.Headers | None = response.headers
            try:
                async for line in response.aiter_lines():
                    chunk = _parse_sse_line(line)
                    if chunk is None:
                        continue
                    if chunk.usage is not None:
                        final_usage = chunk.usage
                    yield chunk
            finally:
                await cm.__aexit__(None, None, None)

            if final_usage is not None and last_headers is not None:
                self._emit_cost_and_rate_limit(
                    ModelResponse(
                        message=ModelMessage(role="assistant"),
                        usage=final_usage,
                    ),
                    last_headers,
                )
            if last_headers is not None:
                self._emit_rate_limit(last_headers)
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

    async def _post_json(self, payload: JsonObject) -> tuple[JsonObject, httpx.Headers]:
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
                if status == 429:
                    self._emit_rate_limit(exc.response.headers)
                    backoff_ms = _compute_429_backoff_ms(attempt, exc.response.headers)
                else:
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
            validated = _JSON_OBJECT_ADAPTER.validate_python(data)
            return validated, response.headers

        raise ModelAdapterError("model endpoint request failed: retries exhausted")

    def _emit_cost(self, data: JsonObject) -> None:
        """Emit ``on_cost`` if usage data is present in the response (issue #1165)."""
        if self.on_cost is None:
            return
        usage = data.get("usage")
        if not isinstance(usage, dict):
            return
        prompt_tokens = usage.get("prompt_tokens", 0) or 0
        completion_tokens = usage.get("completion_tokens", 0) or 0
        in_price, out_price = self.token_pricing()
        cost = (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000.0
        self.on_cost(
            ModelCostEvent(
                provider="openai-compatible",
                model=self.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                estimated_cost_usd=round(cost, 8),
            )
        )

    def _emit_rate_limit(self, headers: httpx.Headers) -> None:
        """Emit ``on_rate_limit`` if rate-limit headers are present (issue #1165)."""
        if self.on_rate_limit is None:
            return
        lowered = {key.lower(): value for key, value in headers.items()}
        names_lower = {name.lower() for name in self._RATE_LIMIT_HEADERS}
        if not names_lower.intersection(lowered):
            return

        def _to_int(value: str | None) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except ValueError:
                return None

        def _parse_duration(value: str | None) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value.rstrip("s").rstrip("ms"))
            except ValueError:
                return None

        self.on_rate_limit(
            ModelRateLimitInfo(
                requests_remaining=_to_int(lowered.get("x-ratelimit-remaining-requests")),
                tokens_remaining=_to_int(lowered.get("x-ratelimit-remaining-tokens")),
                requests_reset_seconds=_parse_duration(lowered.get("x-ratelimit-reset-requests")),
                tokens_reset_seconds=_parse_duration(lowered.get("x-ratelimit-reset-tokens")),
            )
        )

    def token_pricing(self) -> tuple[float, float]:
        """Return ``(input_per_1m_usd, output_per_1m_usd)`` for the model."""
        return _resolve_token_pricing(self.model, _OPENAI_PRICING_PER_1M)

    def _emit_cost_and_rate_limit(
        self,
        response: ModelResponse,
        headers: httpx.Headers | Mapping[str, str],
    ) -> None:
        """Fire ``on_cost`` / ``on_rate_limit`` callbacks when wired (issue #1235)."""
        if self.on_cost is not None and response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            in_price, out_price = self.token_pricing()
            cost = (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000.0
            self.on_cost(
                ModelCostEvent(
                    provider="openai-compatible",
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=round(cost, 8),
                )
            )
        if self.on_rate_limit is not None:
            info = self._extract_rate_limit(headers)
            if info is not None:
                self.on_rate_limit(info)

    def _extract_rate_limit(
        self, headers: httpx.Headers | Mapping[str, str]
    ) -> ModelRateLimitInfo | None:
        """Extract rate-limit info from headers if present."""
        lowered = {key.lower(): value for key, value in headers.items()}
        names_lower = {name.lower() for name in self._RATE_LIMIT_HEADERS}
        if not names_lower.intersection(lowered):
            return None
        return self._parse_rate_limit(lowered)

    def _parse_rate_limit(self, headers_lower: Mapping[str, str]) -> ModelRateLimitInfo:
        """Parse rate-limit headers into ModelRateLimitInfo."""

        def _to_int(value: str | None) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except ValueError:
                return None

        def _parse_duration(value: str | None) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value.rstrip("s").rstrip("ms"))
            except ValueError:
                return None

        return ModelRateLimitInfo(
            requests_remaining=_to_int(headers_lower.get("x-ratelimit-remaining-requests")),
            tokens_remaining=_to_int(headers_lower.get("x-ratelimit-remaining-tokens")),
            requests_reset_seconds=_parse_duration(headers_lower.get("x-ratelimit-reset-requests")),
            tokens_reset_seconds=_parse_duration(headers_lower.get("x-ratelimit-reset-tokens")),
        )

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
    return random.randint(0, ceiling)


def _compute_429_backoff_ms(attempt: int, headers: httpx.Headers) -> int:
    """Backoff for HTTP 429 using Retry-After or rate-limit reset headers (issue #1358).

    Priority:
    1. ``Retry-After`` header (seconds, possibly fractional) — used directly + jitter.
    2. ``x-ratelimit-remaining-requests: 0`` + reset header — parsed and used + jitter.
    3. Falls back to exponential jitter via :func:`_compute_backoff_ms`.

    The result is capped at :data:`_MAX_BACKOFF_MS`. A small uniform random
    jitter in ``[0, 500]`` ms is added on top of any header-derived value to
    avoid thundering-herd synchronisation.
    """
    lowered = {k.lower(): v for k, v in headers.items()}

    retry_after = lowered.get("retry-after")
    if retry_after is not None:
        try:
            seconds = float(retry_after)
            backoff_ms = min(int(seconds * 1000), _MAX_BACKOFF_MS)
            return backoff_ms + random.randint(0, 500)
        except ValueError:
            pass

    if lowered.get("x-ratelimit-remaining-requests") == "0":
        reset = lowered.get("x-ratelimit-reset-requests") or lowered.get(
            "anthropic-ratelimit-requests-reset"
        )
        if reset is not None:
            try:
                backoff_ms = min(int(float(reset) * 1000), _MAX_BACKOFF_MS)
                return backoff_ms + random.randint(0, 500)
            except ValueError:
                pass

    return _compute_backoff_ms(attempt)


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


# ---------------------------------------------------------------------------
# Cloud-native model adapters (issue #1041, ADR-0029)
#
# `CloudModelAdapter` is an ABC that formalises provider-specific override
# points (`build_request`, `parse_response`, `parse_stream_chunk`,
# `rate_limit_headers`, `token_pricing`). It shares the bounded-retry and
# SSE plumbing with `OpenAICompatibleAdapter` so callers see the same
# `ModelResponse` / `ModelResponseChunk` contract regardless of provider.
#
# The implementation talks to provider HTTP endpoints directly via
# `httpx` (no SDK dependency). ADR-0029 lists SDKs as one option; we chose
# `httpx` for consistency with `OpenAICompatibleAdapter`, a smaller change
# (AGENTS.md §2 — never widen scope), and zero new dependencies.
# ---------------------------------------------------------------------------


class ModelRateLimitInfo(BaseModel):
    """Provider rate-limit window snapshot parsed from response headers.

    Anthropic surfaces ``anthropic-ratelimit-requests-remaining`` /
    ``anthropic-ratelimit-tokens-remaining``; OpenAI surfaces
    ``x-ratelimit-remaining-requests`` / ``x-ratelimit-remaining-tokens``.
    Adapters normalise both into this single shape so the trace store can
    reason about headroom without branching on provider.
    """

    requests_remaining: int | None = Field(
        default=None,
        description="Remaining requests in the current window, or null when the provider omits it.",
    )
    tokens_remaining: int | None = Field(
        default=None,
        description="Remaining tokens in the current window, or null when the provider omits it.",
    )
    requests_reset_seconds: float | None = Field(
        default=None,
        description="Seconds until the request-window resets, or null when unknown.",
    )
    tokens_reset_seconds: float | None = Field(
        default=None,
        description="Seconds until the token-window resets, or null when unknown.",
    )


class ModelCostEvent(BaseModel):
    """Per-response cost attribution emitted on each successful response.

    Cost is computed from the provider's per-token pricing table
    (`token_pricing`) and the reported `ModelUsage`. The Runner forwards
    this to the trace store so the improvement-rate KPI can attribute
    spend to harness quality, not just model price.
    """

    provider: str = Field(
        min_length=1, description="Adapter provider tag (e.g. 'anthropic', 'openai')."
    )
    model: str = Field(min_length=1, description="Model identifier as sent in the request body.")
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(
        default=0.0,
        ge=0.0,
        description="Best-effort cost estimate in USD; 0.0 when pricing is unknown.",
    )


CostCallback: TypeAlias = Callable[[ModelCostEvent], None]
RateLimitCallback: TypeAlias = Callable[[ModelRateLimitInfo], None]


class CloudModelAdapter(ABC):
    """Base class for cloud-provider model adapters (ADR-0029, issue #1041).

    Concrete subclasses implement the provider-specific override points
    (`build_request`, `parse_response`, `parse_stream_chunk`,
    `rate_limit_headers`, `token_pricing`). The shared `complete` /
    `stream` / `chat` machinery owns the bounded retry loop, SSE framing,
    and cost/rate-limit trace emission so subclasses stay focused on
    wire-format translation.

    The class implements the `ModelAdapter` protocol structurally; it is
    not registered as a Protocol implementation because `ModelAdapter`
    is a `@runtime_checkable` structural protocol and concrete subclasses
    will pass `isinstance` checks by virtue of method shape.
    """

    #: Short provider tag stamped into `ModelCostEvent.provider`.
    provider: str = "cloud"

    def __init__(
        self,
        *,
        model: str,
        base_url: str,
        api_key: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout: float = 60.0,
        max_retries: int = _DEFAULT_ADAPTER_MAX_RETRIES,
        on_retry: RetryCallback | None = None,
        on_cost: CostCallback | None = None,
        on_rate_limit: RateLimitCallback | None = None,
    ) -> None:
        model_name = model.strip()
        if not model_name:
            raise ValueError("model must be a non-empty model identifier")
        base = base_url.strip().rstrip("/")
        if not base:
            raise ValueError("base_url must be a non-empty provider endpoint URL")
        self.model = model_name
        self.base_url = base
        self._client = client or httpx.AsyncClient(timeout=timeout)
        self._owns_client = client is None
        self._auth_headers = self._build_auth_headers(api_key)
        self.max_retries = max_retries
        self.on_retry = on_retry
        self.on_cost = on_cost
        self.on_rate_limit = on_rate_limit

    # --- provider-specific override points -------------------------------

    @abstractmethod
    def _build_auth_headers(self, api_key: str | None) -> dict[str, str]:
        """Return provider-specific auth headers from *api_key*."""

    @abstractmethod
    def build_request(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None,
        *,
        stream: bool,
        extra_params: dict[str, JsonValue],
    ) -> JsonObject:
        """Translate the normalised inputs into the provider's request body."""

    @abstractmethod
    def request_url(self, *, stream: bool) -> str:
        """Absolute URL for the provider's chat/messages endpoint."""

    @abstractmethod
    def request_headers(self, *, stream: bool) -> dict[str, str]:
        """Return per-request headers (auth + content type + version)."""

    @abstractmethod
    def parse_response(self, data: JsonObject) -> ModelResponse:
        """Translate the provider's non-streaming JSON body into `ModelResponse`."""

    @abstractmethod
    def parse_stream_chunk(self, data: JsonObject) -> ModelResponseChunk | None:
        """Translate one decoded SSE event payload into a chunk, or None to skip."""

    @abstractmethod
    def rate_limit_headers(self) -> tuple[str, ...]:
        """Header names this provider exposes rate-limit info under."""

    @abstractmethod
    def token_pricing(self) -> tuple[float, float]:
        """Return ``(input_per_1m_usd, output_per_1m_usd)``; ``(0.0, 0.0)`` when unknown."""

    # --- shared infrastructure -------------------------------------------

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type: object, exc: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def complete(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse:
        payload = self.build_request(messages, tools, stream=False, extra_params=dict(kwargs))
        data, headers = await self._post_json(payload, stream=False)
        response = self.parse_response(data)
        self._emit_cost_and_rate_limit(response, headers)
        return response

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
        payload = self.build_request(messages, tools, stream=True, extra_params=dict(kwargs))
        headers = {**self.request_headers(stream=True)}
        final_usage: ModelUsage | None = None
        last_headers: httpx.Headers | None = None

        for attempt in range(self.max_retries + 1):
            cm = self._client.stream(
                "POST",
                self.request_url(stream=True),
                json=payload,
                headers=headers,
            )
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

            # Phase 2 — stream the body.  Mid-stream failures are NOT retried;
            # issue #200 / #1164 explicitly exclude partially-received SSE from the
            # retry boundary so that CloudModelAdapter matches OpenAICompatibleAdapter.
            try:
                last_headers = response.headers
                async for chunk_data in self._iter_provider_stream(response):
                    if chunk_data.get("usage") is not None:
                        try:
                            final_usage = ModelUsage.model_validate(chunk_data["usage"])
                        except ValueError:
                            final_usage = None
                    chunk = self.parse_stream_chunk(chunk_data)
                    if chunk is None:
                        continue
                    if chunk.usage is not None:
                        final_usage = chunk.usage
                    yield chunk
            finally:
                await cm.__aexit__(None, None, None)
            break

        if final_usage is not None and last_headers is not None:
            self._emit_cost_and_rate_limit(
                ModelResponse(
                    message=ModelMessage(role="assistant"),
                    usage=final_usage,
                ),
                last_headers,
            )

    def _emit_retry(self, attempt: int, exc: Exception, backoff_ms: int) -> None:
        if self.on_retry is None:
            return
        self.on_retry(
            ModelRetryEvent(
                attempt=attempt,
                error_type=type(exc).__name__,
                backoff_ms=backoff_ms,
            )
        )

    def _emit_cost_and_rate_limit(
        self,
        response: ModelResponse,
        headers: httpx.Headers | Mapping[str, str],
    ) -> None:
        """Fire ``on_cost`` / ``on_rate_limit`` callbacks when wired."""
        if self.on_cost is not None and response.usage is not None:
            prompt_tokens = response.usage.prompt_tokens
            completion_tokens = response.usage.completion_tokens
            in_price, out_price = self.token_pricing()
            cost = (prompt_tokens * in_price + completion_tokens * out_price) / 1_000_000.0
            self.on_cost(
                ModelCostEvent(
                    provider=self.provider,
                    model=self.model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    estimated_cost_usd=round(cost, 8),
                )
            )
        if self.on_rate_limit is not None:
            info = self._extract_rate_limit(headers)
            if info is not None:
                self.on_rate_limit(info)

    def _extract_rate_limit(
        self, headers: httpx.Headers | Mapping[str, str]
    ) -> ModelRateLimitInfo | None:
        names = self.rate_limit_headers()
        present = {name.lower() for name in names}
        lowered = {key.lower(): value for key, value in headers.items()}
        if not present.intersection(lowered):
            return None
        return self.parse_rate_limit(lowered)

    def parse_rate_limit(self, headers_lower: Mapping[str, str]) -> ModelRateLimitInfo:
        """Default no-op rate-limit parser; providers override as needed."""
        return ModelRateLimitInfo()

    async def _iter_provider_stream(self, response: httpx.Response) -> AsyncIterator[JsonObject]:
        """Default SSE parser that decodes ``data: <json>`` frames.

        Providers whose SSE framing differs (Anthropic emits
        ``event:`` / ``data:`` pairs) override this.
        """
        async for line in response.aiter_lines():
            if not line or not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if payload == "[DONE]" or not payload:
                continue
            try:
                data = json.loads(payload)
            except json.JSONDecodeError as exc:
                raise ModelAdapterResponseError("stream chunk was not valid JSON") from exc
            yield _JSON_OBJECT_ADAPTER.validate_python(data)

    async def _post_json(
        self,
        payload: JsonObject,
        *,
        stream: bool,
    ) -> tuple[JsonObject, httpx.Headers]:
        """POST *payload* with bounded retry; return (json_body, response_headers)."""
        headers = self.request_headers(stream=stream)
        url = self.request_url(stream=stream)
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(url, json=payload, headers=headers)
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                status = exc.response.status_code
                if not _is_retryable_status(status) or attempt >= self.max_retries:
                    raise ModelAdapterHTTPError(
                        status_code=status,
                        response_body=exc.response.text,
                    ) from exc
                if status == 429:
                    backoff_ms = _compute_429_backoff_ms(attempt, exc.response.headers)
                else:
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
            return _JSON_OBJECT_ADAPTER.validate_python(data), response.headers

        raise ModelAdapterError("model endpoint request failed: retries exhausted")


def _validate_messages(
    messages: Sequence[MessageInput],
) -> list[ModelMessage]:
    if not messages:
        raise ValueError("messages must be a non-empty sequence")
    return [ModelMessage.model_validate(message) for message in messages]


def _validate_tools(
    tools: Sequence[ToolInput] | None,
) -> list[ToolDefinition] | None:
    if tools is None:
        return None
    return [ToolDefinition.model_validate(tool) for tool in tools]


_GO_DURATION_RE = re.compile(
    r"(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?(?:(?P<ms>\d+)ms)?"
)


def _parse_go_duration(value: str | None) -> float | None:
    """Parse Go ``time.Duration.String()`` output (e.g. ``2h30m``, ``1m0s``, ``500ms``)."""
    if value is None or value == "":
        return None
    match = _GO_DURATION_RE.fullmatch(value.strip())
    if match is None:
        return None
    parts = {k: int(v) for k, v in match.groupdict(default="0").items()}
    total = parts["hours"] * 3600 + parts["minutes"] * 60 + parts["seconds"] + parts["ms"] / 1000.0
    return float(total) if total > 0 else 0.0


class AnthropicAdapter(CloudModelAdapter):
    """CloudModelAdapter for Anthropic's native `/v1/messages` API.

    - Endpoint: ``POST {base_url}/v1/messages``
    - Auth: ``x-api-key`` header (NOT ``Authorization: Bearer``)
    - Streaming: SSE frames with ``event:`` + ``data:`` lines
      (``message_start``, ``content_block_delta``, ``message_delta``, ``message_stop``)
    - Rate-limit headers: ``anthropic-ratelimit-requests-*`` / ``anthropic-ratelimit-tokens-*``
    """

    provider = "anthropic"
    _ANTHROPIC_VERSION = "2023-06-01"

    def _build_auth_headers(self, api_key: str | None) -> dict[str, str]:
        if api_key is None or not api_key.strip():
            raise ValueError("AnthropicAdapter requires an api_key (ANTHROPIC_API_KEY was not set)")
        return {"x-api-key": api_key.strip()}

    def request_url(self, *, stream: bool) -> str:
        return f"{self.base_url}/v1/messages"

    def request_headers(self, *, stream: bool) -> dict[str, str]:
        headers = {
            "content-type": "application/json",
            "anthropic-version": self._ANTHROPIC_VERSION,
        }
        if stream:
            headers["accept"] = "text/event-stream"
        headers.update(self._auth_headers)
        return headers

    def build_request(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None,
        *,
        stream: bool,
        extra_params: dict[str, JsonValue],
    ) -> JsonObject:
        validated = _validate_messages(messages)
        system_parts = [m.content for m in validated if m.role == "system" and m.content]
        system_text = "\n\n".join(system_parts) if system_parts else None
        convo = [m for m in validated if m.role != "system"]

        payload: dict[str, JsonValue] = {
            "model": self.model,
            "messages": [
                m.model_dump(mode="json", exclude_none=True, exclude={"name", "tool_call_id"})
                for m in convo
            ],
            "stream": stream,
        }
        if system_text is not None:
            payload["system"] = system_text
        validated_tools = _validate_tools(tools)
        if validated_tools:
            payload["tools"] = [
                {
                    "name": t.function.name,
                    "description": t.function.description,
                    "input_schema": t.function.parameters,
                }
                for t in validated_tools
            ]
        conflicts = sorted(_RESERVED_REQUEST_KEYS.intersection(extra_params))
        if conflicts:
            raise ValueError(
                f"extra request parameters conflict with reserved keys: {', '.join(conflicts)}"
            )
        payload.update(extra_params)
        return _JSON_OBJECT_ADAPTER.validate_python(payload)

    def parse_response(self, data: JsonObject) -> ModelResponse:
        try:
            role = str(data.get("role", "assistant"))
            content_blocks = data.get("content")
            text_parts: list[str] = []
            tool_calls: list[ModelToolCall] = []
            if isinstance(content_blocks, list):
                for block in content_blocks:
                    if not isinstance(block, dict):
                        continue
                    block_type = block.get("type")
                    if block_type == "text" and isinstance(block.get("text"), str):
                        text_parts.append(str(block["text"]))
                    elif block_type == "tool_use":
                        tool_calls.append(
                            ModelToolCall(
                                id=str(block.get("id", "")),
                                type="function",
                                function=ToolCallFunction(
                                    name=str(block.get("name", "")),
                                    arguments=json.dumps(block.get("input", {})),
                                ),
                            )
                        )
            stop_reason = data.get("stop_reason")
            usage_obj = data.get("usage")
            usage: ModelUsage | None = None
            if isinstance(usage_obj, dict):
                usage = ModelUsage(
                    prompt_tokens=int(usage_obj.get("input_tokens", 0) or 0),
                    completion_tokens=int(usage_obj.get("output_tokens", 0) or 0),
                    total_tokens=int(usage_obj.get("input_tokens", 0) or 0)
                    + int(usage_obj.get("output_tokens", 0) or 0),
                )
            message = ModelMessage(role=role, content="".join(text_parts) or None)
            return ModelResponse(
                message=message,
                tool_calls=tool_calls,
                finish_reason=str(stop_reason) if stop_reason is not None else None,
                usage=usage,
            )
        except (ValueError, TypeError) as exc:
            raise ModelAdapterResponseError(
                "Anthropic response did not match the /v1/messages schema"
            ) from exc

    def parse_stream_chunk(self, data: JsonObject) -> ModelResponseChunk | None:
        event_type = str(data.get("__event_type", "") or data.get("type", ""))
        if event_type == "content_block_start":
            content_block = data.get("content_block")
            if isinstance(content_block, dict) and content_block.get("type") == "tool_use":
                block_index = data.get("index", 0)
                tool_id = content_block.get("id")
                tool_name = content_block.get("name")
                tool_type = content_block.get("type")
                if not hasattr(self, "_content_block_to_tool_call_index"):
                    self._content_block_to_tool_call_index: dict[int, int] = {}
                if block_index not in self._content_block_to_tool_call_index:
                    self._content_block_to_tool_call_index[block_index] = len(
                        self._content_block_to_tool_call_index
                    )
                if not hasattr(self, "_tool_call_name_by_block_index"):
                    self._tool_call_name_by_block_index: dict[int, str] = {}
                self._tool_call_name_by_block_index[block_index] = tool_name
                tc_index = self._content_block_to_tool_call_index[block_index]
                return ModelResponseChunk(
                    tool_calls=[
                        ModelToolCallChunk(
                            index=tc_index,
                            id=tool_id,
                            type=tool_type,
                            function=ToolCallFunctionChunk(name=tool_name),
                        )
                    ]
                )
            return None
        if event_type == "content_block_delta":
            delta = data.get("delta")
            if isinstance(delta, dict) and delta.get("type") == "text_delta":
                text = delta.get("text")
                if isinstance(text, str) and text:
                    return ModelResponseChunk(content=text)
            if isinstance(delta, dict) and delta.get("type") == "tool_use":
                block_index = data.get("index", 0)
                input_json = delta.get("input_json", "")
                if not hasattr(self, "_content_block_to_tool_call_index"):
                    self._content_block_to_tool_call_index: dict[int, int] = {}
                if block_index not in self._content_block_to_tool_call_index:
                    self._content_block_to_tool_call_index[block_index] = len(
                        self._content_block_to_tool_call_index
                    )
                tc_index = self._content_block_to_tool_call_index[block_index]
                tool_name = getattr(self, "_tool_call_name_by_block_index", {}).get(block_index)
                return ModelResponseChunk(
                    tool_calls=[
                        ModelToolCallChunk(
                            index=tc_index,
                            function=ToolCallFunctionChunk(
                                name=tool_name,
                                arguments=input_json if isinstance(input_json, str) else "",
                            ),
                        )
                    ]
                )
            return None
        if event_type == "message_delta":
            delta = data.get("delta")
            usage_data = data.get("usage")
            stop_reason: str | None = None
            if isinstance(delta, dict) and delta.get("stop_reason") is not None:
                stop_reason = str(delta["stop_reason"])
            usage: ModelUsage | None = None
            if isinstance(usage_data, dict) and "output_tokens" in usage_data:
                usage = ModelUsage(
                    prompt_tokens=0,
                    completion_tokens=int(usage_data.get("output_tokens", 0) or 0),
                    total_tokens=int(usage_data.get("output_tokens", 0) or 0),
                )
            if stop_reason is None and usage is None:
                return None
            return ModelResponseChunk(finish_reason=stop_reason, usage=usage)
        if event_type == "message_start":
            message = data.get("message")
            if isinstance(message, dict):
                usage_obj = message.get("usage")
                if isinstance(usage_obj, dict) and "input_tokens" in usage_obj:
                    return ModelResponseChunk(
                        usage=ModelUsage(
                            prompt_tokens=int(usage_obj.get("input_tokens", 0) or 0),
                            completion_tokens=0,
                            total_tokens=int(usage_obj.get("input_tokens", 0) or 0),
                        )
                    )
            return None
        if event_type == "message_stop":
            return None
        return None

    async def _iter_provider_stream(self, response: httpx.Response) -> AsyncIterator[JsonObject]:
        event_name = ""
        async for line in response.aiter_lines():
            stripped = line.strip()
            if not stripped:
                event_name = ""
                continue
            if stripped.startswith("event:"):
                event_name = stripped.removeprefix("event:").strip()
                continue
            if stripped.startswith("data:"):
                payload = stripped.removeprefix("data:").strip()
                if not payload:
                    continue
                try:
                    data = json.loads(payload)
                except json.JSONDecodeError as exc:
                    raise ModelAdapterResponseError(
                        "Anthropic stream chunk was not valid JSON"
                    ) from exc
                obj = _JSON_OBJECT_ADAPTER.validate_python(data)
                if isinstance(obj, dict) and event_name:
                    obj["__event_type"] = event_name
                yield obj

    def rate_limit_headers(self) -> tuple[str, ...]:
        return (
            "anthropic-ratelimit-requests-remaining",
            "anthropic-ratelimit-requests-reset",
            "anthropic-ratelimit-tokens-remaining",
            "anthropic-ratelimit-tokens-reset",
        )

    def parse_rate_limit(self, headers_lower: Mapping[str, str]) -> ModelRateLimitInfo:
        def _to_int(value: str | None) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except ValueError:
                return None

        return ModelRateLimitInfo(
            requests_remaining=_to_int(headers_lower.get("anthropic-ratelimit-requests-remaining")),
            tokens_remaining=_to_int(headers_lower.get("anthropic-ratelimit-tokens-remaining")),
            requests_reset_seconds=_parse_go_duration(
                headers_lower.get("anthropic-ratelimit-requests-reset")
            ),
            tokens_reset_seconds=_parse_go_duration(
                headers_lower.get("anthropic-ratelimit-tokens-reset")
            ),
        )

    def token_pricing(self) -> tuple[float, float]:
        return _resolve_token_pricing(self.model, _ANTHROPIC_PRICING_PER_1M)


def _resolve_token_pricing(
    model: str, hardcoded: dict[str, tuple[float, float]]
) -> tuple[float, float]:
    """Return ``(input_per_1m_usd, output_per_1m_usd)`` for *model*.

    Checks ``FOUNDRY_MODEL_PRICING_<MODEL>`` env var first (format:
    ``input,output``, e.g. ``3.0,15.0``). Falls back to *hardcoded*
    table. Returns ``(0.0, 0.0)`` when pricing is unknown.
    """
    env_key = f"FOUNDRY_MODEL_PRICING_{model.upper().replace('-', '_')}"
    raw = os.environ.get(env_key)
    if raw is not None:
        parts = raw.split(",")
        if len(parts) == 2:
            try:
                return (float(parts[0]), float(parts[1]))
            except ValueError:
                warnings.warn(
                    f"Invalid pricing in {env_key}={raw!r}; expected 'input,output' float pair; "
                    f"falling back to hardcoded table.",
                    RuntimeWarning,
                    stacklevel=2,
                )
        else:
            warnings.warn(
                f"Invalid pricing in {env_key}={raw!r}; expected 'input,output' float pair; "
                f"falling back to hardcoded table.",
                RuntimeWarning,
                stacklevel=2,
            )
    return hardcoded.get(model, (0.0, 0.0))


_ANTHROPIC_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.8, 4.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
}


class OpenAINativeAdapter(CloudModelAdapter):
    """CloudModelAdapter for OpenAI's native chat-completions API.

    Differs from `OpenAICompatibleAdapter` in that it surfaces the
    OpenAI-native rate-limit headers (``x-ratelimit-*``) and the
    OpenAI-native error envelope (``{"error": {"message", "type", "code"}}``),
    which the OpenAI-compatible adapter does not model. The wire body is
    still the OpenAI `/chat/completions` shape, so `build_request`
    reuses `ModelRequest.to_openai_payload()`.
    """

    provider = "openai"

    def _build_auth_headers(self, api_key: str | None) -> dict[str, str]:
        if api_key is None or not api_key.strip():
            raise ValueError("OpenAINativeAdapter requires an api_key (OPENAI_API_KEY was not set)")
        token = api_key.strip()
        if token.lower().startswith("bearer "):
            return {"Authorization": token}
        return {"Authorization": f"Bearer {token}"}

    def request_url(self, *, stream: bool) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/v1"):
            return f"{base}/chat/completions"
        return f"{base}/v1/chat/completions"

    def request_headers(self, *, stream: bool) -> dict[str, str]:
        headers = {"content-type": "application/json"}
        if stream:
            headers["accept"] = "text/event-stream"
        headers.update(self._auth_headers)
        return headers

    def build_request(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None,
        *,
        stream: bool,
        extra_params: dict[str, JsonValue],
    ) -> JsonObject:
        request = _build_request(
            self.model, messages, tools, stream=stream, extra_params=extra_params
        )
        return request.to_openai_payload()

    def parse_response(self, data: JsonObject) -> ModelResponse:
        error = data.get("error")
        if isinstance(error, dict):
            raise ModelAdapterResponseError(
                f"OpenAI native error: {error.get('message', 'unknown')}"
            )
        return _parse_completion_response(data)

    def parse_stream_chunk(self, data: JsonObject) -> ModelResponseChunk | None:
        return _parse_openai_stream_chunk(data)

    def rate_limit_headers(self) -> tuple[str, ...]:
        return (
            "x-ratelimit-remaining-requests",
            "x-ratelimit-remaining-tokens",
            "x-ratelimit-reset-requests",
            "x-ratelimit-reset-tokens",
        )

    def parse_rate_limit(self, headers_lower: Mapping[str, str]) -> ModelRateLimitInfo:
        def _to_int(value: str | None) -> int | None:
            if value is None or value == "":
                return None
            try:
                return int(value)
            except ValueError:
                return None

        def _parse_duration(value: str | None) -> float | None:
            if value is None or value == "":
                return None
            try:
                return float(value.rstrip("s").rstrip("ms"))
            except ValueError:
                return None

        return ModelRateLimitInfo(
            requests_remaining=_to_int(headers_lower.get("x-ratelimit-remaining-requests")),
            tokens_remaining=_to_int(headers_lower.get("x-ratelimit-remaining-tokens")),
            requests_reset_seconds=_parse_duration(headers_lower.get("x-ratelimit-reset-requests")),
            tokens_reset_seconds=_parse_duration(headers_lower.get("x-ratelimit-reset-tokens")),
        )

    def token_pricing(self) -> tuple[float, float]:
        return _resolve_token_pricing(self.model, _OPENAI_PRICING_PER_1M)


_OPENAI_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "gpt-4o": (2.5, 10.0),
    "gpt-4o-mini": (0.15, 0.6),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
}


def _parse_openai_stream_chunk(data: JsonObject) -> ModelResponseChunk | None:
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
    if parsed.usage is not None:
        return ModelResponseChunk(usage=parsed.usage)
    return None


_ADAPTER_PREFIXES: dict[str, str] = {
    "anthropic/": "anthropic",
    "openai/": "openai",
}

_ADAPTER_CLASSES: dict[str, type[ModelAdapter]] = {
    "AnthropicAdapter": AnthropicAdapter,
    "OpenAINativeAdapter": OpenAINativeAdapter,
    "OpenAICompatibleAdapter": OpenAICompatibleAdapter,
}


def resolve_model_adapter(
    model_id: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 60.0,
    max_retries: int = _DEFAULT_ADAPTER_MAX_RETRIES,
    client: httpx.AsyncClient | None = None,
    on_retry: RetryCallback | None = None,
    on_cost: CostCallback | None = None,
    on_rate_limit: RateLimitCallback | None = None,
) -> ModelAdapter:
    """Resolve the adapter class for *model_id* and construct it (ADR-0029 §5).

    Environment variable ``FOUNDRY_MODEL_ADAPTER`` overrides the resolved adapter
    class (checked before prefix-based routing). Valid values:
    ``AnthropicAdapter``, ``OpenAINativeAdapter``, ``OpenAICompatibleAdapter``.
    Unknown values raise ``ValueError`` at resolution time.

    Prefix matching (highest precedence after env-var override):

    | Prefix        | Adapter                |
    | ------------- | ---------------------- |
    | ``anthropic/``| `AnthropicAdapter`     |
    | ``openai/``   | `OpenAINativeAdapter`  |
    | (otherwise)   | `OpenAICompatibleAdapter` |

    The prefix is stripped from *model_id* before it is forwarded to the
    provider so the request body carries the bare provider model name
    (e.g. ``claude-3-5-sonnet-20241022`` not ``anthropic/claude-...``).
    """
    if FOUNDRY_MODEL_ADAPTER := os.environ.get("FOUNDRY_MODEL_ADAPTER"):
        if FOUNDRY_MODEL_ADAPTER not in _ADAPTER_CLASSES:
            valid = ", ".join(sorted(_ADAPTER_CLASSES))
            raise ValueError(
                f"Unknown FOUNDRY_MODEL_ADAPTER value {FOUNDRY_MODEL_ADAPTER!r}. "
                f"Valid values: {valid}"
            )
        if FOUNDRY_MODEL_ADAPTER == "AnthropicAdapter":
            for prefix in ("anthropic/",):
                if model_id.startswith(prefix):
                    bare = model_id.removeprefix(prefix)
                    break
            else:
                bare = model_id
            resolved_base = base_url or "https://api.anthropic.com"
            return AnthropicAdapter(
                model=bare,
                base_url=resolved_base,
                api_key=api_key,
                client=client,
                timeout=timeout,
                max_retries=max_retries,
                on_retry=on_retry,
                on_cost=on_cost,
                on_rate_limit=on_rate_limit,
            )
        if FOUNDRY_MODEL_ADAPTER == "OpenAINativeAdapter":
            for prefix in ("openai/",):
                if model_id.startswith(prefix):
                    bare = model_id.removeprefix(prefix)
                    break
            else:
                bare = model_id
            resolved_base = base_url or "https://api.openai.com"
            return OpenAINativeAdapter(
                model=bare,
                base_url=resolved_base,
                api_key=api_key,
                client=client,
                timeout=timeout,
                max_retries=max_retries,
                on_retry=on_retry,
                on_cost=on_cost,
                on_rate_limit=on_rate_limit,
            )
        resolved_base = base_url or ""
        if not resolved_base:
            raise ValueError(
                "OpenAICompatibleAdapter resolution requires a base_url; "
                "set OPENCODE_SERVER_URL or pass base_url explicitly"
            )
        return OpenAICompatibleAdapter(
            base_url=resolved_base,
            model=model_id,
            api_key=api_key,
            client=client,
            timeout=timeout,
            max_retries=max_retries,
            on_retry=on_retry,
            on_cost=on_cost,
            on_rate_limit=on_rate_limit,
        )

    for prefix, provider in _ADAPTER_PREFIXES.items():
        if model_id.startswith(prefix):
            bare = model_id.removeprefix(prefix)
            if provider == "anthropic":
                resolved_base = base_url or "https://api.anthropic.com"
                return AnthropicAdapter(
                    model=bare,
                    base_url=resolved_base,
                    api_key=api_key,
                    client=client,
                    timeout=timeout,
                    max_retries=max_retries,
                    on_retry=on_retry,
                    on_cost=on_cost,
                    on_rate_limit=on_rate_limit,
                )
            if provider == "openai":
                resolved_base = base_url or "https://api.openai.com"
                return OpenAINativeAdapter(
                    model=bare,
                    base_url=resolved_base,
                    api_key=api_key,
                    client=client,
                    timeout=timeout,
                    max_retries=max_retries,
                    on_retry=on_retry,
                    on_cost=on_cost,
                    on_rate_limit=on_rate_limit,
                )

    resolved_base = base_url or ""
    if not resolved_base:
        raise ValueError(
            "OpenAICompatibleAdapter resolution requires a base_url; "
            "set OPENCODE_SERVER_URL or pass base_url explicitly"
        )
    return OpenAICompatibleAdapter(
        base_url=resolved_base,
        model=model_id,
        api_key=api_key,
        client=client,
        timeout=timeout,
        max_retries=max_retries,
        on_retry=on_retry,
        on_cost=on_cost,
        on_rate_limit=on_rate_limit,
    )
