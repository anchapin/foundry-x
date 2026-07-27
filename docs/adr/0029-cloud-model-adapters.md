# ADR-0029: Cloud-native model adapters for Anthropic and OpenAI

## Status

Proposed.

## Context

Issue #1041: the project currently supports OpenAI-compatible model adapters via
`OpenAICompatibleAdapter` (`src/foundry_x/execution/model_adapter.py`). There is no
support for native Anthropic API (Claude) or native OpenAI API (GPT-4o, o1) which
have different API shapes, streaming formats, and authentication schemes.

`OpenAICompatibleAdapter` works for any server that implements the OpenAI
`/chat/completions` wire format. The Anthropic Claude API and the native OpenAI
API do not speak that format: they use different request schemas, different SSE
event shapes, and different auth mechanisms. Wrapping them behind the existing
adapter is possible but produces a leaky abstraction — the `ModelRequest.to_openai_payload()`
serialisation and the SSE parser in `_parse_sse_line()` would need to branch on
model name, defeating the purpose of the adapter interface.

This ADR introduces two first-party adapter implementations and formally documents
the interface contract all adapters must follow, complementing the model identity
fields defined in ADR-0014.

## Decision

### 1. Interface all adapters must implement

Every model adapter is a class implementing the `ModelAdapter` protocol defined in
`src/foundry_x/execution/model_adapter.py:238–264`:

```python
@runtime_checkable
class ModelAdapter(Protocol):
    async def complete(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse: ...

    async def stream(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> AsyncIterator[ModelResponseChunk]: ...

    async def chat(
        self,
        messages: Sequence[MessageInput],
        tools: Sequence[ToolInput] | None = None,
        **kwargs: JsonValue,
    ) -> ModelResponse: ...
```

The three methods share the same signature contract. Adapters MUST:

- Accept `MessageInput` (`ModelMessage | dict[str, JsonValue]`) and convert
  `dict` inputs to `ModelMessage` via `ModelMessage.model_validate()`.
- Accept `ToolInput` (`ToolDefinition | dict[str, JsonValue]`) and convert
  `dict` inputs analogously.
- Return `ModelResponse` from `complete`/`chat`; yield `ModelResponseChunk`
  from `stream`.
- Be `runtime_checkable` (the protocol uses `@runtime_checkable`).
- Implement `__aenter__` / `__aexit__` for async context manager usage.

#### Normalised response types

All adapters return the same Pydantic types regardless of the underlying API:

| Type | Module location | Purpose |
|---|---|---|
| `ModelResponse` | `model_adapter.py:127–134` | Non-streaming response |
| `ModelResponseChunk` | `model_adapter.py:156–163` | Streaming delta |
| `ModelUsage` | `model_adapter.py:106–125` | Token accounting |
| `ModelMessage` | `model_adapter.py:55–65` | Chat message |
| `ModelToolCall` / `ModelToolCallChunk` | `model_adapter.py:45–53`, `145–154` | Tool calls |

These types are the **only** output contract. Adapters must reshape their
provider's native response into these types — no adapter returns its provider's
raw response type to callers.

### 2. Streaming contract

`stream()` MUST return an `AsyncIterator[ModelResponseChunk]`. Each yielded
chunk represents one incremental delta from the provider. The contract:

- The first chunk MAY carry `content: str | None` with the first text delta.
- Subsequent text deltas arrive in subsequent chunks.
- Tool-call chunks carry `tool_calls: list[ModelToolCallChunk]` with
  `index`, `id`, `type`, and a `function` sub-object that accumulates
  `name` and `arguments` fragments.
- A terminal chunk carries `finish_reason: str | None` on the last non-usage
  chunk.
- If the provider emits a usage-only final chunk (OpenAI `stream_options.include_usage`),
  the adapter forwards it as a `ModelResponseChunk` with only `usage` set; the Runner
  accumulates it for `FOUNDRY_TOKEN_BUDGET` enforcement (issue #197).
- The iterator must not yield `None`. Return early (no yield) if the stream
  is empty.
- The SSE line format (`data:` prefix, `data: [DONE]` sentinel) is an internal
  concern of `OpenAICompatibleAdapter`; other adapters use whatever format their
  SDK emits, converting to `ModelResponseChunk` before yielding.

### 3. Error handling

Three error classes are defined in `model_adapter.py`:

| Class | Base | When raised |
|---|---|---|
| `ModelAdapterError` | `RuntimeError` | Base; transport-level or unexpected failures |
| `ModelAdapterHTTPError` | `ModelAdapterError` | Non-2xx response; carries `status_code` and `response_body` |
| `ModelAdapterResponseError` | `ModelAdapterError` | Response body fails schema validation |

All three are raised by adapter implementations. Callers (the `Runner`) catches
`ModelAdapterError` and emits a `model_error` trace event (CONTEXT.md §Event kinds).

Retry behaviour for transient failures (5xx, 408, 429, transport errors) is
governed by `FOUNDRY_ADAPTER_MAX_RETRIES` (default 2) and the exponential
backoff helper `_compute_backoff_ms`. The `OpenAICompatibleAdapter` retry
implementation (`_post_json`, `stream`) is the reference implementation; new
adapters should replicate the same retryable-error set and backoff formula.
The `ModelRetryEvent` pydantic model (`model_adapter.py:217–228`) MUST be emitted
via the `on_retry` callback (if wired) on each retry attempt.

### 4. Authentication

#### `OpenAICompatibleAdapter` (existing)

Reads `api_key` from the constructor or resolves it from the environment.
Uses `Bearer` token auth:

```python
def _auth_headers(api_key: str | None) -> dict[str, str]:
    if api_key is None or not api_key.strip():
        return {}
    token = api_key.strip()
    if token.lower().startswith("bearer "):
        return {"Authorization": token}
    return {"Authorization": f"Bearer {token}"}
```

No API key is required for local llama.cpp servers that disable auth.

#### `AnthropicModelAdapter` (new)

Uses the `anthropic` Python SDK. Auth is via `ANTHROPIC_API_KEY` environment
variable set on the `anthropic.Anthropic` client instance. The adapter
resolves the key via `os.environ.get("ANTHROPIC_API_KEY")` at construction
time; a missing key raises `ValueError` at construction, not at first request.

Model ID prefix: `anthropic/` (e.g., `anthropic/claude-3-5-sonnet-20241022`).
The runner resolves adapter type by prefix matching on `model_id`.

#### `OpenAIModelAdapter` (new)

Uses the `openai` Python SDK. Auth is via `OPENAI_API_KEY` environment
variable set on the `openai.OpenAI` client instance, resolved at construction
time — missing key raises `ValueError`.

Model ID prefix: `openai/` (e.g., `openai/gpt-4o`, `openai/o1`).
For Azure OpenAI, a separate adapter or configuration path is out of scope for
this ADR; it can be addressed in a follow-up ADR.

### 5. Adapter registration

Adapter resolution follows the prefix-matching scheme described in the issue:

| Model ID prefix | Adapter |
|---|---|
| `anthropic/` | `AnthropicModelAdapter` |
| `openai/` | `OpenAIModelAdapter` |
| (all others) | `OpenAICompatibleAdapter` |

The `FOUNDRY_MODEL_ADAPTER` env var (existing) can override the resolved adapter
class for custom endpoints (e.g., setting `FOUNDRY_MODEL_ADAPTER=OpenAICompatibleAdapter`
for a third-party OpenAI-compatible server using a non-standard model ID).

Registration is implemented in the factory function that builds adapters
(`runner.build_model_adapter` or a new `resolve_model_adapter` helper in
`model_adapter.py`), using a dictionary keyed by prefix string.

### 6. SDK dependencies

| Adapter | Package | pyproject.toml group |
|---|---|---|
| `AnthropicModelAdapter` | `anthropic` | `anthropic` |
| `OpenAIModelAdapter` | `openai` | `openai` |
| `OpenAICompatibleAdapter` | (no new dep; uses `httpx`) | existing |

New dependencies are added via `uv add` and the lockfile updated via `uv sync`
per ADR-0002.

## Consequences

- Two new adapter classes implement the full `ModelAdapter` protocol and are
  indistinguishable from `OpenAICompatibleAdapter` from the caller's perspective.
- Streaming and non-streaming modes are supported for both cloud providers.
- The normalised response schema (`ModelResponse`, `ModelResponseChunk`, `ModelUsage`)
  is the same for all adapters, so the `Runner` and `Digester` need no changes.
- The `ModelAdapter` protocol is now explicitly documented as the adapter contract,
  complementing the model identity fields in ADR-0014.
- Adding a new cloud provider in the future requires: (a) a new adapter class,
  (b) a new prefix entry in the resolution table, (c) a new `pyproject.toml` entry.
  No interface changes are needed.
- Anthropic tool-use (function calling) support requires mapping their
  `assistant_message.tool_use` schema to `ModelToolCall`. The specifics of that
  mapping are an implementation detail documented in the adapter code.
- This ADR does not address vision/multimodal models, batch APIs, or
  token counting endpoints; those can be addressed as follow-up issues with
  their own ADRs.
