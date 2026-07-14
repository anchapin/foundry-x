# Model configuration for real inference runs

FoundryX is model-agnostic: it drives any OpenAI-compatible chat-completion endpoint
via the [`OpenAICompatibleAdapter`][foundry_x.execution.model_adapter.OpenAICompatibleAdapter].
This document covers everything you need to connect a real model and run the full
evolution loop.

## Contents

1. [Endpoint configuration](#1-endpoint-configuration)
2. [Supported model types](#2-supported-model-types)
3. [Minimum model requirements](#3-minimum-model-requirements)
4. [Local inference with llama.cpp (ROCm)](#4-local-inference-with-llamacpp-rocm)
5. [Troubleshooting](#5-troubleshooting)

---

## 1. Endpoint configuration

FoundryX reads the endpoint from two environment variables, checked in this order:

| Priority | Env var | Default | Purpose |
|---|---|---|---|
| 1 | `OPENCODE_SERVER_URL` | `http://127.0.0.1:4096` | Primary knob — any OpenAI-compatible endpoint |
| 2 | `LLAMACPP_HOST` | `http://127.0.0.1:8080` | Local-first fallback for llama.cpp servers |

Set **one** of these in your `.env` (copy from [`.env.example`](.env.example)).

### Resolving model identity

Three environment variables contribute to the model identity that is stamped into every
trace session ([issue #12](https://github.com/anchapin/foundry-x/issues/12)):

| Env var | Role |
|---|---|
| `FOUNDRY_MODEL_ID` | Explicit override — use this when the other fields are ambiguous |
| `LLAMACPP_MODEL_PATH` | Basename of this path becomes the identity for local GGUF files |
| `OPENCODE_SERVER_URL` | Hostname becomes the identity for remote endpoints |

The resolution order is: `FOUNDRY_MODEL_ID` → `LLAMACPP_MODEL_PATH` basename →
`OPENCODE_SERVER_URL` hostname → unknown.

### API keys

| Env var | Purpose |
|---|---|
| `FOUNDRY_MODEL_API_KEY` | Bearer token sent to the endpoint |
| `OPENAI_API_KEY` | Fallback if `FOUNDRY_MODEL_API_KEY` is unset |

If your endpoint requires no token, leave both unset.

### Request timeouts and retries

| Env var | Default | Purpose |
|---|---|---|
| `FOUNDRY_REQUEST_TIMEOUT_S` | `30` | Per-request round-trip cap in seconds |
| `FOUNDRY_ADAPTER_MAX_RETRIES` | `2` | Retry limit on transient errors (408/429/5xx, connect failures) |

---

## 2. Supported model types

Any endpoint that implements the OpenAI `/chat/completions` REST surface is compatible.
The runner sends a standard JSON body and handles both streaming and non-streaming
responses.

### Verified working

| Model type | Notes |
|---|---|
| **llama.cpp server** (`llama-server`) | Local GGUF inference; see [§4](#4-local-inference-with-llamacpp-rocm) |
| **OpenAI official API** | Set `OPENCODE_SERVER_URL=https://api.openai.com/v1` |
| **Azure OpenAI** | Set `OPENCODE_SERVER_URL=https://<resource>.openai.azure.com/openai/deployments/<deployment>/v1` |
| **Any OpenAI-compatible proxy** | Ollama, LM Studio, LocalAI, Tabby, etc. — must implement `/chat/completions` |

### Endpoint requirements

The endpoint must accept a POST request to `/chat/completions` with a JSON body
containing:

```json
{
  "model": "<model name>",
  "messages": [...],
  "tools": [...]
}
```

Response must be a standard OpenAI-style chat completion object (streaming SSE or
non-streaming JSON).

---

## 3. Minimum model requirements

A model must satisfy all of the following to drive the full evolution loop:

### Context window

| Setting | Recommended minimum | Reason |
|---|---|---|
| Context size | **8192 tokens** | Agent system prompt + tool definitions + conversation history + workspace context can exceed 4k tokens for complex tasks |

For llama.cpp, pass `--ctx-size 8192` (or higher) to `llama-server`.

### Tool-calling support

The evolution loop requires a model that can respond to tool-call requests. Any model
advertising the ability to return structured function calls in `tool_calls` is
sufficient. Specifically:

- The model must handle a `tools` array in the request body describing available
  functions (FoundryX uses the OpenAI function-calling schema).
- The model must return zero or more `tool_calls` in its response.

**Quantization constraints for tool calling**: Some quantized models (Q2, Q3) may
lack the vocabulary resolution to reliably produce valid JSON for function arguments.
Q4_K_M and above are recommended for stable tool-calling performance.

### Recommended model families

| Family | Suggested quantization | Notes |
|---|---|---|
| CodeLlama | Q5_K_M or Q6_K | Strong tool-calling; designed for code |
| Llama 3 / 3.1 | Q4_K_M | 8B variant fits in 8 GB VRAM at Q4 |
| Mistral | Q4_K_M | Good all-round performance |
| Qwen 2 / 2.5 | Q4_K_M | Strong tool-calling, large context variants available |

---

## 4. Local inference with llama.cpp (ROCm)

The [`infra/llama-cpp/README.md`](../infra/llama-cpp/README.md) covers the full build
and run procedure for AMD ROCm. This section is the operator-facing summary.

### Quick start

```bash
# Build llama.cpp with ROCm support (one-time)
./infra/llama-cpp/rocm_setup.sh

# Launch llama-server
./llama.cpp/build/bin/llama-server \
  --model /srv/models/codellama-7b.Q5_K_M.gguf \
  --host 0.0.0.0 --port 8080 \
  --n-gpu-layers 35 \
  --ctx-size 8192

# Point FoundryX at it
echo "LLAMACPP_HOST=http://127.0.0.1:8080" >> .env
echo "LLAMACPP_MODEL_PATH=/srv/models/codellama-7b.Q5_K_M.gguf" >> .env
```

### Smoke test

Verify the server is healthy before running the full loop:

```bash
# Block until /health returns 200, then run a test completion
LLAMACPP_SMOKE_MODEL=/srv/models/codellama-7b.Q5_K_M.gguf \
  ./infra/llama-cpp/rocm_setup.sh --smoke-test
```

### ROCm-specific notes

- **GPU selection**: `HIPCFLAGS="-march=native -mtune=native"` targets the host CPU
  architecture; adjust `-DAMDGPU_TARGETS` to match your GPU (e.g. `gfx1032` for RX 6600 XT).
- **VRAM headroom**: Q5/Q6 quantizations require 8 GB VRAM for full model offload.
  Partial offload (`--n-gpu-layers`) reduces VRAM usage at the cost of speed.
- **Context size vs VRAM**: Larger `--ctx-size` consumes more VRAM for KV cache.
  Monitor `rocm-smi` to ensure you stay within limits.
- **Older kernels**: Set `HSA_OVERRIDE_GFX_VERSION=10.3.0` if you see "agent refused"
  errors on kernels older than 5.14.

---

## 5. Troubleshooting

### "Set OPENCODE_SERVER_URL or LLAMACPP_HOST to an OpenAI-compatible endpoint"

**Cause**: Neither endpoint variable is set in the environment.

**Fix**: Add one to your `.env`:
```bash
echo "OPENCODE_SERVER_URL=http://127.0.0.1:4096" >> .env
```

### Model returns 400 Bad Request

**Common causes**:

1. **Wrong model name**: The `model` field in the request body must match a model
   the server knows. Check what model name your server expects (e.g., `llama-server`
   uses the filename by default, not a user-specified alias).
2. **Missing or malformed `messages`**: The request must include at least one message.
3. **Tool schema too large**: Very long tool descriptions can exceed the context window.
   Reduce the number of tools or shorten descriptions.

### Model hangs on every request

**Cause**: `FOUNDRY_REQUEST_TIMEOUT_S` (default 30 s) is too low for the model size
or hardware.

**Fix**: Increase the timeout in `.env`:
```bash
FOUNDRY_REQUEST_TIMEOUT_S=120
```

### Tool calls are malformed or empty

**Cause**: The model quantization is too aggressive (Q2/Q3) or the model lacks
tool-calling capability.

**Fix**: Use Q4_K_M or better. If the model family does not support tool calling,
switch to one that does (CodeLlama, Llama 3, Mistral, Qwen 2).

### llama-server crashes on startup

**Common causes**:

1. **VRAM exhaustion**: Too many GPU layers requested. Reduce `--n-gpu-layers`.
2. **ROCm version mismatch**: Ensure ROCm 5.7+ is installed and `hipinfo` reports
   the correct GPU. See [`infra/llama-cpp/README.md`](../infra/llama-cpp/README.md)
   for the full GPU support matrix.
3. **Missing model file**: Verify the path passed to `--model` exists and is
   readable.

### Connection refused / model unreachable

**Checklist**:

1. Is the server process running? `ps aux | grep llama-server`
2. Is the port correct? Match `--port` on the server with `LLAMACPP_HOST` or
   `OPENCODE_SERVER_URL` in `.env`.
3. Is the host correct? `127.0.0.1` inside a container must be the container's
   loopback, not the host's. Use the container's hostname or the host's IP from
   the container's perspective.
4. Is a firewall blocking the port? `nc -zv 127.0.0.1 8080` to test connectivity.

### Azure OpenAI returns 401 Unauthorized

**Cause**: `FOUNDRY_MODEL_API_KEY` is unset or incorrect. Azure requires a
key distinct from the standard OpenAI key.

**Fix**: Set your Azure key:
```bash
FOUNDRY_MODEL_API_KEY=your-azure-key-here
```
Note: Azure endpoints also require the deployment name in the URL path —
ensure `OPENCODE_SERVER_URL` matches `https://<resource>.openai.azure.com/openai/deployments/<deployment>/v1`.
