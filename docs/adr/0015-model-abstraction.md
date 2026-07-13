# ADR-0015: Model abstraction layer for model-swapping milestone

## Status

Proposed.

## Context

Phase 3 automates the model-swapping milestone (PRD §3). The Runner
currently selects a model binary using a one-off resolution chain in
``runner._resolve_model_request_name`` and ``runner.build_model_adapter``:
``FOUNDRY_MODEL_ID`` → ``LLAMACPP_MODEL_PATH`` basename → server host.
There is no formal contract between the Runner and the benchmark harness
about what fields identify a model, how quantization is parameterized, or
who owns that configuration.

Without an ADR the Evolver cannot safely propose model-swap edits.
This is a prequel ADR — it must land before any model-swapping code.

## Decision

We formalize the model abstraction as follows.

### Model identity fields

A model is identified by three orthogonal fields:

| Field | Type | Description |
|---|---|---|
| ``model_id`` | ``str`` | Stable machine-readable identifier sent in the API request body |
| ``quantization`` | ``str \| None`` | Quantization label from the filename, e.g. ``Q5_K_M`` |
| ``path_or_endpoint`` | ``Path \| str \| None`` | Local GGUF path or remote URL |

These three fields are sufficient to distinguish every model variant we
currently support (local llama.cpp GGUF, OpenAI-compatible remote) and
leave room for future variants (custom GPTQ, vision models) without
breaking the schema.

### Ownership

The **Runner** owns model selection. The benchmark harness and the
Critic receive model identity as a read-only payload stamped into the
trace session (``model_id``); they do not configure model selection.

The ``BenchmarkTask`` structured payload (ADR-0006) carries an optional
``model_requirements: ModelRequirements`` field. When present, the Runner
uses it to override the environment-derived defaults for that task. When
absent, the Runner falls back to the current environment-variable chain.

### Minimal MVP approach (no registry service)

Phase 3 ships with two model-selection mechanisms:

1. **Config file** — ``foundry.yaml`` (or ``foundry.toml``) under the
   ``[model]`` section. Stores the three-field model identity. Read at
   startup; validated before the first request.
2. **CLI flag** — ``--model-id``, ``--quantization``, ``--path-or-endpoint``
   on the runner CLI. Takes precedence over the config file for one-off
   overrides.

A registry service (DNS-based model discovery, hosted model marketplace)
is out of scope for Phase 3. It is explicitly deferred to a future
milestone with its own ADR.

## ModelRequirements pydantic model

``benchmarks/models.py`` defines ``ModelRequirements`` per ADR-0006:

```python
class ModelRequirements(BaseModel):
    model_id: str | None = None
    quantization: str | None = None
    path_or_endpoint: Path | str | None = None
```

``BenchmarkTask`` gains an optional ``model_requirements`` field. It is
omitted for existing benchmarks (backwards compatible); new benchmarks
authored under the model-swapping milestone may include it.

## Consequences

- The Evolver has a stable contract to propose model-swap edits against.
- Model identity is explicit in the trace session, enabling ADR-0007
  improvement-rate KPI to separate success-rate changes from model changes.
- Config + CLI flag MVP avoids a distributed system problem (registry
  consistency, availability) while still giving operators a file-based
  knob.
- ``quantization`` is a plain string; the Runner does not validate
  quantization strings against a allowlist. A future ADR may add
  validation if the Phase 3 rollout surfaces a need.
- Deferring the registry service to a future milestone preserves the
  option to use a different approach (e.g. model cards, HuggingFace
  hub integration) without a breaking change.
