# ADR-0014: Model abstraction layer for model-swapping milestone

## Status

Accepted. 2026-07-13.

## Context

Phase 3 automates the model-swapping milestone. Swapping is mentioned in the
PRD as testing Q4 vs Q5 but nothing captures how the Runner selects a model
binary, how quantization is parameterized, or who owns that configuration.
Without an ADR, the Evolver cannot safely propose model-swap edits without a
shared contract between the Runner and the harness about what a model is.

This is a prequel ADR — it must land before any model-swapping code.

## Decision

We define a `ModelRequirements` pydantic model (ADR-0006 boundary) that
records the minimal set of fields needed to identify and select a model for a
benchmark task. The model abstraction owns these fields; no other subsystem
invents them.

### ModelRequirements fields

```python
class ModelRequirements(BaseModel):
    model_id: str                          # e.g. "codellama-7b", "qwen2.5-coder-7b"
    quantization: str | None = None        # e.g. "Q5_K_M", "q8_0", None = default
    path: str | None = None                # local GGUF path; None = remote endpoint
    endpoint: str | None = None            # OpenAI-compatible URL; None = use harness default
```

- ``model_id`` is the only required field — the minimum needed to name the model.
- ``quantization`` is optional; absent means "use the harness default".
- ``path`` and ``endpoint`` are mutually exclusive hints; at least one must
  be set when the task requires a specific model binary rather than the
  harness default.
- The MVP approach is **config file + CLI flag** (no registry service).

### Model selection ownership

The **Runner** owns model selection at runtime:

1. If a ``BenchmarkTask`` carries a non-null ``model_requirements`` field,
   the Runner uses those fields to configure the ``ModelAdapter``.
2. If ``model_requirements`` is null, the Runner falls back to the harness
   default (environment variables ``OPENCODE_SERVER_URL`` / ``LLAMACPP_HOST``,
   per ``build_model_adapter`` in ``runner.py``).
3. The Evolver may propose ``model_requirements`` values in benchmark task
   definitions; the Critic gates whether the proposal is structurally valid.

This keeps model selection centralized in the Runner rather than scattering
logic across the harness, benchmark definitions, or the Evolver.

### BenchmarkTask extension

``BenchmarkTask`` (defined in ``benchmarks/models.py``) gains an optional
``model_requirements: ModelRequirements | None`` field. The field is nullable
so existing benchmark definitions (authored before this ADR) remain valid and
backwards-compatible.

## Consequences

- The Evolver has a stable, typed contract to propose model-swap edits
  against (model_id, quantization, path/endpoint).
- The Runner has a single place to read per-task model requirements, rather
  than scattering selection logic across multiple code paths.
- ``ModelRequirements`` is a pydantic model at a module boundary (ADR-0006),
  so validation errors surface early with actionable messages.
- The minimal MVP (config file + CLI flag) defers a registry service to a
  future ADR; this ADR only records the abstraction, not the implementation.
- The ``model_requirements`` field on ``BenchmarkTask`` is optional and
  defaults to ``None``, preserving backwards compatibility with all existing
  benchmark definitions.
