# ADR-0025: Cross-model-family sweeps

## Status

Proposed.

## Context

The quantization sweep (ADR-0016) automates selection of the best
quantization (Q4_K_S, Q5_K_M, Q6_K, …) for a single model family
(e.g. qwen2.5-0.5b). It sweeps quantizations along one axis only.

A parallel use case — benchmarking across model families — is unserved.
An operator may want to know: which model family at its recommended
quantization best solves the task suite? Or: does a smaller model family
(phi-3-mini) outperform a larger one (llama-3.2-1b) at equal quantization
cost?

Supporting this requires:

- Routing requests to different server endpoints per model family.
- Discovering which model families and quantizations are available.
- Attributing benchmark results to the correct model family in the trace
  store.
- Aggregating per-family results into a cross-family verdict.

ADR-0014 deferred a registry service to a future milestone. This ADR
re-opens that thread: a lightweight in-process registry via an environment
variable is sufficient for the sweep use case and does not require a
hosted service.

## Decision

### 1. FOUNDRY_MODEL_REGISTRY environment variable

`FOUNDRY_MODEL_REGISTRY` is a JSON dict mapping model-family names to
configuration objects:

```json
{
  "qwen2.5-0.5b": {
    "path": "/models/qwen2.5-0.5b-q4_k_m.gguf",
    "quantization": "Q4_K_M",
    "endpoint": "http://localhost:8080"
  },
  "llama-3.2-1b": {
    "path": "/models/llama-3.2-1b-q4_k_m.gguf",
    "quantization": "Q4_K_M",
    "endpoint": "http://localhost:8081"
  }
}
```

The registry is read once at `foundry-sweep` startup and validated
before the first run. Missing keys are not fatal — a family with no
registry entry falls back to the existing `FOUNDRY_MODEL_PATH` /
`FOUNDRY_MODEL_ID` / endpoint resolution chain.

`FOUNDRY_MODEL_REGISTRY` is intentionally not a config file. A future
ADR may promote it to a file-based registry (ADR-0014 §6 deferred this).

### 2. --model-families CLI flag

`foundry-sweep` (defined in `src/foundry_x/evolution/cli.py`) gains a
`--model-families` flag:

```
foundry-sweep --model-families qwen2.5-0.5b,llama-3.2-1b,phi-3-mini
```

When absent, behaviour is unchanged (single-family quantization sweep).

The flag accepts a comma-separated list of model-family identifiers. Each
identifier is looked up in `FOUNDRY_MODEL_REGISTRY`. If a family is not
in the registry, the sweep skips it with a warning and continues.

### 3. ModelFamilySweepResult pydantic model

`src/foundry_x/evolution/critic.py` defines a new model in the same
module where `QuantizationSweepResult` and `QuantizationVerdict` live:

```python
class QuantizationResult(BaseModel):
    quantization: str
    passed: int
    failed: int
    task_errors: int
    duration_s: float


class ModelFamilySweepResult(BaseModel):
    model_family: str
    results: list[QuantizationResult]
    recommended: str
    regression: bool
```

`ModelFamilySweepResult` aggregates the per-family `QuantizationResult`
list produced by the existing `quantization_sweep()` method. The
`recommended` and `regression` fields mirror `QuantizationVerdict` so
the outer sweep can compute an overall recommendation without knowing the
inner representation.

### 4. Two-axis sweep iteration

The sweep iterates in two nested loops:

```
for model_family in model_families:
    for quantization in quantizations:
        run benchmark with family config
```

Each inner iteration sets `FOUNDRY_MODEL_ID` to
`{model_family}-{quantization}` (e.g. `qwen2.5-0.5b-q4_k_m`) so the
trace store correctly attributes every session.

The outer aggregation step builds `ModelFamilySweepResult` per family.

### 5. Trace store attribution

Every session in the sweep carries `model_id` in `session_start` (ADR-0007)
stamped with the fully qualified name
`{model_family}-{quantization}`. This enables post-hoc slice by family
using the existing `foundry-x-trace` tooling without schema changes.

### 6. ModelFamilyVerdict

A new `ModelFamilyVerdict` extends `QuantizationVerdict` (ADR-0016):

```python
class ModelFamilyVerdict(BaseModel):
    family_results: list[ModelFamilySweepResult]
    recommended_family: str
    recommended_quantization: str
    regression: bool
    regression_by_family: dict[str, bool]
```

`recommended_family` and `recommended_quantization` together name the
single best combination. `regression_by_family` lets the CI gate inspect
whether a regression is isolated to one family or is widespread.

`regression = True` blocks the release gate. A family-level
`regression_by_family[family] = True` does not independently block —
only the aggregate verdict does.

### 7. Critic.quantization_sweep() extension

`Critic.quantization_sweep()` in `src/foundry_x/evolution/critic.py`
accepts an optional `model_families: list[str] | None` parameter.
When `None`, the method behaves as before (single-family sweep).
When set, it delegates to a new private
`_model_family_sweep()` method which orchestrates the two-axis iteration
and aggregation.

The existing single-family code path is unchanged; multi-family is a
strict extension.

## Alternatives considered

### A. Separate databases per model family

Each family writes to its own SQLite file. Rejected: this fragments the
existing trace-driven KPI framework (ADR-0007) and makes cross-family
queries (e.g. "which family solved the most tasks") require multi-database
aggregation that the current tooling does not support. The same
`logs/` database with qualified `model_id` achieves clean attribution
without infrastructure churn.

### B. Hosted registry service (DNS discovery)

A separate process (or DNS-based discovery) to track available models.
Rejected for Phase 3: it adds a distributed systems problem (registry
consistency, availability, deployment) before the sweep use case has
validated the schema. The env-var registry is a bounded, locally
testable intermediate step.

### C. Sweep as a standalone CLI (not on Critic)

A separate `foundry-family-sweep` binary that does not go through the
Critic. Rejected: the Critic already owns the benchmark execution,
verdict aggregation, and CI integration; adding a parallel path duplicates
orchestration logic and splits the trace store usage.

### D. Implicit family discovery from filesystem

Scan `FOUNDRY_MODEL_PATH` parent directory for subfolders named after
model families. Rejected: the mapping from model family to exact GGUF
path and endpoint is not recoverable from the filesystem alone; an
explicit registry entry is required for any non-trivial deployment.

## Consequences

- `foundry-sweep --model-families` enables cross-model-family benchmark
  comparison without changing the single-family path.
- `FOUNDRY_MODEL_REGISTRY` is a minimal, non-breaking extension — absent
  env var leaves the existing code path untouched.
- Trace attribution uses qualified `model_id` strings; existing KPI queries
  (ADR-0007) continue to work, and new queries can filter by family prefix.
- `ModelFamilyVerdict` propagates the `regression` signal to CI without
  requiring CI to parse per-family details.
- The registry concept introduced here is compatible with a future
  file-based or hosted registry (ADR-0014 §6), so long as the JSON shape
  is preserved.
- `QuantizationResult` and `QuantizationVerdict` are unchanged; the new
  models compose around them rather than modifying them.

## Follow-ups

- File-based model registry (`foundry.yaml` `[model_registry]` section) to
  replace the env var for multi-family production deployments. This is
  the natural evolution of the deferred registry from ADR-0014 §6.
- `foundry-sweep --list-families` flag to query the registry and report
  which families would be swept without running them.
- Sweep result dashboard: render `ModelFamilyVerdict` as a per-family
  pass-rate table in the regression report.
