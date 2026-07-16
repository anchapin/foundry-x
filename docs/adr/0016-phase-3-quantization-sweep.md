# ADR-0016: Phase 3 quantization sweep

## Status

Accepted.

## Context

Phase 3 requires an ADR before implementation of the quantization sweep
tracked in issue #464. The `infra/llama-cpp/README.md` §"Phase 3 automation"
identifies `Critic` as the natural home, but the sweep introduces a new
multi-run orchestration pattern that differs from the existing single-harness
evaluation.

This ADR captures the decisions needed to implement the sweep without
blocking the evolution loop.

## Decision

### 1. Where the sweep lives

The sweep is a **method on `Critic`** — `Critic.quantization_sweep()`.
A standalone `QuantizationSweep` class is deliberately deferred; the sweep
is an internal Critic orchestration concern, not an independent agent. If the
orchestration logic grows complex enough to warrant extraction, that refactor
gets its own ADR.

### 2. How traces are attributed

Each quantization run is stored in the **same `logs/` SQLite database** as
single-harness runs. Traces are stamped with a distinct `FOUNDRY_MODEL_ID`
that encodes the quantization label (e.g. `qwen2.5-0.5b-q4_k_m`). The
Critic verdict is the authoritative output; raw traces are an implementation
detail.

A separate sweep database is out of scope — it would fragment the existing
trace-driven KPI framework (ADR-0007) without providing enough value to
justify the extra infrastructure.

### 3. Critic verdict for a sweep

A sweep is a **pass-rate aggregation across quantizations**, not a single
failed benchmark gate.

The sweep produces a `QuantizationVerdict`:

```python
class QuantizationVerdict:
    quantizations: list[QuantizationResult]
    recommended: str  # quantization label
    regression: bool  # true if recommended is worse than baseline
```

A `regression = True` blocks the release gate. The baseline is the current
production quantization; each candidate is compared against it. A candidate
passes if its benchmark pass rate is within `FOUNDRY_REGRESSION_THRESHOLD`
of the baseline (default 2 pp).

Individual task-shaped failures (task too large for context budget — see
§6) do **not** constitute a regression. They are excluded from the
pass-rate denominator.

### 4. CI integration

The sweep runs **on every PR from `develop` to `main`** at the release
gate — the same point where the Critic runs the full benchmark suite. It
does **not** run on every push.

This preserves CI budget while ensuring every production release has a
fresh quantization recommendation.

Triggering at the release gate means:
- The sweep result is available before the merge decision.
- It shares the same `develop`-pinned model artifacts as the benchmark suite.
- No new infrastructure (schedulers, webhooks) is needed.

### 5. Relationship to the Evolution loop

The sweep is **purely a diagnostic and attribution tool**. The Evolver
does **not** propose quantization changes.

Rationale: quantization selection is a hardware- and cost-constrained
decision (VRAM floor, inference throughput) that is not reducible to a
score the Evolver can optimise against. The Evolver operates on the harness
(`manifest.json`, `system_prompt.txt`, `skills/`); quantization is a
deployment parameter outside that scope.

If a future milestone adds an automated quantization proposer, that proposal
gets its own ADR.

### 6. Token budget abort classification

`FOUNDRY_TOKEN_BUDGET` is the **hard abort threshold** — when reached,
the task is classified as a task-shaped failure, not a harness regression.

`FOUNDRY_CONTEXT_TOKENS` is the **pruning threshold** — it controls
when the Runner prunes context to stay within budget. These serve different
mechanisms and purposes; they coexist.

A token budget abort indicates the task was too large for the
model/context budget, not a harness defect. The sweep verdict therefore
classifies these as task-shaped failures and excludes them from regression
analysis.

## Consequences

- The `Critic.quantization_sweep()` method becomes the single entry point
  for multi-quantization evaluation, keeping orchestration logic co-located
  with single-run evaluation.
- Same `logs/` database means existing trace KPI queries (ADR-0007) work
  without modification.
- Sweep verdict is a structured `QuantizationVerdict` — CI can consume it
  as a JSON exit code without parsing natural language.
- CI trigger at the release gate avoids per-push overhead while keeping
  every production release evaluated.
- Evolver scope is unchanged; no accidental coupling between harness
  evolution and quantization selection.
