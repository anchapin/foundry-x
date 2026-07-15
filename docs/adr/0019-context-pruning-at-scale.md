# ADR-0019: Context pruning at scale — Phase 3 findings

## Status

Proposed.

## Context

Issue #553 calls for validation and documentation of the context pruning
behavior on the 5600G / 6600 XT hardware target. The
`TokenAwarePruningHook` (issue #465) is wired into `runner.run_task` for
when `FOUNDRY_CONTEXT_TOKENS` is set, but:

1. The pruning threshold has not been empirically validated against the
   VRAM and context-window constraints of the target hardware.
2. The interaction between `FOUNDRY_CONTEXT_TOKENS` (pruning threshold)
   and `FOUNDRY_TOKEN_BUDGET` (hard abort threshold) is documented in
   ADR-0016 §6 but has not been benchmarked.
3. The existing hook drops events but does not summarize them (out of
   scope per issue #553).

This ADR captures the findings from code analysis and the benchmark
methodology required to validate the defaults.

## Decision

### 1. Current pruning architecture

Two pruning mechanisms coexist:

**Event-count pruning** (`ContextPruningHook`, issue #106):
- Triggered when `FOUNDRY_CONTEXT_TOKENS` is **absent or empty**.
- Threshold: `DEFAULT_THRESHOLD = 200` events.
- Drops oldest events whose `kind` is not in `_PRESERVE_KINDS`
  (`tool_result`, `user_prompt`) until count is at or below threshold.
- Suitable for development and for sessions where token counting is
  unavailable from the model endpoint.

**Token-aware pruning** (`TokenAwarePruningHook`, issue #465):
- Triggered when `FOUNDRY_CONTEXT_TOKENS` is **set** to a positive int.
- Threshold: `DEFAULT_TOKEN_THRESHOLD = 8192` tokens (matches the llama.cpp
  `--ctx-size 8192` default on the 6600 XT).
- Queries cumulative `tokens_used` from the most recent `model_response`
  event (written by the runner on every step; see `runner.py:1407`).
- Drops all non-preserved events when session tokens exceed the threshold.
- Suitable for production where `tokens_used` telemetry is available.

Both hooks record a `context_pruned` trace event carrying the dropped
count and the active threshold, enabling post-hoc KPI analysis.

### 2. Hardware constraints

| Hardware | VRAM | Context window | Notes |
|---|---|---|---|
| RX 6600 XT | 8 GB | 8192 tokens (default) | Q5_K_M full offload recommended; see `infra/llama-cpp/README.md` §"ROCm pitfalls" |
| 5600G (APU) | Shared with RAM | 8192 tokens | Integrated GPU; VRAM is system RAM; throughput lower than discrete GPU |

**Key trade-off (infra/llama-cpp/README.md §"ROCm pitfalls"):**
> Watch VRAM headroom when bumping `--ctx-size`: context caching competes
> with model weights.

At `--ctx-size 8192` on the 6600 XT, the KV cache occupies a portion of
the 8 GB VRAM alongside the model weights. Setting
`FOUNDRY_CONTEXT_TOKENS` above the context window is ineffective — the
model will abort before the pruning hook fires. The effective operating
range is **4096–8192 tokens** on this hardware.

### 3. Relationship between FOUNDRY_CONTEXT_TOKENS and FOUNDRY_TOKEN_BUDGET

Per ADR-0016 §6:

| Env var | Role | Behavior when reached |
|---|---|---|
| `FOUNDRY_CONTEXT_TOKENS` | **Pruning threshold** — hook fires to drop events | Session continues with truncated history |
| `FOUNDRY_TOKEN_BUDGET` | **Hard abort threshold** — runner terminates session | Session classified as `task_aborted(reason="token_budget")` |

These serve different purposes and coexist. The pruning hook runs
*before* the hard abort; it is a proactive measure that tries to keep
the session within budget. If pruning is insufficient and
`FOUNDRY_TOKEN_BUDGET` is also set, the runner aborts rather than
allowing the model to process an over-budget prompt.

**Recommended ordering:** `FOUNDRY_CONTEXT_TOKENS` ≤ `FOUNDRY_TOKEN_BUDGET`.
If the pruning threshold exceeds the abort threshold, pruning never fires
before the abort.

### 4. Default recommendation

The current default of **8192 tokens** is:
- **Correct** as an upper bound — it matches the llama.cpp context
  window, so pruning fires before an OOM abort on well-formed tasks.
- **Potentially aggressive** for complex multi-step tasks — a long
  session may prune events that the model needs to maintain coherence.

Preliminary guidance (awaiting benchmark confirmation):

| Value | Use case |
|---|---|
| 4096 | Memory-constrained sessions; short tasks; 5600G APU |
| 8192 | Default; matches `--ctx-size` on 6600 XT |
| 16384 | Tasks requiring longer context; discrete GPU with headroom |

The manifest default (`harness/manifest.json`
`context_pruning.token_threshold`) should remain at **8192** until
benchmark data shows a statistically significant regression or improvement
at a different value.

### 5. Benchmark methodology

When the 6600 XT hardware is available, run the benchmark suite with
three `FOUNDRY_CONTEXT_TOKENS` values:

```bash
for threshold in 4096 8192 16384; do
  FOUNDRY_CONTEXT_TOKENS=$threshold \
    uv run pytest -m benchmark -v \
      --tb=short \
      --log-cli-level=INFO \
      2>&1 | tee "benchmark_context_${threshold}.log"
done
```

Collect from each run:
- **Pass rate** — fraction of benchmark tasks that pass.
- **Average tokens per session** — from `foundry-kpis --harness-version <version>`,
  the `token_totals` map.
- **`token_budget_hit_rate`** — fraction of sessions with
  `task_aborted(reason="token_budget")`. Computed from the trace store:
  ```python
  from foundry_x.trace.logger import TraceLogger
  from foundry_x.observability.kpis import KpiSummary
  # token_budget_abort_count / total_sessions
  ```

Compare pass rates across thresholds. If 4096 yields a statistically
significant regression (>5 pp) compared to 8192, retain 8192 as default.
If 16384 shows no regression and the 5600G can tolerate it, consider
upward adjustment.

### 6. KPI layer integration

The `context_pruned` payload carries the fields needed for post-hoc
efficiency analysis:

```python
# Event-count pruning payload
{"dropped": int, "threshold": int, "token_threshold": int}

# Token-aware pruning payload
{"dropped": int, "threshold_tokens": int, "session_tokens": int}
```

The `foundry-kpis` tool does not yet compute a `context_efficiency` KPI.
When issue #553 acceptance is confirmed, a future ADR should define:

```
context_efficiency = 1 - (dropped_events / total_events_in_session)
```

A value near 1.0 means pruning rarely fired; near 0.0 means heavy pruning
throughout the session.

## Consequences

- The `manifest.json` `context_pruning.token_threshold` default remains at
  **8192** pending benchmark confirmation.
- A follow-up ADR will define `context_efficiency` KPI once benchmark
  data is available.
- The benchmark methodology in §5 enables operators to reproduce the
  validation on their own hardware.
- No code changes are required to implement this ADR; it is a
  documentation and validation artifact.

## Cross-References

- [ADR-0016 §6](./0016-phase-3-quantization-sweep.md#6-token-budget-abort-classification):
  `FOUNDRY_TOKEN_BUDGET` vs. `FOUNDRY_CONTEXT_TOKENS` distinction.
- [`harness/hooks/context_pruning.py`](../../harness/hooks/context_pruning.py):
  `TokenAwarePruningHook` and `ContextPruningHook` implementation.
- [`src/foundry_x/execution/runner.py:1385-1408`](../../src/foundry_x/execution/runner.py):
  Hook wiring in `run_task`.
- [`infra/llama-cpp/README.md`](../../infra/llama-cpp/README.md):
  Hardware targets (5600G / 6600 XT).
- [`docs/CONTEXT.md`](../../docs/CONTEXT.md): `context_pruned` event
  payload contract.
- Issue #465: `TokenAwarePruningHook` implementation.
- Issue #492: Large-session pruning validation.
- Issue #551: `token_budget_hit_rate` KPI.
