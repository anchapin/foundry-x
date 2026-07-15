# ADR-0020: Context Pruning at Scale Findings

## Status

Proposed.

## Context

Issue #553 tracks the validation and documentation of context pruning at scale behavior. The `TokenAwarePruningHook` (issue #465) is wired into `runner.run_task` for when `FOUNDRY_CONTEXT_TOKENS` is set, but the pruning threshold has not been empirically validated for the 5600G / 6600 XT hardware constraints.

This ADR captures:
1. What the benchmark sweep was designed to measure
2. The current state of benchmark data
3. The relationship between `FOUNDRY_CONTEXT_TOKENS` and `FOUNDRY_TOKEN_BUDGET`
4. General-knowledge recommendations until live sweep data is collected

## Background: Token-Aware Pruning Mechanism

### How it works

`TokenAwarePruningHook` (`harness/hooks/context_pruning.py`) intercepts every `pre_tool` call and queries cumulative `tokens_used` from `model_response` trace events. When the token count exceeds `token_threshold`, the hook drops the oldest non-preserved events (`tool_result` and `user_prompt` are always kept) and emits a `context_pruned` trace event.

The `context_pruned` payload shape:

```
kind="context_pruned"
payload={
  "dropped": <int>,           -- events removed
  "threshold_tokens": <int>,  -- trigger threshold
  "session_tokens": <int>     -- tokens at time of prune
}
```

### Relationship to FOUNDRY_TOKEN_BUDGET

Per ADR-0016 §6:

| Parameter | Purpose | Behavior when exceeded |
|-----------|---------|----------------------|
| `FOUNDRY_CONTEXT_TOKENS` | Pruning threshold | Drops old events via `TokenAwarePruningHook`; task continues |
| `FOUNDRY_TOKEN_BUDGET` | Hard abort threshold | `task_aborted(reason="token_budget")`; task fails |

These serve different purposes and coexist. A token budget abort indicates the task was too large for the model/context budget, not a harness defect.

## Current Status

The pruning infrastructure is implemented and tests exist (`tests/harness/test_context_pruning.py`). However, **a systematic benchmark sweep across `FOUNDRY_CONTEXT_TOKENS` values has not been executed** with results persisted to the trace store.

The benchmark sweep would run the full benchmark suite (`uv run pytest -m benchmark`) against multiple `FOUNDRY_CONTEXT_TOKENS` values (e.g., 4096, 8192, 16384) to establish:
- Benchmark pass rate vs. context token threshold
- Average tokens per session (from `model_response.token_usage`)
- `token_budget_hit_rate` — sessions that would have triggered `task_aborted(reason='token_budget')`

## Benchmark Sweep Design

### What to measure

For each `FOUNDRY_CONTEXT_TOKENS` value under test:

| Metric | Description | How to obtain |
|--------|-------------|---------------|
| Pass rate | Benchmark pass rate at this threshold | `passed_tasks / total_tasks` |
| Avg session tokens | Mean `tokens_used` per session | AVG from `model_response` events |
| Prune rate | How often pruning triggers | COUNT `context_pruned` events |
| Token budget hit rate | Sessions hitting hard abort | COUNT `task_aborted(reason='token_budget')` |
| Prune efficiency | Token reduction per prune | AVG `session_tokens - threshold_tokens` at prune time |

### SQL queries for sweep analysis

```sql
-- Prune event rate per session
SELECT session_id, COUNT(*) as prune_count
FROM events WHERE kind = 'context_pruned'
GROUP BY session_id;

-- Token budget aborts
SELECT COUNT(*) FROM events
WHERE kind = 'task_aborted'
  AND json_extract(payload, '$.reason') = 'token_budget';

-- Average tokens at prune time
SELECT AVG(json_extract(payload, '$.session_tokens'))
FROM events WHERE kind = 'context_pruned';
```

### Threshold values to test

| Threshold | Rationale |
|-----------|-----------|
| 4096 | Lower bound — aggressive pruning, may lose context |
| 8192 | Current default — moderate pruning |
| 16384 | Upper bound — minimal pruning, may hit token budget |

## General Knowledge: Token Threshold Tradeoffs

Until live sweep data is available, the following represents generally understood tradeoffs for 7B coding models on RX 6600 XT (8 GB VRAM):

### Threshold Tradeoffs

| Threshold | Context Retention | Memory Pressure | Risk |
|-----------|-----------------|-----------------|------|
| **4096** | Low — aggressive pruning | Minimal | May lose intermediate context |
| **8192** | Medium — balanced | Moderate | Current default; generally safe |
| **16384** | High — minimal pruning | Higher | May hit token budget on long sessions |

### Key Observations

1. **8192 is the generally recommended default** for 7B models on the RX 6600 XT. It provides sufficient context for most benchmark tasks without excessive memory pressure.

2. **Lower thresholds (4096) may cause context loss** on complex multi-step tasks where the model needs to reference earlier parts of the conversation.

3. **Higher thresholds (16384) risk hitting token budget** on long sessions but preserve more context for the model.

4. **The optimal threshold depends on task complexity** — simpler tasks may work fine with aggressive pruning, while complex refactoring tasks may need more context.

## Threshold Recommendations (Pending Benchmark Data)

| Model Size | Recommended `token_threshold` | Rationale |
|------------|------------------------------|-----------|
| 7B | 8192 (default) | Room for full benchmark session |
| 13B+ | 6144 | Fewer tokens fit in context at same VRAM |
| RX 6600 XT 8 GB | 8192 | Current default; adjust after sweep |

## Recommendations

### Maintain Current Default

**Keep `FOUNDRY_CONTEXT_TOKENS=8192`** as the default in `harness/manifest.json` until benchmark data indicates otherwise.

Rationale:
- The current default is a reasonable middle ground between context retention and memory pressure
- No benchmark data exists to justify changing the default
- The `FOUNDRY_TOKEN_BUDGET` provides a hard abort cap as a safety net

### Future Sweep Requirements

When the sweep is executed, capture:
1. Per-threshold pass rates (overall and per-task-difficulty)
2. Average `tokens_used` per session at each threshold
3. `context_pruned` event frequency and token reduction per prune
4. `task_aborted(reason='token_budget')` count at each threshold
5. Per-task breakdown to identify which tasks are threshold-sensitive

### If Benchmark Data Supports a Change

If sweep data shows:
- **Pass rate improves significantly** with a higher threshold (e.g., 16384) without excessive token budget hits → recommend increasing default
- **Pass rate is unchanged** at a lower threshold (e.g., 4096) → recommend decreasing default to reduce memory pressure
- **Token budget hit rate increases** at higher thresholds → maintain or decrease default

Any change to the manifest default must be routed through the Critic (ADR-0004) before shipping.

## Consequences

- Until live sweep data is collected, token threshold recommendations are based on general knowledge rather than measured data.
- The `TokenAwarePruningHook` and `resolve_context_tokens_threshold()` are implemented and functional.
- Results should be attached to this ADR or a linked issue when the sweep is executed.
- The current default of 8192 is preserved until benchmark data justifies a change.

## References

- [Issue #465](https://github.com/anchapin/foundry-x/issues/465): TokenAwarePruningHook implementation
- [Issue #553](https://github.com/anchapin/foundry-x/issues/553): Validate and document context pruning at scale behavior
- [ADR-0016 §6](./0016-phase-3-quantization-sweep.md#6-token-budget-abort-classification): Token budget vs. context tokens distinction
- `harness/hooks/context_pruning.py`: TokenAwarePruningHook implementation
- `harness/manifest.json`: Default `context_pruning.token_threshold` value
