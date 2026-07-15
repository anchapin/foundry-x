# Phase 3 Quantization Sweep Findings

> **Status**: Quantization sweep is pending RX 6600 XT hardware access.
> All framework infrastructure (hook, CLI, ADR) is implemented; real sweep
> data has not yet been recorded.

## Hardware Profile

| Property | Value |
|----------|-------|
| GPU | AMD RX 6600 XT |
| VRAM | 8 GB |
| OS | Linux (target) |
| ROCm Version | 6.x (target) |

## 1. Quantization Intelligence Floor

### Sweep Command

```bash
FOUNDRY_MODEL_PATH=/srv/models \
  foundry-sweep \
  --quantizations Q4_K_S,Q5_K_M,Q6_K,Q8_0 \
  --harness-dir ./harness \
  --output ./logs/sweep-YYYYMMDD.json
```

### Expected Output Format

```json
{
  "sweep_id": "...",
  "timestamp": "...",
  "hardware": { "gpu": "AMD RX 6600 XT", "vram_gb": 8 },
  "quantizations": [
    {
      "name": "Q4_K_S",
      "model_path": "/srv/models/Q4_K_S/",
      "result": {
        "passed": 18, "failed": 2, "pass_rate": 0.90,
        "task_results": [
          { "task": "bash_skill_real_exec", "status": "PASSED" },
          { "task": "grep_search_fix", "status": "PASSED" }
        ]
      }
    }
  ],
  "recommended_floor": "Q4_K_S"
}
```

The sweep script is at `logs/run-sweep-rx6600xt.sh`.

### Per-Quantization Pass Rates (Pending Sweep)

| Quantization | Pass Rate | VRAM Est. | Quality Retention | Relative Speed |
|--------------|-----------|-----------|-------------------|----------------|
| Q4_K_S | TBD | ~4.5 GB | ~85-89% | ~1.5-1.6x |
| Q5_K_M | TBD | ~5.5 GB | ~92-95% | ~1.2-1.3x |
| Q6_K | TBD | ~6.5 GB | ~95-97% | ~1.1-1.2x |
| Q8_0 | TBD | ~8 GB | ~98-99% | baseline (1.0x) |

*Projected figures based on general llama.cpp benchmarks until live sweep data is available.*

### Recommended Floor

**TBD** — pending sweep execution.

The recommended floor will be the lowest quantization that maintains >85% benchmark
pass rate while fitting in RX 6600 XT VRAM without offload.

Until sweep data exists, **Q5_K_M is the generally recommended minimum viable
quantization** for production use on the RX 6600 XT based on the generally
understood tradeoff curve: sufficient quality retention (~92-95% vs Q8), fits
7B models in full offload on 8 GB VRAM, and ~20-30% faster than Q8.

Below Q5, quality degradation on instruction-following tasks becomes noticeable.

## 2. Context Pruning Effectiveness

### Implementation

Context pruning is implemented by `TokenAwarePruningHook` in
`harness/hooks/context_pruning.py` (issue #465). It is registered via
`harness/manifest.json` under the `hooks` key:

```json
"context_pruning": {
  "token_threshold": 8192
}
```

The hook intercepts every `pre_tool` call and queries the running token total
from `model_response` trace events. When cumulative tokens exceed
`token_threshold`, the hook drops the oldest non-preserved events
(`tool_result` and `user_prompt` are always kept) and emits a
`context_pruned` trace event carrying `dropped`, `threshold_tokens`, and
`session_tokens`.

### Default Threshold

| Parameter | Value | Source |
|-----------|-------|--------|
| `token_threshold` | 8192 | `harness/manifest.json` |
| `DEFAULT_TOKEN_THRESHOLD` | 8192 | `harness/hooks/context_pruning.py:50` |

The `FOUNDRY_CONTEXT_TOKENS` env var can override the manifest value at runtime
via `resolve_context_tokens_threshold()`.

### Token Counts Before/After Pruning (Pending Benchmark Data)

Real before/after token counts from a benchmark sweep have not yet been recorded.
The `context_pruned` trace event shape is:

```
kind="context_pruned"
payload={
  "dropped": <int>,           -- events removed
  "threshold_tokens": 8192,    -- trigger threshold
  "session_tokens": <int>     -- tokens at time of prune
}
```

When sweep data is available, this section will be updated with observed
token counts before and after pruning, and the percentage reduction per run.

### Threshold Recommendations

| Model Size | Recommended `token_threshold` | Rationale |
|------------|------------------------------|-----------|
| 7B | 8192 (default) | Room for full benchmark session |
| 13B+ | 6144 | Fewer tokens fit in context at same VRAM |
| RX 6600 XT 8 GB | 8192 | Current default; adjust after sweep |

## 3. Token Budget Behavior

### Mechanism

`FOUNDRY_TOKEN_BUDGET` (env var) → `RunLimits.token_budget` → enforced in
`Runner.run_task()` after each `model_response`:

```
tokens_used (cumulative) > token_budget
  → task_aborted(reason="token_budget")
  → outcome.status="failed", outcome.reason="token_budget"
```

`FOUNDRY_TASK_TIMEOUT=600` (seconds) enforces a wall-clock cap independently;
it is plumbed through `RunLimits.task_timeout_s` and fires
`task_aborted(reason="wall_clock")`.

Both caps write a `task_aborted` trace event with the active limit value
at abort time, so the trace store records which cap fired and what the
consumed value was.

### Default Values

| Env Var | Default | File |
|---------|---------|-------|
| `FOUNDRY_TASK_TIMEOUT` | 600 | `.env.example:11` |
| `FOUNDRY_TOKEN_BUDGET` | _(unset / no cap)_ | `.env.example:12` |

`FOUNDRY_TOKEN_BUDGET` is intentionally opt-in (empty = no cap) because the
primary runaway guard is `FOUNDRY_TASK_TIMEOUT`. Token budget is a secondary
guard for hardware where wall-clock time does not correlate with token spend
(e.g., slow inference on large batches).

### Sessions Hitting Token Budget (Pending Sweep Data)

The number of benchmark sessions that would have triggered `task_aborted(reason='token_budget')`
has not yet been measured. Once a sweep is run, query:

```sql
SELECT COUNT(*) FROM events
WHERE kind = 'task_aborted'
  AND json_extract(payload, '$.reason') = 'token_budget';
```

### Timeout Appropriateness

`FOUNDRY_TASK_TIMEOUT=600` (10 minutes per task) is calibrated for the RX 6600
XT running Q5_K_M on 7B coding models. Tasks that consistently time out before
hitting the token budget may indicate the model is thrashing (excessive
regeneration or tool-call loops); tasks that hit the token budget before the
wall-clock cap suggest the task itself is token-hungry and may need to be split.

## 4. Operator Recommendations

### Which Quantization to Pin

**Recommended**: Q5_K_M for production.

| Quantization | When to Use |
|--------------|-------------|
| **Q5_K_M** | Default production. Best VRAM/quality tradeoff for 7B on 6600 XT. |
| Q6_K | When quality-sensitive tasks require near-Q8 performance and VRAM allows. |
| Q8_0 | Baseline for benchmark comparisons; prefer Q5 unless pass rate regresses. |
| Q4_K_S | Only for VRAM-constrained multi-model setups; expect some quality loss. |

### Which `FOUNDRY_TOKEN_BUDGET` to Set

Leave unset (empty) unless you observe runaway token accumulation without
wall-clock timeout. If needed, start with `FOUNDRY_TOKEN_BUDGET=8192` and
tune based on observed `task_aborted(reason='token_budget')` rate from
`foundry-kpis` output.

### Which `FOUNDRY_CONTEXT_TOKENS` to Use

The default `FOUNDRY_CONTEXT_TOKENS=8192` (via `manifest.json`) is appropriate
for 7B models on the RX 6600 XT. For 13B+ models on the same hardware, lower
to 6144 to account for reduced context capacity.

## Related Issues and Documents

- [Issue #495](https://github.com/anchapin/foundry-x/issues/495): Run suite across Q4/Q5/Q8 quantizations
- [Issue #541](https://github.com/anchapin/foundry-x/issues/541): Run foundry-sweep on RX 6600 XT
- `docs/adr/0019-quantization-intelligence-floor.md`: Full ADR with quantization tradeoffs
- `docs/adr/0016-phase-3-quantization-sweep.md`: `Critic.quantization_sweep()` design
- `docs/adr/0020-context-pruning-at-scale.md`: Context pruning at scale findings (issue #553)
- `logs/run-sweep-rx6600xt.sh`: Sweep automation script
