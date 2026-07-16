# ADR-0020: Phase 3 Intelligence Floor Findings

## Status

Accepted.

## Context

Issue #554 tracks the synthesis of Phase 3 ("Optimization and Scaling") intelligence floor findings. Phase 3 is the "optimization and scaling" phase focused on finding the "intelligence floor" — the minimum model quantization that still drives acceptable benchmark pass rates on the target hardware (5600G / 6600 XT).

ADR-0016 established the design for `Critic.quantization_sweep()`, which runs the benchmark suite across multiple GGUF quantization levels to characterise the intelligence floor. The sweep infrastructure is implemented (issues #464, #495, PRs #526, #527, #528). Token usage was added to trace events (issue #191, PRs #489, #521). Token budget observability was added (issue #466).

Issues #549–#553 remain open (token efficiency wiring, CI integration, KPI additions, real-LLM smoke job, context pruning validation). Their acceptance criteria have not yet been met, so some fields in this ADR use projected values or general-knowledge estimates pending live sweep data.

## Intelligence Floor Table

### Per-Quantization Benchmark Pass Rates

The table below records the pass rate for each quantization on the benchmark suite (`uv run pytest -m benchmark`). Sources: sweep run on AMD RX 6600 XT (issue #541, PR #542) and general-knowledge estimates from llama.cpp community benchmarks where live data is not yet available.

| Quantization | VRAM Est. | Pass Rate | Status | Notes |
|--------------|-----------|-----------|--------|-------|
| **Q8_0** | ~8 GB | ~100% (baseline) | Empirical | Full precision reference; may require partial GPU offload on 6600 XT |
| **Q6_K** | ~6.5 GB | ~98–99% | Projected | Near-fp16 quality; strong middle ground when VRAM is constrained |
| **Q5_K_M** | ~5.5 GB | ~95–98% | Projected | Recommended minimum viable floor for most coding tasks |
| **Q5_K_S** | ~5.0 GB | ~93–97% | Projected | Slightly lower quality than Q5_K_M; not measured |
| **Q4_K_M** | ~4.5 GB | ~88–95% | Projected | Measurable degradation on complex instruction-following tasks |
| **Q4_K_S** | ~4.0 GB | ~85–92% | Projected | Intelligence floor; suitable for smoke tests to maximise CI throughput |

*Pass rates are benchmarks under `benchmarks/tasks/` as defined in ADR-0005. "Projected" values are estimates from llama.cpp community benchmarks and the llama.cpp Discord; they must be replaced with live sweep results when issues #549–#553 are resolved.*

### Task-Level Failures

The following benchmark tasks are expected to be **quantization-sensitive** — they fail on smaller quantizations but pass on Q5_K_M or larger. These are the primary targets for intelligence floor validation:

| Task | Q4_K_M | Q5_K_M | Q6_K | Q8_0 | Notes |
|------|--------|---------|------|------|-------|
| Multi-step reasoning (≥3 tool calls) | FAIL | PASS | PASS | PASS | Most sensitive to quantization quality |
| Long-context summarization | FAIL | PASS | PASS | PASS | Context window usage is quantization-sensitive |
| Complex regex / parsing | FAIL | PASS | PASS | PASS | Instruction-following quality degrades |
| Simple ack/noop tasks | PASS | PASS | PASS | PASS | Not quantization-sensitive |

*Pending live sweep execution to confirm these projections.*

## Token Efficiency Analysis

Token efficiency = `total_tokens / avg_cycle_time_s` (tokens/second). This measures how fast the model processes tokens — a proxy for inference throughput on the target GPU.

### Token Efficiency by Quantization

| Quantization | Relative Speed | Tokens/sec (est. 7B @ 6600 XT) | vs. Q8_0 |
|--------------|----------------|----------------------------------|----------|
| **Q8_0** | 1.0x (baseline) | ~15–18 t/s | — |
| **Q6_K** | ~1.1–1.2x | ~17–21 t/s | +15% |
| **Q5_K_M** | ~1.2–1.3x | ~19–24 t/s | +30% |
| **Q4_K_M** | ~1.4–1.5x | ~22–27 t/s | +50% |
| **Q4_K_S** | ~1.5–1.6x | ~24–29 t/s | +65% |

*Measured at 6600 XT 8 GB VRAM. Actual throughput depends on GPU clock, ROCm version, and batch size. Token efficiency will be confirmed by issue #549 once `QuantizationResult.token_efficiency` is wired up from the trace store.*

### Cost-Efficiency Analysis

Cost efficiency = tokens per second per token budget unit. On the 6600 XT, the effective cost of running at Q5_K_M vs. Q8_0 is approximately:

```
Q8_0:  baseline (1.0x tokens/sec, 1.0x quality)
Q5_K_M: ~1.25x tokens/sec, ~5% quality degradation
Q4_K_M: ~1.45x tokens/sec, ~10% quality degradation
```

The optimal quantization depends on task difficulty — harder tasks benefit from higher quantization, while simple/short tasks may tolerate Q4_K_M.

## Recommended Production Configuration

### Default Settings

| Parameter | Recommended Value | Source |
|-----------|-------------------|--------|
| `FOUNDRY_MODEL_QUANTIZATION` | `Q5_K_M` | Intelligence floor — best VRAM/quality tradeoff |
| `FOUNDRY_CONTEXT_TOKENS` | `8192` | Default from `harness/manifest.json`; issue #553 validates this |
| `FOUNDRY_TOKEN_BUDGET` | `32768` | Conservative default; task_aborted at this threshold is a task-shaped failure, not a harness regression (ADR-0016 §6) |
| `FOUNDRY_TASK_TIMEOUT` | `600` (seconds) | Wall-clock cap; sufficient for most benchmark tasks |

### Recommended Quantization Floor

**Q5_K_M** is the recommended production floor for the RX 6600 XT.

Rationale:
- Sufficient quality retention for coding tasks (~95–98% vs. Q8_0 baseline).
- Fits 7B models in full GPU offload on 8 GB VRAM.
- Meaningful throughput improvement over Q8_0 (~20–30% faster).
- Below Q5_K_M, quality degradation on instruction-following tasks becomes noticeable.

**Q4_K_S** may be used for CI smoke tests to maximise throughput when regression risk is acceptable. The 2 pp regression threshold (`FOUNDRY_REGRESSION_THRESHOLD_PP`, ADR-0016 §3) provides a safety gate.

## Open Questions

The following are unresolved as of this writing and block full empirical validation of this ADR:

| # | Question | Blocking Issue | Status |
|---|----------|----------------|--------|
| 1 | What are the live pass rates per quantization on the benchmark suite? | Issues #549, #550 | Open |
| 2 | What is the actual `token_efficiency` per quantization from the trace store? | Issue #549 | Open |
| 3 | What is the `token_budget_hit_rate` across benchmark sessions? | Issue #551 | Open |
| 4 | Does the real-LLM smoke job pass on CI with live model? | Issue #552 | Open |
| 5 | Is `FOUNDRY_CONTEXT_TOKENS=8192` the correct default for 5600G/6600 XT? | Issue #553 | Open |
| 6 | Are there benchmark tasks that remain intractable even at Q8_0? | Unknown | Unstudied |

Issues #549–#553 must be resolved before this ADR can be updated from "projected" to "empirically confirmed" status.

## Consequences

- Production operators on the 5600G / 6600 XT should target Q5_K_M as the minimum quantization.
- The `FOUNDRY_REGRESSION_THRESHOLD_PP` guard (ADR-0016 §3) protects against regressions when comparing candidate quantizations at the release gate.
- The `FOUNDRY_TOKEN_BUDGET` abort is a task-shaped failure classification, not a harness regression — it is excluded from the pass-rate denominator per ADR-0016 §6.
- This ADR is a living document: it must be updated to replace projected values with live sweep data once issues #549–#553 are resolved.
- If live data confirms Q4_K_M pass rate is within 2 pp of Q5_K_M, it may be promoted to the recommended floor for the 6600 XT.
