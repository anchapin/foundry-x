# ADR-0019: Quantization Intelligence Floor Findings

## Status

Proposed.

## Context

Issue #530 tracks the documentation of findings from the Phase 3 quantization sweep. ADR-0016 established the design for `Critic.quantization_sweep()`, which runs the benchmark suite across Q4, Q5, and Q8 quantizations to find the "intelligence floor" — the minimum quantization level that preserves benchmark performance.

The sweep infrastructure was implemented under issue #495. This ADR captures what the sweep was designed to produce, the current status of results, and general-knowledge recommendations until live data is collected.

## Background: What the Sweep Was Designed to Do

Per ADR-0016, the sweep:

- Runs the full benchmark suite (`uv run pytest -m benchmark`) against multiple quantization levels.
- Produces a `QuantizationVerdict` with per-quantization `QuantizationResult` containing pass rates, task-level breakdowns, cycle times, and token efficiency.
- Stores traces in the same `logs/` SQLite database with quantization-stamped `FOUNDRY_MODEL_ID`.
- Compares each candidate against the current production quantization as baseline, flagging regression if pass rate drops more than 2 pp (`FOUNDRY_REGRESSION_THRESHOLD_PP`).
- Task-shaped failures (token budget aborts) are excluded from pass-rate denominators.

The `QuantizationResult` model (per `benchmarks/models.py` and `src/foundry_x/evolution/critic.py`) captures:

```python
class QuantizationResult(BaseModel):
    quantization: str              # e.g. "Q4_K_M", "Q5_K_M", "Q8_K_M"
    model_path: str
    model_id: str
    total_tasks: int
    passed_tasks: int
    failed_tasks: int
    task_shaped_failures: int
    pass_rate: float
    avg_cycle_time_s: float | None
    total_tokens: int
    token_efficiency: float | None  # total_tokens / avg_cycle_time_s
```

## Current Status

The sweep infrastructure is implemented and tests exist (`tests/evolution/test_quantization_sweep.py`). However, **the sweep has not been executed against a live model with results persisted to `logs/quantization_results.json`** as of this writing.

Contributors running the sweep should:

1. Ensure GGUF model files exist for each target quantization (Q4_K_M, Q5_K_M, Q8_K_M).
2. Run `foundry-sweep quantize --output logs/quantization_results.json` (or equivalent CLI entry point).
3. Attach the output JSON to the relevant issue or PR.

## General Knowledge: Quantization Level Tradeoffs

Until live sweep data is available, the following represents the generally understood tradeoff curve for GGUF quantizations on AMD RX 6600 XT (8 GB VRAM):

### Quantization Levels

| Level | Description | VRAM Footprint | Quality Retention | Relative Speed |
|-------|-------------|-----------------|-------------------|----------------|
| **Q8_K_M** | 8-bit integer, medium matrix quantization | ~7-8 GB for 7B models | ~98-99% quality | Baseline (1.0x) |
| **Q6_K_M** | 6-bit integer, medium matrix quantization | ~5.5-6.5 GB | ~95-97% quality | ~1.1-1.2x faster |
| **Q5_K_M** | 5-bit integer, medium matrix quantization | ~4.5-5.5 GB | ~92-95% quality | ~1.2-1.3x faster |
| **Q4_K_M** | 4-bit integer, medium matrix quantization | ~3.5-4.5 GB | ~88-92% quality | ~1.4-1.5x faster |
| **Q4_K_S** | 4-bit integer, small matrix quantization | ~3.0-4.0 GB | ~85-89% quality | ~1.5-1.6x faster |

*Figures are approximate; actual quality retention varies by model size, architecture, and quantization method.*

### Key Observations

1. **Q5_K_M is the generally recommended minimum viable quantization** for most coding tasks. It offers a good balance between VRAM efficiency and quality retention. The 6600 XT can run full 7B models at Q5 with GPU offloading.

2. **Q4_K_M may show measurable degradation** on tasks requiring precise instruction following or multi-step reasoning. The intelligence floor for complex benchmarks may be higher than for simple tasks.

3. **Q6_K_M is a strong middle ground** when VRAM is constrained but Q5 quality is desired. It is rarely offered in pre-built quantizations but can be generated with llama.cpp.

4. **VRAM constraints on 6600 XT**: Full model offload is possible for Q5 and below. Q6 and Q8 require partial offload or may exceed 8 GB for larger models.

## Performance vs. Quantization Level (Projected)

Without live sweep data, the following is a projected table based on general llama.cpp benchmarks and community reports for coding tasks:

| Quantization | Pass Rate (Projected) | Relative Cost Efficiency |
|--------------|---------------------|------------------------|
| Q8_K_M       | 100% (baseline)     | 1.0x (tokens/sec)     |
| Q6_K_M       | 98-99%             | 1.1-1.2x              |
| Q5_K_M       | 95-98%             | 1.2-1.3x              |
| Q4_K_M       | 88-95%             | 1.4-1.5x              |

*These are estimates. Actual numbers must be replaced with sweep results.*

## Recommendations

### Minimum Viable Quantization

**Q5_K_M** is recommended as the minimum viable quantization for production use on the RX 6600 XT.

Rationale:
- Sufficient quality retention for coding tasks (~92-95% vs Q8).
- Fits 7B models in full offload on 8 GB VRAM.
- Meaningful cost/speed improvement over Q8 (~20-30% faster).
- Below Q5, quality degradation on instruction-following tasks becomes noticeable.

### Cost-Efficiency Analysis (Conceptual)

Cost efficiency = effective tokens processed per second = `token_efficiency` field in `QuantizationResult`.

```
Q8_K_M: token_efficiency = 1.0x (baseline)
Q5_K_M: token_efficiency ≈ 1.25x (20-30% more tokens/sec, ~5% quality cost)
Q4_K_M: token_efficiency ≈ 1.45x (45% more tokens/sec, ~10% quality cost)
```

The optimal point depends on:
- Task difficulty (harder tasks may need Q5 minimum)
- VRAM pressure (if running other workloads concurrently)
- Throughput requirements (batch processing favors lower quant)

### Future Sweep Data Requirements

When the sweep is executed, the following data points must be captured in `logs/quantization_results.json`:

1. Per-quantization pass rates (overall and per-difficulty-tier).
2. Per-task pass/fail breakdown to identify which tasks are quantization-sensitive.
3. `avg_cycle_time_s` and `total_tokens` for cost-efficiency calculation.
4. `task_shaped_failures` count to separate task-size issues from quantization issues.
5. VRAM utilization during each run to confirm 6600 XT compatibility.

## Consequences

- Until live sweep data is collected, quantization recommendations are based on general knowledge rather than measured intelligence floor.
- The `QuantizationResult` model and `Critic.quantization_sweep()` method are in place; running the sweep is a matter of execution.
- Results should be attached to this ADR or a linked issue when available.
- If Q4_K_M shows <5% regression on benchmark pass rate, it may be recommended as the minimum for smoke tests to maximize CI throughput.
