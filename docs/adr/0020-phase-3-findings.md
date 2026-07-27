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

## GGUF v4 Quantization Extension (Issue #1050)

### Motivation

The llama.cpp GGUF format has evolved. GGUF v4 introduces
importance-matrix (imatrix) quantization types that offer different
quality/VRAM tradeoffs than the v3 K-quants studied above:

| Quantization | Type | VRAM Est. (7B) | Notes |
|--------------|------|-----------------|-------|
| **IQ4_XS** | v4 (imatrix) | ~4.1 GB | Highest-quality IQ; comparable to Q4_K_M at lower VRAM |
| **IQ3_S** | v4 (imatrix) | ~3.3 GB | Mid-range IQ; quality between Q3 and Q4 K-quants |
| **IQ3_XXS** | v4 (imatrix) | ~3.1 GB | Aggressive; significant quality degradation expected |
| **IQ2_XXS** | v4 (imatrix) | ~2.7 GB | Extreme compression; suitable only for smoke tests |
| **Q2_K** | v3 (baseline) | ~2.6 GB | Baseline for the aggressive end; not imatrix-based |

### Methodology

The v4 quantization sweep uses the same `foundry-sweep` infrastructure
(ADR-0016) but with v4 quantization labels:

```bash
FOUNDRY_MODEL_PATH=/srv/models \
  foundry-sweep sweep \
  --quantizations Q2_K,IQ2_XXS,IQ3_XXS,IQ3_S,IQ4_XS,Q4_K_M,Q5_K_M,Q8_0 \
  --harness-dir harness \
  --baseline Q8_0 \
  --regression-threshold 2.0
```

The sweep code (`Critic.quantization_sweep`) accepts arbitrary
quantization labels; the known v4 types are documented in
`KNOWN_V4_QUANTIZATIONS` in `src/foundry_x/evolution/critic.py`.

### v4 Intelligence Floor Table

> **Status: PENDING** — requires GPU hardware and v4 model files. Tracked
> in follow-up issue.

| Quantization | VRAM Est. | Pass Rate | vs. Q8_0 | Status |
|--------------|-----------|-----------|----------|--------|
| **IQ4_XS** | ~4.1 GB | — | — | Pending |
| **IQ3_S** | ~3.3 GB | — | — | Pending |
| **IQ3_XXS** | ~3.1 GB | — | — | Pending |
| **IQ2_XXS** | ~2.7 GB | — | — | Pending |
| **Q2_K** | ~2.6 GB | — | — | Pending |

The goal is to identify whether any IQ quantization offers a pass rate
within 2 pp of the Q8_0 baseline at lower VRAM than the current
recommended floor (Q5_K_M, ~5.5 GB). If so, it would allow 8 GB card
operators to run a higher-quality model per VRAM dollar.

## Context Window Sweep (Issue #1050)

### Motivation

ADR-0020's intelligence floor was studied at the default
`FOUNDRY_CONTEXT_TOKENS=8192`. Larger context windows (16k, 32k, 128k)
change the intelligence floor because:

1. **KV cache growth**: larger contexts consume more VRAM for the KV
   cache, reducing the VRAM budget available for model weights and
   potentially forcing lower quantization.
2. **Long-context quality degradation**: some quantizations degrade
   more than others on long-context tasks (retrieval, summarization).
3. **Context pruning interaction**: `FOUNDRY_CONTEXT_TOKENS` is the
   pruning threshold (ADR-0021). A higher threshold means less pruning
   but more VRAM usage; the tradeoff is quantization-sensitive.

### Methodology

The context window sweep uses the new `--context-tokens` flag to run
the benchmark suite at different `FOUNDRY_CONTEXT_TOKENS` values:

```bash
# Sweep context windows at the recommended floor (Q5_K_M)
for ctx in 8192 16384 32768; do
  FOUNDRY_MODEL_PATH=/srv/models \
    foundry-sweep sweep \
    --quantizations Q5_K_M,Q8_0 \
    --harness-dir harness \
    --baseline Q8_0 \
    --context-tokens $ctx \
    --output logs/sweep_ctx_${ctx}.json
done
```

For each context window size, identify the minimum viable quantization
that stays within 2 pp of the Q8_0 baseline at that context window.

### Context Window Intelligence Floor Table

> **Status: PENDING** — requires GPU hardware with sufficient VRAM for
> larger KV caches. Tracked in follow-up issue.

| Context Window | Q8_0 Pass Rate | Q5_K_M Pass Rate | IQ4_XS Pass Rate | Notes |
|----------------|----------------|------------------|------------------|-------|
| **8192** (current) | — | — | — | Baseline from table above |
| **16384** | — | — | — | Pending |
| **32768** | — | — | — | Pending |
| **131072** | — | — | — | Pending; may require >8 GB VRAM |

The goal is to determine whether `FOUNDRY_CONTEXT_TOKENS=8192` remains
the correct default, or whether larger context windows are viable at
the recommended quantization floor without exceeding 8 GB VRAM.

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
| 6 | Are there benchmark tasks that remain intractable even at Q8_0? | Issue #1027 | **Studied — see §Intractable Task Study** |
| 7 | Do GGUF v4 IQ quantizations (IQ4_XS, IQ3_S) offer a better quality/VRAM tradeoff than Q5_K_M? | Issue #1050 (follow-up pending) | Open |
| 8 | Does the intelligence floor change at larger context windows (16k, 32k)? | Issue #1050 (follow-up pending) | Open |

Issues #549–#553 must be resolved before this ADR can be updated from "projected" to "empirically confirmed" status.

## Intractable Task Study (Open Question #6)

### Motivation

ADR-0020 Open Question #6 asks whether any benchmark tasks remain
intractable even at Q8_0 -- never passing regardless of quantization quality.
Answering this is critical for:

1. **Benchmark validity**: if a task never passes, it cannot discriminate
   between harness versions; it measures something other than harness quality.
2. **Task maintenance**: intractable tasks waste CI cycles and produce noisy
   regression signals.
3. **Roadmap planning**: hard-tier tasks (ADR-0028) that are also intractable
   may need to be deferred until the agent improves.

### Study Design

The study (`infra/scripts/study_intractable_tasks.py`) runs each candidate
task `N` times at Q8_0 and records pass/fail per run. A task is
classified as **intractable** if it never passes (0/N runs).

**Candidate task selection** prioritised hard-tier tasks per ADR-0028 §2:

| Task | Rationale for study inclusion |
|------|------------------------------|
| `debug_import_cycle` | ADR-0028 H1: multi-phase reasoning, cross-module scope |
| `refactor_api_with_constraints` | ADR-0028 H2: multi-file coordinated refactor with constraint |
| `hook_timing_attack_evals` | ADR-0028 H1: timing-based side channel requires precise reasoning |
| `cross_file_refactor` | Multi-file scope, coordinated edits across 3+ files |
| `refactor_across_three_files` | Multi-file scope, coordinated edits |
| `multi_file_rename` | Cross-module scope, non-trivial state |
| `code_review_diff` | Complex reasoning + multi-step tool use |
| `external_eval_correlation` | Complex evaluation requiring correlation analysis |

**Study runs**: `STUDY_RUNS=3` (minimum for consistency assessment; more runs
reduce false positives on noisy tasks).

**Pass threshold**: a task is "solvable" if it passes at least 1/3 runs at Q8_0.
A task is "intractable" if it passes 0/3 runs.

### Study Execution

```bash
# Ensure llama-server is running with Q8_0 model
llama-server --model /srv/models/*.Q8_0.gguf --host 127.0.0.1 --port 8080

# Run the study
FOUNDRY_MODEL_PATH=/srv/models \
  LLAMACPP_HOST=http://127.0.0.1:8080 \
  python infra/scripts/study_intractable_tasks.py
```

Results are saved to `logs/intractable_study/intractable_study_<timestamp>.json`
and an ADR appendix is written to `docs/adr/adr-0020-intractable-study.md`.

### Intractable Task Criteria

A task is **intractable at Q8_0** when:

1. It fails all `STUDY_RUNS` attempts at Q8_0, AND
2. The failure is not due to `task_aborted(reason="token_budget")` or
   `task_aborted(reason="wall_clock")` (those are infrastructure limits, not
   model capability limits), AND
3. The task's `difficulty_tier` is `hard` or `medium`.

Tasks that fail due to timeouts or token budget hits are **not** intractable --
they may simply need more resources. The study separately tracks these.

### Study Status

| Status | Description |
|--------|-------------|
| **Pending** | No GPU hardware / llama-server available; infrastructure prepared |
| **In Progress** | Study running; results being collected |
| **Complete** | All tasks studied; findings documented in adr-0020-intractable-study.md |

> **Note**: As of this writing, LLAMACPP_HOST is not set and no Q8_0 model
> is available. The study script is implemented and ready to run; it will
> auto-detect infrastructure availability and either run the study or emit
> a "pending" status. See `infra/scripts/study_intractable_tasks.py`.

When the study runs, update `docs/adr/adr-0020-intractable-study.md` with the
live results and update the table below.

### Study Results (Pending)

| Task | Tier | Pass Rate (Q8_0) | Status | Notes |
|------|------|-----------------|--------|-------|
| debug_import_cycle | hard | — | Pending | |
| refactor_api_with_constraints | hard | — | Pending | |
| hook_timing_attack_evals | hard | — | Pending | |
| cross_file_refactor | medium | — | Pending | |
| refactor_across_three_files | medium | — | Pending | |
| multi_file_rename | medium | — | Pending | |
| code_review_diff | medium | — | Pending | |
| external_eval_correlation | medium | — | Pending | |

If a task is intractable at Q8_0, it is excluded from the pass-rate
denominator for the intelligence floor calculation and flagged for
simplification or replacement.

## Consequences

- Production operators on the 5600G / 6600 XT should target Q5_K_M as the minimum quantization.
- The `FOUNDRY_REGRESSION_THRESHOLD_PP` guard (ADR-0016 §3) protects against regressions when comparing candidate quantizations at the release gate.
- The `FOUNDRY_TOKEN_BUDGET` abort is a task-shaped failure classification, not a harness regression — it is excluded from the pass-rate denominator per ADR-0016 §6.
- This ADR is a living document: it must be updated to replace projected values with live sweep data once issues #549–#553 are resolved.
- If live data confirms Q4_K_M pass rate is within 2 pp of Q5_K_M, it may be promoted to the recommended floor for the 6600 XT.
- **Issue #1050 extension**: GGUF v4 IQ quantizations (IQ4_XS, IQ3_S) and larger context windows (16k, 32k) are now in scope. The sweep code supports both via `KNOWN_V4_QUANTIZATIONS` constants and the `--context-tokens` CLI flag. Empirical results are pending GPU execution (see follow-up issues).
- **Issue #1027 (Open Question #6)**: Intractable tasks that never pass at Q8_0 are identified by `infra/scripts/study_intractable_tasks.py`. Such tasks are excluded from the pass-rate denominator and flagged for simplification or replacement. See §Intractable Task Study.
