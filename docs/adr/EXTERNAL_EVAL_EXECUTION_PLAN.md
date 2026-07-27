# External-Eval Pearson Correlation Study: Execution Plan

> **Issue:** #1028
> **ADR:** [ADR-0032](0032-external-eval-pearson-study-protocol.md)
> **Status:** Pending — LLAMACPP_HOST not available in current environment
> **Date:** 2026-07-27

## Executive Summary

This document describes the execution plan for the ADR-0023 external-eval
Pearson correlation study required by issue #1028. The study validates whether
the internal benchmark suite (`benchmarks/tasks/`) ranks agent configurations
the same way HumanEval+ ranks them.

**Study goal:** Compute Pearson correlation coefficient r between per-configuration
internal pass rates and external HumanEval+ pass rates across ≥30 paired
observations.

## Prerequisites

### 1. Hardware Requirements

- **GPU** with sufficient VRAM for at least one GGUF model at a time
  - Qwen2.5-7B: ~8GB VRAM for Q8_0, ~4GB for Q4_K_M
  - Qwen2.5-14B: ~16GB VRAM for Q8_0, ~8GB for Q4_K_M
- **RAM:** 32GB recommended for host + container
- **Disk:** 50GB for model files + trace store

### 2. Software Requirements

- [ ] `llama.cpp` built with `llama-server` binary available
- [ ] GGUF model files downloaded (see Table 1 below)
- [ ] Git tags created for harness versions (see §Harness Versions below)
- [ ] `foundry-x` dependencies installed: `uv sync`

### 3. Environment Variables

```bash
# Required
export LLAMACPP_HOST="http://127.0.0.1:8080"    # or your llama-server host
export LLAMACPP_DIR="$HOME/llama.cpp"              # llama.cpp checkout
export LLAMACPP_SERVER_BIN="$LLAMACPP_DIR/build/bin/llama-server"
export LLAMACPP_NGL="99"                          # GPU layers (99 = all)

# Optional
export FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS="30"       # default; do not change
export FOUNDRY_EXTERNAL_EVAL_STATE_FILE="logs/.run_external_eval_state.json"
```

## Configuration Matrix

The study uses **36 configurations** (3 harness versions × 6 quantizations × 2 model sizes):

**Table 1: Required GGUF Model Files**

| Model Size | Quantization | Expected Filename | VRAM (approx) |
|------------|-------------|-------------------|---------------|
| Qwen2.5-7B | Q4_K_M | Qwen2.5-7B-Q4_K_M.gguf | ~4GB |
| Qwen2.5-7B | Q5_K_M | Qwen2.5-7B-Q5_K_M.gguf | ~5GB |
| Qwen2.5-7B | Q6_K_M | Qwen2.5-7B-Q6_K_M.gguf | ~6GB |
| Qwen2.5-7B | Q8_0 | Qwen2.5-7B-Q8_0.gguf | ~8GB |
| Qwen2.5-7B | IQ4_XS | Qwen2.5-7B-IQ4_XS.gguf | ~4GB |
| Qwen2.5-7B | IQ4_NL | Qwen2.5-7B-IQ4_NL.gguf | ~4GB |
| Qwen2.5-14B | Q4_K_M | Qwen2.5-14B-Q4_K_M.gguf | ~8GB |
| Qwen2.5-14B | Q5_K_M | Qwen2.5-14B-Q5_K_M.gguf | ~10GB |
| Qwen2.5-14B | Q6_K_M | Qwen2.5-14B-Q6_K_M.gguf | ~12GB |
| Qwen2.5-14B | Q8_0 | Qwen2.5-14B-Q8_0.gguf | ~16GB |
| Qwen2.5-14B | IQ4_XS | Qwen2.5-14B-IQ4_XS.gguf | ~8GB |
| Qwen2.5-14B | IQ4_NL | Qwen2.5-14B-IQ4_NL.gguf | ~8GB |

Model files should be placed in `/srv/models/` or another location accessible
to `llama-server`. Update `--model` paths in the configs file or use symlinks.

## Harness Versions

The study uses three harness versions. These must be git tags in the repository:

```bash
# Create harness version tags (operator must do this before running)
git tag harness-v1.0 <commit-hash-for-baseline>
git tag harness-v1.1 <commit-hash-for-current>
git tag harness-v1.2 <commit-hash-for-latest-evolved>

# Verify tags exist
git tag -l "harness-v1.*"
```

If a tag is absent for a configuration, the `run_external_eval.sh` script
will skip that configuration and print a warning. The study proceeds with
remaining configurations provided at least 30 paired observations remain.

## Execution Steps

### Step 1: Pre-flight Validation

Before running any model evaluations, validate the infrastructure:

```bash
cd /home/alex/AI/foundry-x

# 1a. Validate slice integrity (no model tokens spent)
uv run --quiet python -c "
from foundry_x.evaluation.humaneval_plus import load_humaneval_slice, slice_pass_rates
tasks = load_humaneval_slice('benchmarks/external/humaneval_plus_sample.jsonl')
passed, total = slice_pass_rates(tasks)
assert passed == total, f'slice integrity: {passed}/{total}'
print(f'slice ok: {total} tasks, all canonical solutions pass')
"

# 1b. Verify git tags for harness versions
for tag in harness-v1.0 harness-v1.1 harness-v1.2; do
    git rev-parse "$tag" > /dev/null 2>&1 && echo "$tag: ok" || echo "$tag: MISSING"
done

# 1c. Verify llama-server binary exists
ls -la "$LLAMACPP_SERVER_BIN" || echo "llama-server not found at $LLAMACPP_SERVER_BIN"

# 1d. Verify model files exist
for model in \
    /srv/models/Qwen2.5-7B-Q4_K_M.gguf \
    /srv/models/Qwen2.5-14B-Q8_0.gguf; do
    ls -la "$model" || echo "Model not found: $model"
done
```

### Step 2: Run the Study

The study is executed in **batches** to prevent progress loss from interruptions:

```bash
cd /home/alex/AI/foundry-x

# Run with the reference model (Qwen2.5-14B-Q8_0) as the llama-server model
# The script will run each config's internal + external legs

# Full run (all 36 configs in one invocation)
infra/scripts/run_external_eval.sh \
    --model /srv/models/Qwen2.5-14B-Q8_0.gguf \
    --configs configs/external_eval_configs_36.txt \
    --output logs/external_eval_36configs_$(date -u +%Y%m%dT%H%M%SZ)_report.json

# OR incremental batched run (recommended for long studies)
# Batch size of 6 configs per invocation allows checkpointing
infra/scripts/run_external_eval.sh \
    --model /srv/models/Qwen2.5-14B-Q8_0.gguf \
    --configs configs/external_eval_configs_36.txt \
    --batch-size 6 \
    --output logs/external_eval_36configs_$(date -u +%Y%m%dT%H%M%SZ)_report.json
# Re-run with --resume to continue after each batch
```

### Step 3: Interpret Results

After completion, check the verdict in the JSON report:

```bash
# Read the verdict
cat logs/external_eval_*_report.json | jq '.verdict, .pearson, .pearson_ci_95'
```

**Interpretation (per ADR-0023 §Thresholds):**

| Band | Range | Meaning |
|------|-------|---------|
| `valid_proxy` | r ≥ 0.7 | Internal suite is a defensible proxy for external ranking. |
| `weak_proxy` | 0.3 ≤ r < 0.7 | Internal suite partially ranks like HumanEval+. |
| `invalid_proxy` | r < 0.3 | Internal suite does not reproduce external ranking. |

### Step 4: Archive Artifacts

After successful completion, archive study artifacts:

```bash
STUDY_ID="external_eval_36configs_YYYYMMDDTHHMMSSZ"  # from report

# Archive required artifacts
mkdir -p logs/studies/$STUDY_ID
cp logs/${STUDY_ID}_report.json logs/studies/$STUDY_ID/
cp logs/${STUDY_ID}.jsonl logs/studies/$STUDY_ID}/  # if created
cp logs/${STUDY_ID}_external_results.jsonl logs/studies/$STUDY_ID}/  # if created
cp logs/.run_external_eval_state.json logs/studies/$STUDY_ID}/  # checkpoint
cp configs/external_eval_configs_36.txt logs/studies/$STUDY_ID}/

# Commit artifacts (within 48 hours of completion)
git add logs/studies/$STUDY_ID/
git commit -m "docs(study): archive $STUDY_ID Pearson correlation study artifacts"
```

## Cost Estimate

| Item | Estimate |
|------|----------|
| Total configurations | 36 |
| Runs per configuration | 2 (internal + external) |
| Total fx-runner invocations | 72 |
| Estimated time per run | ~5-15 minutes (model + task complexity) |
| Total estimated time | 6-18 hours |
| Model tokens | Varies by model size and task |

## Troubleshooting

### llama-server not starting

```bash
# Check if port 8080 is already in use
lsof -i :8080

# Try manually starting llama-server
$LLAMACPP_SERVER_BIN \
    --model /srv/models/Qwen2.5-14B-Q8_0.gguf \
    --host 127.0.0.1 --port 8080 \
    --n-gpu-layers 99
```

### Study interrupted mid-run

```bash
# Resume from checkpoint (safe default - run same command with --resume)
infra/scripts/run_external_eval.sh \
    --model /srv/models/Qwen2.5-14B-Q8_0.gguf \
    --configs configs/external_eval_configs_36.txt \
    --resume

# Checkpoint state file
cat logs/.run_external_eval_state.json | jq '.configs_completed | length'
```

### Exit code 3: Under-powered study

Fewer than 30 configurations completed. Re-run with more configs or check
that configurations are being recorded correctly.

### Exit code 4: Zero variance

All internal or all external pass rates are identical. The task set is not
discriminating enough. Consider using a larger/different HumanEval+ slice.

### Exit code 5: Run failure

Check stderr output for which configuration failed. The checkpoint file
preserves completed configurations - fix the issue and re-run with `--resume`.

## Files Reference

| File | Purpose |
|------|---------|
| `infra/scripts/run_external_eval.sh` | Study orchestrator |
| `configs/external_eval_configs_36.txt` | 36 configuration definitions |
| `benchmarks/external/humaneval_plus_sample.jsonl` | 20-task HumanEval+ slice |
| `src/foundry_x/evaluation/correlation.py` | Pearson math + guards |
| `src/foundry_x/evaluation/humaneval_plus.py` | HumanEval+ loader/scorer |
| `src/foundry_x/evaluation/aggregator.py` | Per-config aggregation |
| `src/foundry_x/evaluation/study_state.py` | Incremental checkpoint state |
| `docs/adr/0023-external-eval-validation-study.md` | Machinery documentation |
| `docs/adr/0032-external-eval-pearson-study-protocol.md` | Study protocol |

## Next Steps After Study Completion

1. Update ADR-0023 Status with Pearson number, CI, and verdict
2. If `weak_proxy` or `invalid_proxy`: file follow-up issue per ADR-0023 criterion 4
3. Consider proxy-broadening ladder per ADR-0027 (Level 1: 8 implementation tasks)
