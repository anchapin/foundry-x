# ADR-0032: External-eval Pearson correlation study protocol (issue #1028)

## Status

**Accepted. 2026-07-27.** (supersedes "Proposed" from issue #1028 implementation)

**Pending — awaiting execution.** LLAMACPP_HOST not available in current
environment; study configuration and execution plan are prepared.
See `docs/adr/EXTERNAL_EVAL_EXECUTION_PLAN.md` for the full execution guide.

## Context

[ADR-0023](0023-external-eval-validation-study.md) shipped the
machinery for a Pearson correlation study: the
`src/foundry_x/evaluation/correlation.py` math layer, the
`src/foundry_x/evaluation/humaneval_plus.py` HumanEval+ loader and scorer,
the 20-task slice under `benchmarks/external/humaneval_plus_sample.jsonl`,
and the `infra/scripts/run_external_eval.sh` operator script. The
machinery was validated offline in `benchmarks/tasks/test_external_eval_correlation.py`.

ADR-0023 §"Follow-ups" item 2 states:

> Run the study: 30+ configurations, real llama.cpp endpoint,
> produce the Pearson number, and update this ADR's "Status" with the
> result.

Issue #1028 executes that follow-up. It requires an ADR because the
correlation study is a one-time empirical validation whose protocol
must be documented before execution so the result is reproducible and
auditable. This ADR is that documentation.

### Prerequisites satisfied

ADR-0023 follow-up item 1 (runner-side session-metadata tagging) was
completed as issue #1007: every `fx-runner` session now records
`model_id`, `quantization`, and `harness_version` in the
`session_start` event and in the sqlite `sessions` table, enabling the
aggregator to group `critic_verdict` events per configuration without
manual labelling.

## Decision

### Study scope

The study runs **36 configurations** — the cross-product of:

**3 harness versions** × **6 quantizations** = **18 configurations**, run
against **2 model sizes** = **36 total configurations**.

| Factor | Values |
|--------|--------|
| Model sizes | `Qwen2.5-7B`, `Qwen2.5-14B` |
| Quantizations | `Q4_K_M`, `Q5_K_M`, `Q6_K_M`, `Q8_0`, `IQ4_XS`, `IQ4_NL` |
| Harness versions | `v1.0` (baseline), `v1.1` (current), `v1.2` (latest evolved) |

The harness versions map to git tags in the repository. The operator
must verify the tag exists before running:

```bash
git tag -l "harness-v1.*"
```

If a tag is absent the configuration is skipped and a warning is
printed; the study proceeds with the remaining configurations provided
at least 30 paired observations remain.

### Pearson r computation

Pearson r is computed by `foundry_x.evaluation.correlation.pearson_binary`
across the 36 paired scalar observations `(internal_pass_rate_k,
external_pass_rate_k)` for `k = 1 … 36`.

**Confidence interval**: Fisher's z-transformation is applied to obtain a
95% confidence interval for r:

```
z = atanh(r)                          # Fisher's z
SE_z = 1 / sqrt(n - 3)                # standard error of z
z_lower = z - 1.96 * SE_z
z_upper = z + 1.96 * SE_z
r_lower = tanh(z_lower)
r_upper = tanh(z_upper)
```

where `n = 36` and `tanh` is the hyperbolic tangent. The CI is computed
in `foundry_x/evaluation/correlation.py` and included in the JSON report
under the key `pearson_ci_95`.

### Commands to run

#### Pre-flight (no model tokens)

```bash
# 1. Validate the slice integrity
uv run --quiet python -c "
from foundry_x.evaluation.humaneval_plus import load_humaneval_slice, slice_pass_rates
tasks = load_humaneval_slice('benchmarks/external/humaneval_plus_sample.jsonl')
passed, total = slice_pass_rates(tasks)
assert passed == total, f'slice integrity: {passed}/{total}'
print(f'slice ok: {total} tasks, all canonical solutions pass')
"

# 2. Verify git tags exist for the harness versions
for tag in harness-v1.0 harness-v1.1 harness-v1.2; do
    git rev-parse "$tag" > /dev/null 2>&1 && echo "$tag: ok" || echo "$tag: MISSING (skipping)"
done
```

#### Study execution

```bash
# Build the configs file (36 lines: see §Study scope above)
cat > /tmp/eval_configs_36.txt << 'EOF'
q7b-q4km   --model Qwen2.5-7B-Q4_K_M.gguf    --quantization Q4_K_M  --harness-version v1.0
q7b-q5km   --model Qwen2.5-7B-Q5_K_M.gguf    --quantization Q5_K_M  --harness-version v1.0
q7b-q6km   --model Qwen2.5-7B-Q6_K_M.gguf    --quantization Q6_K_M  --harness-version v1.0
q7b-q8     --model Qwen2.5-7B-Q8_0.gguf       --quantization Q8_0    --harness-version v1.0
q7b-iq4xs  --model Qwen2.5-7B-IQ4_XS.gguf    --quantization IQ4_XS  --harness-version v1.0
q7b-iq4nl  --model Qwen2.5-7B-IQ4_NL.gguf    --quantization IQ4_NL  --harness-version v1.0
q7b-q4km   --model Qwen2.5-7B-Q4_K_M.gguf    --quantization Q4_K_M  --harness-version v1.1
q7b-q5km   --model Qwen2.5-7B-Q5_K_M.gguf    --quantization Q5_K_M  --harness-version v1.1
q7b-q6km   --model Qwen2.5-7B-Q6_K_M.gguf    --quantization Q6_K_M  --harness-version v1.1
q7b-q8     --model Qwen2.5-7B-Q8_0.gguf       --quantization Q8_0    --harness-version v1.1
q7b-iq4xs  --model Qwen2.5-7B-IQ4_XS.gguf    --quantization IQ4_XS  --harness-version v1.1
q7b-iq4nl  --model Qwen2.5-7B-IQ4_NL.gguf    --quantization IQ4_NL  --harness-version v1.1
q7b-q4km   --model Qwen2.5-7B-Q4_K_M.gguf    --quantization Q4_K_M  --harness-version v1.2
q7b-q5km   --model Qwen2.5-7B-Q5_K_M.gguf    --quantization Q5_K_M  --harness-version v1.2
q7b-q6km   --model Qwen2.5-7B-Q6_K_M.gguf    --quantization Q6_K_M  --harness-version v1.2
q7b-q8     --model Qwen2.5-7B-Q8_0.gguf       --quantization Q8_0    --harness-version v1.2
q7b-iq4xs  --model Qwen2.5-7B-IQ4_XS.gguf    --quantization IQ4_XS  --harness-version v1.2
q7b-iq4nl  --model Qwen2.5-7B-IQ4_NL.gguf    --quantization IQ4_NL  --harness-version v1.2
q14b-q4km  --model Qwen2.5-14B-Q4_K_M.gguf   --quantization Q4_K_M  --harness-version v1.0
q14b-q5km  --model Qwen2.5-14B-Q5_K_M.gguf   --quantization Q5_K_M  --harness-version v1.0
q14b-q6km  --model Qwen2.5-14B-Q6_K_M.gguf   --quantization Q6_K_M  --harness-version v1.0
q14b-q8    --model Qwen2.5-14B-Q8_0.gguf      --quantization Q8_0    --harness-version v1.0
q14b-iq4xs --model Qwen2.5-14B-IQ4_XS.gguf   --quantization IQ4_XS  --harness-version v1.0
q14b-iq4nl --model Qwen2.5-14B-IQ4_NL.gguf   --quantization IQ4_NL  --harness-version v1.0
q14b-q4km  --model Qwen2.5-14B-Q4_K_M.gguf   --quantization Q4_K_M  --harness-version v1.1
q14b-q5km  --model Qwen2.5-14B-Q5_K_M.gguf   --quantization Q5_K_M  --harness-version v1.1
q14b-q6km  --model Qwen2.5-14B-Q6_K_M.gguf   --quantization Q6_K_M  --harness-version v1.1
q14b-q8    --model Qwen2.5-14B-Q8_0.gguf      --quantization Q8_0    --harness-version v1.1
q14b-iq4xs --model Qwen2.5-14B-IQ4_XS.gguf   --quantization IQ4_XS  --harness-version v1.1
q14b-iq4nl --model Qwen2.5-14B-IQ4_NL.gguf   --quantization IQ4_NL  --harness-version v1.1
q14b-q4km  --model Qwen2.5-14B-Q4_K_M.gguf   --quantization Q4_K_M  --harness-version v1.2
q14b-q5km  --model Qwen2.5-14B-Q5_K_M.gguf   --quantization Q5_K_M  --harness-version v1.2
q14b-q6km  --model Qwen2.5-14B-Q6_K_M.gguf   --quantization Q6_K_M  --harness-version v1.2
q14b-q8    --model Qwen2.5-14B-Q8_0.gguf      --quantization Q8_0    --harness-version v1.2
q14b-iq4xs --model Qwen2.5-14B-IQ4_XS.gguf   --quantization IQ4_XS  --harness-version v1.2
q14b-iq4nl --model Qwen2.5-14B-IQ4_NL.gguf   --quantization IQ4_NL  --harness-version v1.2
EOF

# 3. Run the study (llama-server auto-started if not running)
infra/scripts/run_external_eval.sh \
    --model /srv/models/Qwen2.5-14B-Q8_0.gguf \
    --configs /tmp/eval_configs_36.txt \
    --output logs/external_eval_36configs_$(date -u +%Y%m%dT%H%M%SZ)_report.json
```

The script requires `FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS=30` (the default) or
higher; it exits with code 3 if fewer than 30 configurations are
supplied. Exit code 4 means one of the pass-rate series has zero
variance (Pearson undefined); the operator must choose a more
discriminating task set. Exit code 5 means a run failed non-recoverably.

### Interpretation (per ADR-0023 §Thresholds)

| Band | Range | Meaning |
|------|-------|---------|
| `valid_proxy` | `r ≥ 0.7` | Internal suite is a defensible proxy for the external ranking. No follow-up required. |
| `weak_proxy` | `0.3 ≤ r < 0.7` | Internal suite ranks configurations partially like HumanEval+. File a follow-up to broaden the internal task distribution. |
| `invalid_proxy` | `r < 0.3` | Internal suite does not reproduce the external ranking. File a follow-up issue per issue #900 criterion 4. |

### Output format

The aggregator writes a JSON report with this shape:

```jsonc
{
  "study_id": "external_eval_36configs_20260726T143052Z",
  "adr": "ADR-0032",
  "slice": "benchmarks/external/humaneval_plus_sample.jsonl",
  "slice_task_count": 20,
  "model": "/srv/models/Qwen2.5-14B-Q8_0.gguf",
  "harness_versions": ["v1.0", "v1.1", "v1.2"],
  "quantizations": ["Q4_K_M", "Q5_K_M", "Q6_K_M", "Q8_0", "IQ4_XS", "IQ4_NL"],
  "model_sizes": ["Qwen2.5-7B", "Qwen2.5-14B"],
  "min_pairs_required": 30,
  "configs_planned": 36,
  "configs_observed": 36,         // may be < 36 if a tag was absent
  "internal_rates": [0.73, ...],  // length == configs_observed
  "external_rates": [0.65, ...],  // length == configs_observed
  "pearson": 0.81,
  "pearson_ci_95": [0.67, 0.90],  // [r_lower, r_upper]
  "verdict": "valid_proxy",        // per interpret_correlation
  "interpreted_at": "2026-07-26T14:35:52Z",
  "operator": "<operator name or $USER>",
  "exit_code": 0
}
```

The report is written to the path given by `--output`; if omitted, the
script writes to `logs/external_eval_<study_id>_report.json`.

### Required artifacts

After a successful run the operator must archive:

1. **`logs/external_eval_<study_id>_report.json`** — the JSON report above.
2. **`logs/external_eval_<study_id>.jsonl`** — raw per-run event stream
   (the JSONL file the aggregator reads from `logs/traces.db`).
3. **`logs/external_eval_<study_id>_configs.txt`** — a copy of the configs
   file actually used (comment lines kept, to record which configurations
   were skipped due to missing tags).

These artifacts must be committed to the repository under `logs/` or a
named study directory within 48 hours of completing the run, so the
Pearson number is auditable and the study is reproducible. The ADR's
"Status" section is then updated with the Pearson number, the 95% CI,
and the verdict, and a follow-up issue is opened if the verdict is
`weak_proxy` or `invalid_proxy`.

## Consequences

- **What this ADR adds**: the exact experimental protocol for executing
  the correlation study that ADR-0023 §"Follow-ups" item 2 leaves
  unspecified.
- **Confidence interval**: Fisher's z CI is the standard method for
  Pearson r; it is symmetric around r after back-transformation and
  avoids the boundedness artefacts of naive confidence intervals near
  ±1.
- **36 configurations** provides 6 extra observations beyond the 30
  minimum, giving the study modest protection against a single
  non-recoverable run dropping the count below the threshold.
- **Two model sizes** (7B and 14B) ensure the correlation is not
  spuriously driven by a single model size; the 6 quantizations span
  the full quality/compression tradeoff space used in Phase 3.
- **Three harness versions** cover the evolved harness lineage and test
  whether correlation is robust across the version history.
- If the study exits with code 4 (zero variance), the internal and/or
  external task sets are insufficiently discriminating; the operator
  must replace the slice with a harder one or broaden the internal
  benchmark distribution before re-running.
- See [ADR-0023](0023-external-eval-validation-study.md) for the
  machinery this protocol executes, [ADR-0016](0016-phase-3-quantization-sweep.md)
  for the Phase 3 sweep infrastructure, and [ADR-0011](0011-failure-report-class-taxonomy.md)
  for the `critic_verdict` event payload consumed by the aggregator.
