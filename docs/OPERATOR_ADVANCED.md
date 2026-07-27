# OPERATOR_ADVANCED.md

> Advanced operational workflows for operators who have completed the
> [basic tutorial](./TUTORIAL.md) and read [OPERATOR.md](./OPERATOR.md).
> This document does not repeat the evolution loop basics — it covers
> the two workflows that go beyond single-harness iteration:
>
> 1. **Quantization sweep** — running the benchmark suite across
>    quantizations to discover the intelligence floor.
> 2. **External eval study** — correlating the internal benchmark suite
>    against a HumanEval+ slice to validate it as a proxy.
>
> Both workflows require a live model endpoint (a local `llama-server`
> or any OpenAI-compatible host). See
> [MODEL_CONFIG.md](./MODEL_CONFIG.md) for endpoint configuration and
> [`infra/llama-cpp/README.md`](../infra/llama-cpp/README.md) for
> hardware setup.

## Prerequisites

Both workflows assume you can already:

- Run the [tutorial](./TUTORIAL.md) end-to-end (plant a trace, read it,
  compute KPIs).
- Launch a `llama-server` endpoint or point FoundryX at a remote model
  ([MODEL_CONFIG.md §1](./MODEL_CONFIG.md#1-endpoint-configuration)).
- Read the [evolution loop](./OPERATOR.md#the-evolution-loop) diagram in
  OPERATOR.md.

These workflows are **real-model operations** — they consume tokens and
GPU time. They are not part of CI. The offline plumbing for each is
validated by a benchmark task under `benchmarks/tasks/` that runs in
every `pytest` invocation.

---

## Part 1 — Quantization sweep

The quantization sweep runs the benchmark suite against multiple GGUF
quantizations of the same model and produces a comparison table showing
pass rates, cycle times, token efficiency, and cost per task. The goal
is to find the **intelligence floor** — the minimum quantization level
that preserves benchmark performance within the regression threshold.

The sweep is governed by [ADR-0016](./adr/0016-phase-3-quantization-sweep.md)
(design) and [ADR-0019](./adr/0019-quantization-intelligence-floor.md)
(intelligence-floor findings). It is **purely diagnostic**: the Evolver
never proposes quantization changes (ADR-0016 §5) — quantization
selection is a hardware- and cost-constrained operator decision.

### 1.1 — Prerequisites

**Hardware:**

| Component | Minimum | Notes |
|-----------|---------|-------|
| GPU VRAM | 8 GB | A Q5_K_M 7B model needs ~5.5 GB at full offload. Q8 needs ~7–8 GB and may require partial offload on an 8 GB card. See ADR-0019 §"Quantization Levels". |
| Disk | GGUF files for each target quantization | One `.gguf` per quantization label you sweep. |

**Software:**

- A working `llama-server` build, or a remote OpenAI-compatible endpoint.
- GGUF model files named `<model>.<QUANT>.gguf` under a single directory
  pointed to by `FOUNDRY_MODEL_PATH`.

**Environment variables:**

| Env var | Required? | Purpose |
|---------|-----------|---------|
| `FOUNDRY_MODEL_PATH` | **Yes** | Directory containing the quantized GGUF files. The sweep matches each file via a glob `*.<quant>.gguf`. |
| `OPENCODE_SERVER_URL` / `LLAMACPP_HOST` | Yes (one) | The live model endpoint. See [MODEL_CONFIG.md §1](./MODEL_CONFIG.md#1-endpoint-configuration). |
| `FOUNDRY_REGRESSION_THRESHOLD` | No | Regression threshold in percentage points (default 2.0). A candidate's pass rate must be within this many pp of the baseline. |

> **Friction point — missing model files.** The sweep matches model files
> by the glob `*.<QUANT>.gguf` inside `FOUNDRY_MODEL_PATH`. If a
> quantization label in `--quantizations` has no matching file, the
> `Critic.quantization_sweep()` call raises `FileNotFoundError` and the
> CLI exits with code 1. Verify the files exist before running:
>
> ```bash
> ls "$FOUNDRY_MODEL_PATH"/*.Q{4,5,6,8}*.gguf
> ```

### 1.2 — Running the sweep

The sweep is invoked via the `foundry-sweep` CLI (registered in
`pyproject.toml` §`[project.scripts]`, entry point
[`sweep_main`][foundry_x.evolution.cli.sweep_main]):

```bash
uv run foundry-sweep \
    --quantizations Q4_K_M,Q5_K_M,Q6_K,Q8_0 \
    --harness-dir harness \
    --baseline Q5_K_M \
    --regression-threshold 2.0 \
    --output logs/quantization_results.json
```

You can also reach it as a subcommand of `foundry-evolve`:

```bash
uv run foundry-evolve sweep \
    --quantizations Q4_K_M,Q5_K_M,Q6_K,Q8_0 \
    --harness-dir harness
```

**Flags:**

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `--quantizations` | Yes | — | Comma-separated quantization labels (e.g. `Q4_K_M,Q5_K_M,Q6_K,Q8_0`). |
| `--harness-dir` | Yes | — | Path to the harness directory to evaluate. |
| `--baseline` | No | First in `--quantizations` | Baseline quantization to compare candidates against. |
| `--regression-threshold` | No | `2.0` | Pass-rate slack in percentage points. |
| `--cost-per-token` | No | — | USD per token for cost-per-task. Also settable via `FOUNDRY_COST_PER_TOKEN`. |
| `--output` | No | — | Path to write the verdict as JSON. |

**Exit codes:**

| Code | Meaning |
|------|---------|
| 0 | Sweep completed, no regression detected. |
| 1 | Sweep completed but a candidate regressed past the threshold, **or** a sweep error occurred. |
| 2 | Usage error (e.g. empty quantization list). |

> **Dry run.** There is no built-in `--dry-run` for the sweep. To confirm
> your environment without spending tokens, set
> `TEST_MODEL_MODE=mock` and run one benchmark task first:
>
> ```bash
> uv run pytest benchmarks/tasks/ -m benchmark -x
> ```
>
> If that passes, your harness loads cleanly and the benchmark suite is
> wired. Then switch back to a real model endpoint for the sweep.

### 1.3 — Interpreting the results

The CLI prints a comparison table:

```
Quantization Sweep Results
==========================================================================================
  Quantization   | Pass Rate |  Avg Cycle |     Tokens |     Tok/s |  Cost/Task | Model ID
------------------------------------------------------------------------------------------
  Q4_K_M         |     88.0% |    142.3s |     128450 |      902.7 |    $0.0128 | qwen2.5-7b-q4_k_m
  Q5_K_M         |     95.0% |    168.1s |     135200 |      804.3 |    $0.0135 | qwen2.5-7b-q5_k_m
  Q6_K           |     97.0% |    185.4s |     137800 |      743.1 |    $0.0138 | qwen2.5-7b-q6_k
  Q8_0           |     99.0% |    210.7s |     141000 |      668.8 |    $0.0141 | qwen2.5-7b-q8_0
==========================================================================================
Recommended: Q5_K_M  [No regression]
```

**The intelligence floor** is the lowest quantization whose pass rate is
within `--regression-threshold` (default 2 pp) of the baseline. In the
table above, Q5_K_M is recommended because its 95% pass rate is within
2 pp of the Q8_0 baseline (99%) — wait, that is a 4 pp gap, so Q5_K_M
would actually be flagged as a regression against Q8_0. Adjust your
`--baseline` and `--regression-threshold` to reflect what you consider
acceptable.

**Reading the regression flag:**

- `No regression` — every candidate's pass rate is within the threshold
  of the baseline. The recommended quantization is the *lowest* pass-rate
  candidate that clears the bar (maximising speed/VRAM efficiency).
- `REGRESSION DETECTED` — at least one candidate dropped more than the
  threshold below the baseline. The exit code is 1; if this runs at the
  CI release gate it blocks the merge (ADR-0016 §4).

**Token-budget aborts** (`FOUNDRY_TOKEN_BUDGET` reached) are classified
as *task-shaped failures*, not harness regressions (ADR-0016 §6). They
are excluded from the pass-rate denominator. A task that aborts on Q4
because the context budget was too small does not count against Q4's
quality — it counts against the task's fit for that model size.

**When `--output` is set**, the verdict is persisted as JSON matching the
`QuantizationVerdict` schema (pydantic model per ADR-0006):

```jsonc
{
  "quantizations": [ { "quantization": "Q5_K_M", "pass_rate": 0.95, ... } ],
  "recommended": "Q5_K_M",
  "regression": false
}
```

### 1.4 — How results feed the Evolver

The sweep is diagnostic only. The `QuantizationVerdict` does **not**
trigger a `ProposedEdit` — the Evolver operates on harness files
(`manifest.json`, `system_prompt.txt`, `skills/`), not on deployment
parameters (ADR-0016 §5). The operator reads the verdict and decides
whether to change the production quantization in `.env` /
`FOUNDRY_MODEL_ID`. If you want the harness itself to adapt, route a
failure report through the
[`foundry-evolve evolve`](./OPERATOR.md#foundry-evolve-flag-reference-issue-888)
loop as you would for any harness change.

### 1.5 — Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `FileNotFoundError: no model file matching *.<Q>.gguf` | Missing GGUF for a quantization label. | Download or generate the file; verify the naming matches the glob. |
| Exit 1 with all-zero pass rates | The model endpoint never responded (wrong host, server down). | Check `LLAMACPP_HOST` / `OPENCODE_SERVER_URL`; confirm `/health` returns ok. See OPERATOR.md §[Server supervision](./OPERATOR.md#server-supervision-foundryservermanager-issue-899). |
| `server_unavailable` events in the trace | GPU OOM crashed `llama-server` mid-sweep. | Reduce `FOUNDRY_SERVER_NGpuLayers` (partial offload) or drop the heaviest quantization from the sweep. See OPERATOR.md §[Recovering from server unavailable](./OPERATOR.md#recovering-from-server-unavailable). |
| All quantizations pass at ~100% | The benchmark slice is too easy to discriminate. | The sweep needs harder tasks to find the floor. The external eval study (Part 2) exists to catch exactly this. |

---

## Part 2 — External eval study

The external eval study correlates the **internal** benchmark suite
against an **external**, community-recognized eval (a HumanEval+ slice)
to answer: *does our internal suite rank agents the same way HumanEval+
ranks them?* The output is a Pearson correlation coefficient `r` and a
verdict band (`valid_proxy`, `weak_proxy`, or `invalid_proxy`).

The study is governed by [ADR-0023](./adr/0023-external-eval-validation-study.md)
(machinery) and [ADR-0032](./adr/0032-external-eval-pearson-study-protocol.md)
(execution protocol). Pearson `r` is computed **per configuration**
(model × quantization × harness variant), not per task — so you need
≥ 30 paired observations to satisfy the study's power requirement.

### 2.1 — Prerequisites

**Hardware:**

Same as the quantization sweep (§1.1), plus enough time for
`≥ 30 configurations × 2 runs` (internal suite + external slice) = ≥ 60
live `fx-runner` invocations. Budget several hours on a single GPU.

**Software / artifacts:**

- The HumanEval+ slice at
  [`benchmarks/external/humaneval_plus_sample.jsonl`](../benchmarks/external/humaneval_plus_sample.jsonl)
  (20 tasks; provenance in
  [`benchmarks/external/README.md`](../benchmarks/external/README.md)).
- The operator script
  [`infra/scripts/run_external_eval.sh`](../infra/scripts/run_external_eval.sh).
- A configs file listing ≥ 30 agent configurations (one per line).

**Environment variables:**

| Env var | Required? | Default | Purpose |
|---------|-----------|---------|---------|
| `LLAMACPP_HOST` | No | `http://127.0.0.1:8080` | Host for the `/health` probe and model endpoint. |
| `LLAMACPP_SERVER_BIN` | No | `$LLAMACPP_DIR/build/bin/llama-server` | Path to the `llama-server` binary (auto-launches if endpoint is down). |
| `LLAMACPP_DIR` | No | `$HOME/llama.cpp` | llama.cpp checkout root. |
| `LLAMACPP_NGL` | No | `0` | GPU layers to offload when auto-launching. |
| `LLAMACPP_HEALTH_TIMEOUT` | No | `60` | Seconds to wait for `/health` after launch. |
| `FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS` | No | `30` | Override the paired-observation minimum (issue #900 criterion 2). |

> **Friction point — the 30-configuration minimum.** The script exits
> with code **3** if your configs file lists fewer than
> `FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS` (default 30) configurations. This is
> the statistical power floor: below 30 pairs, `pearson_binary` raises
> `UnderpoweredStudyError`. ADR-0032 §"Study scope" provides a ready-made
> 36-configuration file (3 harness versions × 6 quantizations × 2 model
> sizes) — copy it rather than authoring your own.

### 2.2 — Pre-flight (no model tokens)

Before spending any tokens, validate the slice integrity offline:

```bash
uv run --quiet python -c "
from foundry_x.evaluation.humaneval_plus import load_humaneval_slice, slice_pass_rates
tasks = load_humaneval_slice('benchmarks/external/humaneval_plus_sample.jsonl')
passed, total = slice_pass_rates(tasks)
assert passed == total, f'slice integrity: {passed}/{total}'
print(f'slice ok: {total} tasks, all canonical solutions pass')
"
```

If this fails, the slice file is corrupt or the loader changed — do not
proceed. The offline benchmark task
[`benchmarks/tasks/test_external_eval_correlation.py`](../benchmarks/tasks/test_external_eval_correlation.py)
validates the same thing in CI, so a failure here means your checkout is
out of sync.

### 2.3 — Building the configs file

Each non-comment line in the configs file is one agent configuration:

```
<label> <fx-runner args...>
```

For example (abbreviated — see [ADR-0032](./adr/0032-external-eval-pearson-study-protocol.md)
§"Commands to run" for the full 36-line file):

```
q7b-q4km  --model Qwen2.5-7B-Q4_K_M.gguf  --quantization Q4_K_M  --harness-version v1.0
q7b-q5km  --model Qwen2.5-7B-Q5_K_M.gguf  --quantization Q5_K_M  --harness-version v1.0
# ...28 more lines for >= 30 configurations...
```

The `--quantization` and `--harness-version` args are consumed by the
aggregator to group `critic_verdict` events per configuration from
`logs/traces.db`. This grouping relies on the runner-side
session-metadata tagging completed in issue #1007 (ADR-0023 follow-up
item 1): every `fx-runner` session now records `model_id`,
`quantization`, and `harness_version` in the `session_start` event and
the sqlite `sessions` table.

> **Friction point — missing harness tags.** ADR-0032 maps harness
> versions to git tags (`harness-v1.0`, etc.). Verify the tags exist
> before running:
>
> ```bash
> git tag -l "harness-v1.*"
> ```
>
> A missing tag is skipped with a warning; the study proceeds with the
> remaining configurations **provided at least 30 paired observations
> remain**. If too many tags are absent you drop below the power floor
> and the aggregator refuses to compute Pearson.

### 2.4 — Running the study

```bash
infra/scripts/run_external_eval.sh \
    --model /srv/models/Qwen2.5-14B-Q8_0.gguf \
    --configs /tmp/eval_configs_36.txt \
    --output logs/external_eval_36configs_report.json
```

**Flags:**

| Flag | Required | Default | Purpose |
|------|----------|---------|---------|
| `--model` | Yes | — | GGUF path; auto-launches `llama-server` if the endpoint is down. |
| `--configs` | Yes | — | File with one configuration per line. |
| `--slice` | No | `benchmarks/external/humaneval_plus_sample.jsonl` | HumanEval+ slice JSONL. |
| `--keep-server` | No | off | Do not tear down an auto-launched `llama-server` on exit. |
| `--dry-run` | No | off | Print the planned runs and exit without invoking `fx-runner`. |
| `--output` | No | `logs/external_eval_<study_id>_report.json` | JSON report path. |

**Dry run first.** Always run with `--dry-run` once to confirm the plan
before committing GPU time:

```bash
infra/scripts/run_external_eval.sh \
    --model /srv/models/Qwen2.5-14B-Q8_0.gguf \
    --configs /tmp/eval_configs_36.txt \
    --dry-run
```

**Exit codes:**

| Code | Meaning | Action |
|------|---------|--------|
| 0 | Study completed; correlation is reportable. | Archive artifacts (§2.6). |
| 2 | CLI usage error (missing `--model` / `--configs`). | Fix arguments. |
| 3 | Under-powered: fewer than `MIN_PAIRS` configurations. | Add configurations to the file. |
| 4 | Zero variance in one series (Pearson undefined). | Choose a more discriminating task set / slice. |
| 5 | One or more runs failed non-recoverably. | Read `stderr`; check endpoint and model files. |

### 2.5 — Interpreting Pearson `r`

The aggregator writes a JSON report (shape documented in
[ADR-0032](./adr/0032-external-eval-pearson-study-protocol.md) §"Output
format") containing `pearson`, `pearson_ci_95` (a Fisher's-z 95%
confidence interval), and a `verdict` band:

| Band | Range | Meaning |
|------|-------|---------|
| `valid_proxy` | `r ≥ 0.7` | The internal suite is a defensible proxy for the external ranking. No follow-up required. |
| `weak_proxy` | `0.3 ≤ r < 0.7` | The internal suite ranks configurations *partially* like HumanEval+. File a follow-up to broaden the internal task distribution. |
| `invalid_proxy` | `r < 0.3` | The internal suite does not reproduce the external ranking. File a follow-up issue per issue #900 criterion 4. |

The thresholds are a convention established by ADR-0023 §"Thresholds"
(they echo Cohen's large/medium/small-effect boundaries). They are
overridable by a follow-up ADR, but any change must re-run the full
study.

**Guards you may hit:**

- `UnderpoweredStudyError` — fewer than `MIN_PAIRED_OBSERVATIONS` (30)
  paired observations survived aggregation. The script surfaces this as
  exit code 3. Add configurations.
- `ZeroVarianceError` — one of the pass-rate series is constant (e.g.
  every configuration passed 100% of the internal suite). Pearson is
  undefined. The script surfaces this as exit code 4. You need tasks
  that *discriminate* between configurations — the sweep in Part 1 is
  the tool for finding that floor.

### 2.6 — Archiving the result

After a successful run (exit 0), archive the artifacts per
[ADR-0032](./adr/0032-external-eval-pearson-study-protocol.md) §"Required
artifacts":

1. `logs/external_eval_<study_id>_report.json` — the JSON report.
2. `logs/external_eval_<study_id>.jsonl` — raw per-run event stream.
3. The configs file actually used.

These must be committed (or attached to the relevant issue) within 48
hours so the Pearson number is auditable. If the verdict is `weak_proxy`
or `invalid_proxy`, open a follow-up issue describing how the internal
suite will be broadened — issue #900 criterion 4 makes this mandatory.

### 2.7 — Common failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Exit 3: "requires ≥ 30 paired observations" | Configs file has too few lines. | Use the 36-configuration file from ADR-0032, or lower `FOUNDRY_EXTERNAL_EVAL_MIN_PAIRS` only if you understand the power tradeoff. |
| Exit 4: "zero variance" | Every config passed (or failed) identically on one leg. | The internal suite or the slice is too easy/hard. Run the quantization sweep (Part 1) to find discriminating tasks first. |
| Exit 5: "traces.db did not grow" | `fx-runner` ran but wrote nothing — usually a model error. | Check the session outcome in the trace store: `foundry-trace show <sid>`. Look for `server_unavailable` events. |
| Verdict is `pending-runner-side-aggregation-plumbing` | The legacy placeholder path (ADR-0023) fired instead of the ADR-0032 aggregator. | Ensure `run_external_eval.sh` and `foundry_x.evaluation.correlation` are both from the same checkout (post-issue-#1028). |

---

## Relationship between the two workflows

The quantization sweep and the external eval study are complementary:

- **Sweep** answers *"which quantization is fastest without losing
  quality?"* — it finds the intelligence floor.
- **External eval** answers *"is our benchmark suite measuring what we
  think it is?"* — it validates the suite as a proxy.

A typical advanced operator sequence: run the sweep to settle the
production quantization, then run the external eval to confirm the
benchmark suite that the sweep (and the Critic gate) depend on is a
trustworthy proxy. If the external eval returns `weak_proxy` or
`invalid_proxy`, the sweep's verdicts — and every Critic gate decision —
should be treated with correspondingly more caution until the internal
suite is broadened.

Both workflows write to the same `logs/` SQLite database (ADR-0016 §2),
so the trace-driven KPI framework ([ADR-0007](./adr/0007-trace-driven-development.md))
queries work without modification across sweep, eval, and routine
evolution sessions.

---

## Back to OPERATOR.md

This document covers the two advanced workflows. For everything else —
the evolution loop, degradation modes, server supervision, the
`foundry-evolve` flag reference — see [OPERATOR.md](./OPERATOR.md).
