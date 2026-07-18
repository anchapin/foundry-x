# External Eval Slices

This directory holds slices of external coding evals that the internal
benchmark suite under `benchmarks/tasks/` is validated against. The
validation study is governed by [ADR-0023](../../docs/adr/0023-external-eval-validation-study.md)
and motivated by [issue #900](https://github.com/anchapin/foundry-x/issues/900).

## Why this exists

[ADR-0005](../../docs/adr/0005-pytest-as-evaluation-framework.md) deferred
adoption of external benchmark frameworks until a "concrete limitation
forced an ADR." [Issue #900](https://github.com/anchapin/foundry-x/issues/900)
is that limitation: we cannot answer whether an improvement in
`kpi-improvement-rate` (PRD §5) measured on `benchmarks/tasks/` translates
to real coding capability. This directory is the external half of the
study that closes that gap.

## Layout

```
benchmarks/external/
  README.md                          # this file
  __init__.py                        # import marker
  humaneval_plus_sample.jsonl        # 20-task HumanEval+ slice
```

## `humaneval_plus_sample.jsonl`

A 20-task slice modeled on the EvalPlus `HumanEval+` JSONL shape. Each
row has the canonical EvalPlus fields:

| Field                | Meaning                                                        |
| -------------------- | -------------------------------------------------------------- |
| `task_id`            | EvalPlus-style id (e.g. `HumanEval/0`).                        |
| `prompt`             | Function signature + docstring handed to the agent.            |
| `canonical_solution` | Reference solution body; concatenated onto `prompt` to exec.   |
| `test`               | Source of a `check(candidate)` function in EvalPlus format.    |
| `entry_point`        | Function name the agent must implement.                        |

The slice is intentionally small (20 tasks, well under the 164-task full
HumanEval+): the ADR-0023 methodology pairs each agent configuration
across the full internal suite plus this slice, so 20 external tasks
give enough signal at a fraction of the model-token cost.

### Provenance

The tasks are reconstructions of the original OpenAI `HumanEval` problems
(MIT-licensed, https://github.com/openai/human-eval) in the EvalPlus
JSONL shape. The slice is *not* a verbatim copy of the EvalPlus dataset:
test bodies are reduced to a small number of representative assertions
per task to keep the slice auditable in a single screen. The canonical
solutions are the textbook implementations; each one is verified to
pass its own `check` at slice-load time by
`foundry_x.evaluation.humaneval_plus.slice_pass_rates`, and that check
is re-run by the plumbing-validation benchmark
`benchmarks/tasks/test_external_eval_correlation.py` on every pytest
invocation that includes the `benchmark` marker.

### Swapping in the real EvalPlus dataset

An operator who wants the full 164-task study can drop the official
`humaneval-plus.jsonl` in place of `humaneval_plus_sample.jsonl` and
re-run `infra/scripts/run_external_eval.sh`. The loader
(`foundry_x.evaluation.humaneval_plus.load_humaneval_slice`) reads any
file in this shape; no other code needs to change.

## How the slice is used

Two paths, mirroring the [benchmarks/](../README.md) split:

1. **Plumbing validation (offline, deterministic, in CI).**
   `benchmarks/tasks/test_external_eval_correlation.py` loads the slice
   via `foundry_x.evaluation.humaneval_plus.load_humaneval_slice`,
   verifies every canonical solution passes its own `check`, then runs
   a synthetic correlation scenario through
   `foundry_x.evaluation.correlation.pearson_binary` to confirm the
   math is correct. This catches regressions in the plumbing without
   spending model tokens.

2. **Real-model study (online, orchestrated).**
   `infra/scripts/run_external_eval.sh` launches `llama-server`, drives
   `fx-runner` once per agent configuration across the internal suite
   and the external slice, persists the per-task pass/fail into the
   trace store, then computes Pearson correlation across the
   configurations. Issue #900 requires ≥30 paired observations before
   a correlation is reportable.

## See also

- [ADR-0023](../../docs/adr/0023-external-eval-validation-study.md) — methodology and decision.
- [ADR-0005](../../docs/adr/0005-pytest-as-evaluation-framework.md) — pytest as the unified evaluation framework.
- [Issue #900](https://github.com/anchapin/foundry-x/issues/900) — motivating issue.
- `src/foundry_x/evaluation/correlation.py` — Pearson correlation math.
- `src/foundry_x/evaluation/humaneval_plus.py` — slice loader and canonical-solution runner.
