# ADR-0023: External-eval validation study (issue #900)

## Status

Accepted. 2026-07-18.

## Context

[ADR-0005](0005-pytest-as-evaluation-framework.md) makes the Critic's
pass/fail signal come from `pytest` tasks marked
`@pytest.mark.benchmark`. [ADR-0009](0009-security-evals-benchmark-family.md)
grew that suite with security-shaped tasks, and
[ADR-0016](0016-phase-3-quantization-sweep.md) /
[ADR-0019](0019-quantization-intelligence-floor.md) /
[ADR-0020](0020-phase-3-findings.md) rely on the suite to discriminate
between model × quantization × harness-variant configurations. The
suite is internally consistent (every task ships with a deterministic
fixture, see `benchmarks/conftest.py::benchmark_workspace`), but it has
never been calibrated against an external, community-recognized coding
eval. Issue #900 asks the obvious question: *does our internal suite
rank agents the same way HumanEval+ ranks them?*

Issue #900's acceptance criteria are:

1. Run the internal benchmark suite against a HumanEval+ slice under
   `fx-runner`.
2. Compute Pearson correlation between internal and external pass/fail
   across ≥ 30 paired observations.
3. Document the outcome in an ADR.
4. Open a follow-up issue if the correlation is below the "valid proxy"
   threshold.

Pearson correlation across configurations is a per-configuration
quantity, not a per-task quantity. Concretely: each agent configuration
`k` (model × quantization × harness variant) produces two scalars —
`internal_pass_rate_k` (fraction of `benchmarks/tasks/` it passes) and
`external_pass_rate_k` (fraction of the HumanEval+ slice it passes).
Pearson is then computed across the `K` configurations, and criterion 2
requires `K ≥ 30`. Producing those 30 paired observations is an
expensive real-model operation: 30 configurations × (one internal-suite
run + one external-slice run) per configuration = ≥ 60 live
`fx-runner` invocations against a llama.cpp endpoint. That is operator
work, not CI work.

## Decision

We ship the machinery and an honest placeholder, not a fake number.

### Machinery (shipped in this PR)

- **`src/foundry_x/evaluation/correlation.py`** — pure-Python Pearson
  math with two guards and one threshold table:
  - `pearson_binary(x, y)` raises `UnderpoweredStudyError` when
    `MIN_PAIRED_OBSERVATIONS` (30, per criterion 2) is not met, and
    `ZeroVarianceError` (naming the offending series) when either
    series has zero variance. Both errors surface as actionable failures
    rather than silently emitting `nan`.
  - `interpret_correlation(r)` returns `'valid_proxy'` (`r ≥ 0.7`),
    `'weak_proxy'` (`0.3 ≤ r < 0.7`), or `'invalid_proxy'` (`r < 0.3`).
    The thresholds are documented in this ADR §"Thresholds" and pinned
    by `tests/test_correlation.py`.
- **`src/foundry_x/evaluation/humaneval_plus.py`** — EvalPlus-format
  loader and scorer: `HumanEvalTask` (pydantic model, per ADR-0006),
  `load_humaneval_slice()`, `run_canonical_solution()`,
  `run_candidate_solution()`, `slice_pass_rates()`. The `test` field is
  executed verbatim because the EvalPlus canonical format includes the
  `def check(candidate):` line in the assertion body.
- **`benchmarks/external/humaneval_plus_sample.jsonl`** — a 20-task
  HumanEval+ slice (`HumanEval/0,2,3,4,5,8,9,10,11,13,14,15,16,19,20,29,32,33,51,60`)
  with all canonical solutions verified to pass under
  `run_canonical_solution()`. Provenance and swap-in instructions are
  in `benchmarks/external/README.md`.
- **`benchmarks/tasks/test_external_eval_correlation.py`** — a
  `BenchmarkTask` carrying `@pytest.mark.benchmark` that exercises the
  full plumbing offline: slice integrity (20/20 canonical pass), wrong
  candidate returns `False`, broken candidate raises
  `HumanEvalExecutionError`, `pearson_binary` accepts perfect / inverse
  / zero / underpowered / zero-variance inputs correctly, and
  `interpret_correlation` honors the three thresholds. This runs in
  every pytest invocation that includes the benchmark marker and needs
  no network.
- **`tests/test_correlation.py`** and
  **`tests/test_humaneval_plus_loader.py`** — unit tests pinning the
  math, the guards, and the loader's edge cases (empty slice, missing
  fields, syntax-error candidate, malformed JSONL line, etc.).
- **`infra/scripts/run_external_eval.sh`** — the operator surface for
  the real-model study: launch llama-server (mirroring
  `infra/scripts/run_benchmark.sh`), drive one `fx-runner` invocation
  per configuration across the internal suite and the external slice,
  and aggregate the results. The script refuses to emit a fake Pearson
  number — see §"Placeholder" below.

### Placeholder (also shipped in this PR)

The aggregator inside `run_external_eval.sh` writes a *structural*
report rather than a Pearson number. The reason is honest:
per-configuration aggregation requires the runner to tag each session
with the agent configuration (model, quantization, harness variant) so
the aggregator can group `critic_verdict` events per configuration
(ADR-0011 governs that taxonomy). That runner-side session-metadata
tagging is not part of this PR; bundling it would violate the
single-concern rule (§"FoundryX way" 3.3 of `AGENTS.md`). The script
emits a `verdict: 'pending-runner-side-aggregation-plumbing'` field and
points back to this ADR.

### Thresholds

The three-band interpretation in `interpret_correlation` is a
*convention* this ADR establishes, not a derivation from first
principles:

| Band            | Range        | Meaning                                        |
| --------------- | ------------ | ---------------------------------------------- |
| `valid_proxy`   | `r ≥ 0.7`    | Internal suite is a defensible proxy for the   |
|                 |              | external ranking within the studied config     |
|                 |              | space. No follow-up required.                  |
| `weak_proxy`    | `0.3 ≤ r < 0.7` | Internal suite ranks configurations          |
|                 |              | *partially* like HumanEval+. Use with care;   |
|                 |              | file a follow-up to broaden the internal task  |
|                 |              | distribution.                                  |
| `invalid_proxy` | `r < 0.3`    | Internal suite does not reproduce the external |
|                 |              | ranking. File a follow-up issue per criterion  |
|                 |              | 4 of issue #900.                               |

The 0.7 / 0.3 cut points are conventional (they echo Cohen's
large/medium/small-effect boundaries for behavioral-science
correlations, scaled to a 0–1 positive-only range). They are
*overrideable*: a follow-up ADR can move them with evidence, but any
such move must re-run the full study.

## Consequences

- **What this PR proves**: the plumbing is sound. The slice loads, the
  canonical solutions pass, candidates are scored correctly, Pearson
  math accepts well- and ill-formed inputs, and the guards fire on
  under-powered / zero-variance inputs. Every code path the
  real-model study will exercise is exercised here offline.
- **What this PR does not prove**: the actual correlation number.
  Producing it requires ≥ 30 live `fx-runner` runs per leg (≥ 60
  total) against a llama.cpp endpoint, plus a runner-side change that
  records the agent configuration in session metadata so the
  aggregator can group `critic_verdict` events per configuration.
- **Follow-ups** (to be filed as separate issues, not bundled here):
  1. Runner-side session-metadata tagging: record model id,
     quantization, and harness-variant label on each session so the
     aggregator can recover per-configuration groupings from
     `logs/traces.db`.
  2. Run the study: 30+ configurations, real llama.cpp endpoint,
     produce the Pearson number, and update this ADR's "Status" with
     the result. If the result is `weak_proxy` or `invalid_proxy`,
     criterion 4 of issue #900 requires a follow-up issue describing
     how the internal suite will be broadened.
  3. (Optional) Replace the 20-task slice with the full 500-task
     EvalPlus set once the runner is fast enough to make the
     per-configuration cost acceptable.
- **No `pyproject.toml` change** is required: pydantic ≥ 2.6 is
  already a dependency ([ADR-0002](0002-uv-for-dependency-management.md),
  [ADR-0006](0006-pydantic-for-module-boundaries.md)), the slice is
  JSONL (stdlib `json`), and the loader uses `subprocess` + `tempfile`
  from the stdlib for sandboxed candidate execution.
- **No `harness/` change** is required: the agent is not involved in
  scoring. The loader and scorer are pure-Python foundry code; the
  agent only emits candidate function bodies, which the orchestrator
  scores via `run_candidate_solution()`.
- See [ADR-0005](0005-pytest-as-evaluation-framework.md) for the
  benchmark contract, [ADR-0006](0006-pydantic-for-module-boundaries.md)
  for the `BenchmarkTask` schema, [ADR-0011](0011-failure-report-class-taxonomy.md)
  for the `critic_verdict` event payload the future aggregator will
  consume, and [ADR-0016](0016-phase-3-quantization-sweep.md) for the
  real-model sweep infrastructure this study will eventually ride on.
