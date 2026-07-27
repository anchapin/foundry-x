# ADR-0027: Proxy-broadening contingency (issue #1045)

## Status

Accepted. 2026-07-26.

## Context

[ADR-0023] defines three proxy bands from the Pearson correlation between the
internal benchmark suite and a HumanEval+ slice:

| Band            | Range        | Meaning                                        |
| --------------- | ------------ | ---------------------------------------------- |
| `valid_proxy`   | `r ≥ 0.7`    | Internal suite is a defensible proxy for the     |
|                 |              | external ranking. No follow-up required.        |
| `weak_proxy`    | `0.3 ≤ r < 0.7` | Internal suite ranks configurations          |
|                 |              | *partially* like HumanEval+. File a follow-up  |
|                 |              | to broaden the internal task distribution.       |
| `invalid_proxy` | `r < 0.3`    | Internal suite does not reproduce the external  |
|                 |              | ranking. File a follow-up issue per criterion  |
|                 |              | 4 of issue #900.                              |

ADR-0023's §"Follow-ups" acknowledges that `weak_proxy` and `invalid_proxy`
require follow-up issues describing how the internal suite will be broadened,
but neither ADR-0023 nor any issue describes what the broadening plan actually
is. This ADR fills that gap.

## Gap analysis

### Audit of existing task families

Every task in `benchmarks/tasks/*.py` carries a `BenchmarkTask` with a `tags`
field (per `benchmarks/models.py::BenchmarkTask.tags`, ADR-0006). The
population as of this writing is:

| Tag family | Representative tags | HumanEval+ coverage |
| ---------- | ------------------- | -------------------- |
| Security   | `security`, `injection`, `sandbox`, `runaway`, `rate-limit` | None |
| Debugging  | `debugging`, `syntax`, `imports`, `grep`, `diagnosis` | Minimal |
| Refactoring | `refactoring`, `rename`, `cross-file`, `multi-file` | None |
| Multi-step | `multi-step`, `chained`, `planning`, `decision-making` | None |
| File I/O   | `file-creation`, `write_file`, `read_file`, `read`, `write` | None |
| Agent-loop | `agent-loop`, `context-pruning`, `token-aware`, `phase-3` | None |
| Infrastructure | `smoke`, `infrastructure`, `pytest`, `benchmark` | None |
| Configuration | `config`, `editing`, `precision`, `surgical` | None |
| Git/ops    | `git`, `ops` | None |
| Testing    | `testing` | None |
| Documentation | `documentation`, `comprehension` | None |
| Math/alg   | `algorithms`, `hashing`, `math`, `recurrence`, `sorting`, `strings` | Minimal |
| Io         | `io` | Partial |

**Finding**: HumanEval+ is an *implementation* benchmark — it measures how
reliably an agent can produce a correct single-function body from a natural-
language specification. The internal suite has abundant coverage for debugging,
refactoring, multi-file, and security task families, but **minimal coverage for
pure single-function implementation tasks**. This structural mismatch explains
why the correlation can be weak: a model that excels at multi-step debugging
and refactoring may perform differently on isolated function implementation.

## Decision

We define a two-level broadening ladder. Progression through the levels is
sequential; each level is a prerequisite for the next.

### Level 1 — `weak_proxy` remediation

**Trigger**: Pearson correlation falls in `0.3 ≤ r < 0.7`.

**Goal**: Close the implementation gap with a small, high-signal batch of
single-function tasks that map directly to HumanEval+ categories.

**Action**: Add **8 single-function implementation tasks** drawn from the
following HumanEval+ categories, implemented as `BenchmarkTask` instances in
`benchmarks/tasks/`:

| # | Task name pattern | HumanEval+ category | Difficulty |
|---|-------------------|-------------------|-----------|
| 1 | `test_implement_two_sum` | Array/hashing | easy |
| 2 | `test_implement_reverse_string` | String | easy |
| 3 | `test_implement_nth_fibonacci` | Recurrence/math | easy |
| 4 | `test_implement_is_palindrome` | String | easy |
| 5 | `test_implement_merge_sorted_lists` | Linked list / merging | medium |
| 6 | `test_implement_binary_search` | Search | medium |
| 7 | `test_implement_valid_parentheses` | Stack | medium |
| 8 | `test_implement_max_subarray` | Dynamic programming | medium |

Each task:
- Is a standalone `BenchmarkTask` with `name`, `description`, `prompt`,
  `expected_outcome`, `difficulty_tier`, and `tags=["implementation"]`.
- Requires only `read_file`/`write_file` (no bash, no multi-file navigation).
- Is deterministic and offline-executable (no network, no live server).
- Follows the fixture convention from `benchmarks/conftest.py::benchmark_workspace`.

**Exit criterion**: After adding the 8 tasks, re-run the correlation study
against the same 20-task HumanEval+ slice. If `r ≥ 0.7`, the suite is a
`valid_proxy`. If still `r < 0.7`, escalate to Level 2.

**Files to create/modify**:
- `benchmarks/tasks/test_implement_two_sum.py`
- `benchmarks/tasks/test_implement_reverse_string.py`
- `benchmarks/tasks/test_implement_nth_fibonacci.py` (note: `test_nth_fibonacci.py` exists; confirm it is tagged `implementation` and add if missing)
- `benchmarks/tasks/test_implement_is_palindrome.py`
- `benchmarks/tasks/test_implement_merge_sorted_lists.py`
- `benchmarks/tasks/test_implement_binary_search.py`
- `benchmarks/tasks/test_implement_valid_parentheses.py`
- `benchmarks/tasks/test_implement_max_subarray.py`
- `docs/adr/0023-external-eval-validation-study.md` — add a "Broadening log"
  section recording which level was applied and the resulting `r`.

### Level 2 — `invalid_proxy` remediation

**Trigger**: Pearson correlation falls below `r < 0.3`.

**Goal**: Achieve full structural alignment with the external benchmark by
expanding to the complete EvalPlus dataset.

**Action**: Replace the 20-task `humaneval_plus_sample.jsonl` slice with the
full **164-task EvalPlus `humaneval_plus.jsonl`** dataset. The loader in
`src/foundry_x/evaluation/humaneval_plus.py` already supports arbitrary slices
via `load_humaneval_slice()`; the transition is purely a data swap and a
corresponding update to `benchmarks/external/README.md`.

**Rationale**: A 20-task slice is statistically underpowered and vulnerable to
sampling bias. The full 164-task set covers a wider range of difficulty levels
and problem types, providing a more robust correlation signal. The cost increase
per configuration (164 vs 20 tasks) is acceptable at `invalid_proxy` severity.

**Precondition**: Level 1 tasks must be present and passing at the same
correlation run to ensure the additional tasks are not themselves confounded by
coverage gaps.

**Exit criterion**: After expanding to 164 tasks, re-run the correlation study.
If `r ≥ 0.7`, the suite is a `valid_proxy`. If `r ≥ 0.3` but `< 0.7`,
Level 1 was insufficient and this ADR should be updated with additional
implementation tasks. If `r < 0.3` persists, the fundamental assumption that
the internal suite can replicate HumanEval+ rankings is itself false and a
separate ADR is required to revisit the correlation study design.

**Files to create/modify**:
- `benchmarks/external/humaneval_plus.jsonl` (164-task EvalPlus full set)
- `benchmarks/external/README.md` (swap-in instructions updated)
- `infra/scripts/run_external_eval.sh` (slice selector updated to `full`)

## Consequences

- **Level 1 tasks are implementation-only**: they do not overlap with any
  existing security, debugging, or refactoring task families, so they add
  orthogonal signal without diluting existing coverage.
- **Level 2 is a one-time data swap**: no code changes to `correlation.py`,
  `humaneval_plus.py`, or the `BenchmarkTask` schema are required.
- **The broadening ladder is monotonic**: Level 1 tasks remain in the suite
  regardless of whether Level 2 is reached; they are not reverted.
- **The correlation study remains the source of truth**: the ladder is a
  contingency plan, not a commitment to broaden on a schedule. The study
  results drive the decision.
- **No `harness/` changes** are involved: the broadening plan operates on
  the benchmark suite only, not on the agent harness.
- See [ADR-0005](0005-pytest-as-evaluation-framework.md) for the benchmark
  contract, [ADR-0006](0006-pydantic-for-module-boundaries.md) for the
  `BenchmarkTask` schema, and [ADR-0023](0023-external-eval-validation-study.md)
  for the proxy-band definitions and study protocol.
