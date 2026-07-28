# ADR-0028: Add "hard" to DifficultyTier and define hard-tier task archetypes

## Status

Accepted. 2026-07-28.

## Context

ADR-0005 defines the benchmark framework and ADR-0006 established the
`BenchmarkTask` schema.  `benchmarks/models.py` defines
`DifficultyTier` as:

```python
DifficultyTier = Literal["smoke", "easy", "medium"]
```

The `cross_file_refactor` task (`benchmarks/tasks/test_cross_file_refactor.py`)
is the reference "medium" task: it requires reading three files and
editing two of them in a coordinated way.  The `grep_search_fix` task
(`benchmarks/tasks/test_grep_search_fix.py`) requires using a search tool
to locate a stale reference across a multi-file package before editing it.

Both medium tasks are solvable in a single linear pass: read → locate →
edit → verify.  The benchmark suite currently has no tasks that require
a branching decision tree, that demand keeping intermediate state across
multiple tool-call rounds, or that require the agent to recover from an
incorrect hypothesis partway through a multi-step plan.

Issue #1043 proposes adding a "hard" tier to improve the discriminative
power of the suite.  This ADR defines what "hard" means in this
codebase, specifies concrete task archetypes, and updates `DifficultyTier`
to include the new value.

## Decision

### 1. Extend DifficultyTier

`DifficultyTier` in `benchmarks/models.py` is updated to:

```python
DifficultyTier = Literal["smoke", "easy", "medium", "hard"]
```

No other schema changes are required.  Existing tasks keep their current
`tier` values.

### 2. Define what makes a task "hard"

A task belongs to the **hard** tier when it satisfies **all** of:

1. **Multi-phase reasoning** — the task cannot be solved with a single
   read → locate → edit pass.  The agent must form and test a hypothesis,
   observe a secondary failure, and revise its plan before completing the
   fix.  At least two distinct reasoning states are required (e.g.
   "initial hypothesis failed → pivot to alternative").

2. **Cross-module scope** — the task involves four or more files across
   at least two distinct Python packages or module namespaces (e.g. a
   library package, a CLI driver, a config module, and a test package).
   The agent must correctly identify which files require changes without
   a hint in the prompt.

3. **Non-trivial state management** — the task requires the agent to
   track or update state that spans multiple tool-call rounds and is not
   directly observable in a single file (e.g. a configuration value
   read early in the session that constrains a fix applied later, or a
   runtime value computed in one module that must be propagated to a
   different module).

4. **Precise expected outcome** — the task has a deterministic pass/fail
   criterion that can be evaluated programmatically (see §4 below).

A task that satisfies only one or two of the above criteria remains at
the medium tier.  This definition intentionally excludes tasks that
require parallel independent subtasks (a future "extreme" tier concern).

### 3. Hard-tier task archetypes

Two archetypes are defined.  Each archetype maps to one or more concrete
tasks that must be added to `benchmarks/tasks/` as part of the
implementation of this ADR.

#### Archetype H1: Complex debugging across modules

A bug lives at the intersection of two modules.  The symptom is visible
only when the full stack is exercised, but the root cause is in a
different module than the one that raises the exception.  The agent must:

- Trace a runtime exception back to its origin module.
- Identify that the caller in module A passes invalid arguments to
  module B because of a stale contract between them.
- Fix module B to be defensive against the invalid input (or fix module
  A to stop producing it).
- Verify the fix by running the full test suite or exercising the
  CLI entry point.

The `benchmark_workspace` is seeded with four or more files spanning
two packages.  The pre-condition check shows a real failure (exception
or test failure) that is not in the file that will be edited.  The
post-condition requires both the library tests and the CLI driver to
produce correct output.

Example concrete task: `test_debug_import_cycle` — two packages
`pkg_a` and `pkg_b` each import the other, causing an `ImportError`
during collection.  The agent must identify the cycle and break it by
moving the offending import inside the function that needs it (deferred
import pattern), then confirm `python -m pytest` exits 0.

#### Archetype H2: Multi-file coordinated refactor with constraint

A library exposes a public API that is used across multiple packages.
The agent must change the signature or behaviour of the public API in a
backward-incompatible way, then update every caller to match the new
contract.  The agent is not told which files use the API; it must
discover them.  Additionally, a constraint applies: a test in one of
the caller packages must still pass after the refactor, demonstrating
the agent did not simply delete the test to make the suite green.

The agent must:

- Read the library's public API to understand the current signature.
- Grep or search to discover all callers across the workspace.
- Update the library's implementation.
- Update every caller in a coordinated way.
- Leave a specific test passing to prove the constraint was respected.
- Confirm `python -m pytest` exits 0 across all packages.

Example concrete task: `test_refactor_api_with_constraints` — a library
`math_utils.py` exposes `clamp(value, low, high)` as a three-argument
function.  The agent must refactor it to `clamp(value, *, lo, hi)` (keyword-only
arguments), find its three callers in `calc.py`, `stats.py`, and
`ui.py`, update all three, and ensure `tests/test_stats.py::test_clamp_mean`
still passes (proving the test was updated, not deleted).

### 4. Hard-tier acceptance criteria

Every hard-tier task must satisfy:

- **Deterministic pre-condition** — the seeded workspace must produce a
  reproducible failure (exception, test failure, or non-zero exit code)
  before any agent work begins.
- **Deterministic post-condition** — the pass/fail outcome is determined
  by a programmatic check: `returncode == 0` from `python -m pytest`
  or an equivalent shell command.  There is no human judgement call.
- **No hint leakage** — the prompt does not name the specific files
  that need editing.  The agent must locate them via search, grep,
  or code comprehension.
- **Explicit `difficulty_tier="hard"`** — the task's `BenchmarkTask`
  entry sets `difficulty_tier="hard"` so it is correctly weighted in
  the improvement-rate KPI (PRD §5, ADR-0005 §Consequences).
- **`requires_skills` is non-empty** — hard tasks require at minimum
  `["bash"]` and likely additional skills (e.g. `grep_search`);
  the field documents what the Critic checks when the harness lacks
  a skill.

### 5. Update DifficultyTier literal

In `benchmarks/models.py` line 16, change:

```python
DifficultyTier = Literal["smoke", "easy", "medium"]
```

to:

```python
DifficultyTier = Literal["smoke", "easy", "medium", "hard"]
```

This is the only code change required.  The `field_validator` chains on
`BenchmarkTask` already handle `Literal` unions correctly; no new
validator is needed.

## Consequences

- **Positive**: The benchmark suite gains discriminative power.  A
  harness change that passes all smoke/easy/medium tasks but fails a
  hard task now surfaces a real capability gap rather than silently
  passing the gate.
- **Positive**: Hard tasks serve as leading indicators of harness
  brittleness — a hard task failure often precedes a regression in
  easier tasks, giving the Evolver an earlier signal.
- **Positive**: The `difficulty_tier` field on every task enables
  per-tier KPI slicing (e.g. "improvement rate for hard tasks only").
- **Negative**: Hard tasks increase benchmark runtime.  A timeout
  appropriate for a 30-second medium task may be insufficient for a
  complex hard task; each new task must be timed empirically.
- **Negative**: Hard tasks require more fixture data and fixture
  maintenance than smoke/easy tasks.  The `benchmarks/fixtures/hard/`
  directory should follow the same naming convention as existing
  fixtures.
- **Risk**: A hard task whose pre-condition is not reliably reproducible
  (e.g. a race condition in the seeded code) will produce noisy
  benchmark results.  Seeded workspaces must be deterministic; avoid
  randomness in fixture setup.
- **Risk**: Over time, tasks may drift in actual difficulty as the
  harness improves.  A task that was "hard" in 2026 may be "medium" in
  2027 as the agent gets better at multi-step reasoning.  The tier
  assignment should be reviewed annually.

## Implementation plan

1. Update `DifficultyTier` in `benchmarks/models.py` (this ADR).
2. Add `benchmarks/fixtures/hard/debug_import_cycle/` with the
   two-package import-cycle fixture.
3. Add `benchmarks/tasks/test_debug_import_cycle.py` implementing
   Archetype H1.
4. Add `benchmarks/fixtures/hard/refactor_api_with_constraints/` with
   the `math_utils` + callers fixture.
5. Add `benchmarks/tasks/test_refactor_api_with_constraints.py`
   implementing Archetype H2.
6. Run the full benchmark suite (`uv run pytest -m benchmark`) and
   record pass rates for both new tasks to calibrate `timeout_seconds`.
7. Update `CONTEXT.md` §DifficultyTier to document the four-tier scale.
