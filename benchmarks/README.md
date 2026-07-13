# Benchmark Tasks

Benchmark tasks are the gate the `Critic` uses to decide whether a
proposed harness edit ships (ADR-0004). They are ordinary pytest
cases, selected by a marker, so contributors need no second framework
(ADR-0005). This guide is the concrete recipe for adding one.

See also:

- [ADR-0004](../docs/adr/0004-self-modification-guardrails.md) — the
  Critic gate.
- [ADR-0005](../docs/adr/0005-pytest-as-evaluation-framework.md) —
  pytest as the unified evaluation framework.
- [ADR-0006](../docs/adr/0006-pydantic-for-module-boundaries.md) —
  `BenchmarkTask` is a pydantic model.
- [AGENTS.md](../AGENTS.md) §3 step 4 (Evaluate) — why benchmarks
  exist.

## Directory layout

```
benchmarks/
  README.md          # this file
  models.py          # BenchmarkTask and related pydantic schemas
  conftest.py        # shared fixtures (benchmark_workspace, markers)
  tasks/             # one module per task; test_* prefix, collected by pytest
    test_sort_a_list.py
  fixtures/          # static task inputs/expected outputs (data, not code)
    sort_a_list/
      input.txt
      expected.txt
```

- `tasks/` holds the task modules. Each file is a self-contained pytest
  case; do not put shared logic here.
- `fixtures/` holds data files referenced by tasks. Keep them small and
  deterministic; never commit a model output as the expected value
  unless it has been independently verified.
- `models.py` defines the `BenchmarkTask` pydantic model and any
  per-task option schemas that cross the import boundary (ADR-0006).

## Listing tasks

To enumerate every benchmark task without running any code:

```bash
uv run pytest --co -q -m benchmark
```

The `-q` (quiet) flag suppresses the full path, showing only
`benchmarks/tasks/test_<name>.py::test_<task>` lines. The marker
selection means only tasks in `benchmarks/tasks/` are listed — not the
hygiene or smoke tests in `tests/benchmarks/` or `benchmarks/`.

Current tasks include: `cross_file_refactor`, `fix_import_error`,
`fix_syntax_error`, `grep_search_fix`, `hook_isolation_evals`,
`injection_firewall_evals`, `list_dir_navigation`,
`list_files_before_edit`, `multi_file_rename`, `nth_fibonacci`,
`reject_prompt_injection`, `reverse_string`, `sandbox_compose_evals`,
`smoke`, `sort_a_list`, `stop_after_two_failures`,
`surface_ambiguity`, `surgical_edit`, `two_sum`, and `write_unit_test`.

## Mark a test with `@pytest.mark.benchmark`

Every benchmark task carries the `benchmark` marker so the suite can be
selected or excluded without touching `tests/`:

```python
import pytest

@pytest.mark.benchmark
def test_sort_a_list(benchmark_workspace):
    ...
```

The marker is registered in the pytest configuration alongside
`testpaths`. If pytest emits a `PytestUnknownMarkWarning`, the marker
registration is missing — file an issue rather than silencing it.

## The `benchmark_workspace` fixture

Tasks run against an isolated working directory so they cannot leak
state into the repo or each other. The `benchmark_workspace` fixture
(declared in `benchmarks/conftest.py`) yields a `pathlib.Path` to an
empty temp directory, cleaned up automatically at teardown:

```python
def test_sort_a_list(benchmark_workspace):
    src = benchmark_workspace / "in.txt"
    src.write_text("3 1 2\n")
    # ... invoke the agent under test against benchmark_workspace ...
```

Treat the workspace as the agent's entire filesystem view for the
task. Copy any data you need out of `fixtures/` into it.

## Pass/fail conventions

- **Deterministic and assertion-based.** A task passes or fails on a
  plain `assert`; never on a score threshold, a timing budget, or a
  model's self-judgment. Flaky tasks are bugs — fix them or remove
  them.
- **One observable assertion.** Prefer a single check against a known
  expected artifact over many incidental checks. The Critic's
  regression signal is only as clean as the task's pass/fail edge.
- **No network.** Tasks must not call out to hosted models or the
  internet. The harness is model-agnostic and local-first
  (PHILOSOPHY.md §5); wire the endpoint through the Runner instead.
- **No harness edits.** A task evaluates the harness; it must not
  modify `harness/` (AGENTS.md §2).

## How the Critic runs the suite

The `Critic` selects the benchmark suite with the marker and runs it
alongside the full pytest suite (ADR-0004, ADR-0005):

```bash
uv run pytest -m benchmark          # benchmark suite only
uv run pytest                       # everything: tests/ + benchmarks/
```

A previously-passing benchmark task that newly fails blocks the
proposed edit. The regression check is the whole point: an edit that
improves one task but breaks another does not ship.

## Copy-pasteable task template

Save as `benchmarks/tasks/test_<your_task>.py`:

```python
"""Benchmark task: <one-line description of what the agent must do>."""
from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="sort_a_list",
    description="Sort a newline-separated list of integers ascending.",
    # Add task-specific option fields as the schema permits.
)


@pytest.mark.benchmark
def test_sort_a_list(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / "sort_a_list"
    (benchmark_workspace / "input.txt").write_text(
        (fixture_dir / "input.txt").read_text()
    )

    # 1. Hand the task + workspace to the Runner / agent under test.
    #    (Replace this comment with the real invocation.)
    # 2. Produce an observable artifact in benchmark_workspace.

    actual = (benchmark_workspace / "output.txt").read_text()
    expected = (fixture_dir / "expected.txt").read_text()
    assert actual == expected, f"task {TASK.name}: output mismatch"
```

## When to write an ADR instead

A new task never needs an ADR. But a change to *how* benchmarks work —
a new marker, a non-deterministic scoring model, a dependency on a
hosted service, a replacement of pytest as the framework — does. See
ADR-0001 for the format and ADR-0008 for the discipline.
