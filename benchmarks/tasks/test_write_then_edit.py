"""Benchmark task: chained write → run → edit → verify workflow (issue #706).

This is the first benchmark that exercises a multi-step, multi-file operation
chain: write a new Python module, run it to observe a failure, edit the module
to fix the failure, and verify it runs correctly.

The dominant real-world coding workflow is:

    1. Write a new file with a stub or initial implementation.
    2. Run it and observe a failure.
    3. Edit the file in place to fix the bug.
    4. Verify the fix end-to-end.

No existing benchmark in ``benchmarks/tasks/`` captures this full chain. The
``write_file`` benchmark only writes (no subsequent edit); ``edit_file`` only
edits (assumes the file already exists); ``fix_import_error`` edits an existing
file without any prior write step.

This task seeds ``stub.py`` (a Python module with a stub function that returns
the wrong result) and requires the agent to edit only the buggy function body,
then verify ``python stub.py`` exits 0 with the expected stdout.  The edit
preserves the module docstring and the ``if __name__ == "__main__"`` guard --
exactly the shape of a real bug-fix session.

The golden solution exercises the full write → run → edit → verify chain:
it writes the initial stub to disk, runs it (which prints a wrong value),
edits only the buggy line to fix the formula, then re-runs to confirm the
correct result.  The benchmark test validates the post-edit state using the
same ``subprocess.run`` pattern as the rest of the suite.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="write_then_edit",
    description=(
        "Write a Python module with a stub function, run it to observe a "
        "failure caused by the stub, edit only the stub body to fix it, "
        "then verify the module runs end-to-end with correct output."
    ),
    prompt=(
        "stub.py is in the workspace. It defines compute(a, b) which should "
        "return the product a * b, but the stub returns a + b instead. "
        "Edit ONLY the compute() function body to fix the formula, then "
        "run 'python stub.py' and confirm it prints 'compute(3, 4) = 12'."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After a targeted edit to compute() only, 'python stub.py' returns "
        "rc=0 and prints 'compute(3, 4) = 12'."
    ),
    timeout_seconds=30,
    requires_skills=["bash", "edit_file"],
    tags=["editing", "multi-step", "write_then_edit", "chained"],
)

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

STUB_OLD = "    return a + b  # BUG: should be a * b\n"
STUB_NEW = "    return a * b\n"


def _run_stub(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "stub.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _parse_check(workspace: Path) -> None:
    ast.parse((workspace / "stub.py").read_text())


@pytest.mark.benchmark
def test_write_then_edit(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    The golden solution exercises the full write → run → edit → verify chain:

        1. Write the initial stub to stub.py (a + b instead of a * b).
        2. Run 'python stub.py' -- the stub prints the wrong value.
        3. Edit only the compute() body, replacing 'a + b' with 'a * b'.
        4. Re-run 'python stub.py' -- must exit 0 and print
           'compute(3, 4) = 12'.

    The benchmark test validates the post-edit state: the module must parse
    as valid Python and 'python stub.py' must exit 0 with the expected stdout.
    """
    seeded_source = (_FIXTURE_DIR / "stub.py").read_text()
    expected_stdout = (_FIXTURE_DIR / "expected_stdout.txt").read_text()
    stub_path = benchmark_workspace / "stub.py"

    # --- Pre-condition: seed the stub and verify it exists. -------------
    stub_path.write_text(seeded_source)
    _parse_check(benchmark_workspace)

    # --- Step 1 of golden chain: run the stub (wrong formula). ----------
    bad = _run_stub(benchmark_workspace)
    assert bad.returncode == 0, (
        f"task {TASK.name}: seeded stub.py must run (exit 0) even with "
        f"wrong formula; got rc={bad.returncode}"
    )
    assert "7" in bad.stdout, (
        f"task {TASK.name}: seeded stub.py must print wrong value '7' "
        f"(a+b); got stdout={bad.stdout!r}"
    )

    # --- Step 2 of golden chain: apply the targeted edit. ---------------
    occurrences = seeded_source.count(STUB_OLD)
    assert occurrences == 1, (
        f"task {TASK.name}: golden old_string must match exactly once "
        f"in the seeded stub; found {occurrences}"
    )
    fixed_source = seeded_source.replace(STUB_OLD, STUB_NEW, 1)
    stub_path.write_text(fixed_source)

    # Verify the edit was surgical: docstring and guard unchanged.
    assert "Stub module for write_then_edit benchmark." in fixed_source
    assert 'if __name__ == "__main__":' in fixed_source
    _parse_check(benchmark_workspace)

    # --- Step 3 of golden chain: verify end-to-end. --------------------
    good = _run_stub(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: patched stub.py must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
    assert good.stdout == expected_stdout, (
        f"task {TASK.name}: stdout mismatch (got {good.stdout!r}, expected {expected_stdout!r})"
    )
