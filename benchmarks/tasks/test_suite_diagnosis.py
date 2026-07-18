"""Benchmark task: diagnose and fix a failing pytest suite (issue #808).

The benchmark suite covers fix_import_error, stop_after_two_failures, and
surgical_edit, but no task exercises the "read pytest output -> identify
which test fails and why -> fix the root cause" loop. This is the single
most common developer workflow and is entirely absent from the skill coverage
matrix.

The fixture seeds a multi-file Python package (``pkg/``) with a test suite
(``tests/test_math.py``) where exactly one test asserts an incorrect expected
value. The golden solution must:

    1. Run pytest and capture stderr to identify the failing test.
    2. Pinpoint the incorrect assertion in tests/test_math.py.
    3. Edit only that assertion line to use the correct expected value.
    4. Re-run pytest and exit 0.

This exercises the read-then-edit-then-verify loop that is the dominant
real-world developer workflow, giving the benchmark suite credible coverage
of test-suite comprehension. See ADR-0005 and ADR-0010.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="test_suite_diagnosis",
    description=(
        "Diagnose a pytest suite that fails due to an incorrect assertion "
        "in tests/test_math.py, identify the wrong expected value, fix it, "
        "and verify pytest passes."
    ),
    prompt=(
        "The workspace contains a Python package 'pkg/' with a test suite "
        "'tests/test_math.py'. Running 'pytest' exits non-zero because one "
        "test asserts an incorrect expected value. Read the pytest output to "
        "identify which test fails and why, edit only the incorrect assertion "
        "line in tests/test_math.py, then re-run pytest to confirm it passes. "
        "Do not modify any other files or code."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After identifying and fixing the incorrect assertion, 'pytest' "
        "exits 0 with all tests passing."
    ),
    timeout_seconds=30,
    requires_skills=["bash", "edit_file"],
    tags=["pytest", "debugging", "diagnosis", "multi-file"],
)

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name


def _copy_fixture_to_workspace(fixture_dir: Path, workspace: Path) -> None:
    """Copy the fixture pkg/ and tests/ directories into workspace."""
    skip_dirs = {"__pycache__", ".git", ".venv", "node_modules"}
    for subdir in ("pkg", "tests"):
        src = fixture_dir / subdir
        dst = workspace / subdir
        for file in src.rglob("*"):
            if file.is_file() and not any(part in skip_dirs for part in file.parts):
                rel = file.relative_to(src)
                dst_file = dst / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(file.read_text())


def _clear_pycache(workspace: Path) -> None:
    """Strip ``__pycache__`` directories from a workspace."""
    import shutil

    for cache in workspace.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)


def _run_pytest(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


@pytest.mark.benchmark
def test_test_suite_diagnosis(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Pre-condition: the seeded suite genuinely fails with AssertionError.
    Post-condition: after fixing the incorrect assertion, pytest passes.
    """
    _copy_fixture_to_workspace(FIXTURE_DIR, benchmark_workspace)

    test_file = benchmark_workspace / "tests" / "test_math.py"
    seeded_test = test_file.read_text()

    assert "== 5" in seeded_test, (
        f"task {TASK.name}: fixture must contain the broken assertion (== 5)"
    )

    bad = _run_pytest(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded suite must fail before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    combined_output = bad.stdout + bad.stderr
    assert "AssertionError" in combined_output, (
        f"task {TASK.name}: expected AssertionError in pytest output; "
        f"got stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )

    fixed_test = seeded_test.replace("== 5", "== 4", 1)
    assert "== 5" not in fixed_test, (
        f"task {TASK.name}: golden fix must remove the broken assertion"
    )
    test_file.write_text(fixed_test)
    _clear_pycache(benchmark_workspace)

    good = _run_pytest(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected suite must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
