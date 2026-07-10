"""Benchmark task: fix a syntax error in a provided snippet."""

from __future__ import annotations

import py_compile
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="fix_syntax_error",
    description="Repair a syntax error in a Python snippet so it runs.",
    prompt=(
        "The file buggy.py contains a syntax error. Write solution.py with a "
        "corrected version that runs and prints the intended output."
    ),
    tags=["syntax", "debugging"],
)

GOLDEN_SOLUTION = """\
def greet(name):
    return "hello, " + name


print(greet("world"))
"""


@pytest.mark.benchmark
def test_fix_syntax_error(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    buggy_path = benchmark_workspace / "buggy.py"
    buggy_path.write_text((fixture_dir / "buggy.py").read_text())

    with pytest.raises(py_compile.PyCompileError):
        py_compile.compile(str(buggy_path), doraise=True)

    result = run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    expected = (fixture_dir / "expected_stdout.txt").read_text()
    assert result.stdout == expected, f"task {TASK.name}: stdout mismatch"
