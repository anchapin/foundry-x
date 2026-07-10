"""Benchmark task: write a unit test for a given function."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_module

TASK = BenchmarkTask(
    name="write_unit_test",
    description="Author a passing pytest suite for a provided function.",
    prompt=(
        "The file target.py defines add(a, b). Write test_add.py with pytest "
        "cases that exercise add, then leave it in the workspace."
    ),
    tags=["testing"],
)

GOLDEN_TEST = """\
from target import add


def test_add_positive():
    assert add(2, 3) == 5


def test_add_zero():
    assert add(0, 0) == 0


def test_add_negative():
    assert add(-1, 1) == 0
"""


@pytest.mark.benchmark
def test_write_unit_test(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "target.py").write_text((fixture_dir / "target.py").read_text())
    (benchmark_workspace / "test_add.py").write_text(GOLDEN_TEST)

    result = run_module(benchmark_workspace, "pytest")

    assert result.returncode == 0, f"task {TASK.name}: tests failed\n{result.stdout}{result.stderr}"
    assert "3 passed" in result.stdout, f"task {TASK.name}: unexpected pytest summary"
