"""Benchmark task: return the nth Fibonacci number."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="nth_fibonacci",
    description="Return the nth Fibonacci number (F(0)=0, F(1)=1).",
    prompt=("Read integer n from input.txt and write F(n) to output.txt, where F(0)=0 and F(1)=1."),
    tags=["math", "recurrence"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def fib(n: int) -> int:
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def main() -> None:
    n = int(Path("input.txt").read_text().strip())
    Path("output.txt").write_text(f"{fib(n)}\\n")


if __name__ == "__main__":
    main()
"""


@pytest.mark.benchmark
def test_nth_fibonacci(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text()
    expected = (fixture_dir / "expected.txt").read_text()
    assert actual == expected, f"task {TASK.name}: output mismatch"
