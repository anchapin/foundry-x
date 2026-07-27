"""Benchmark task: multiplication table via list comprehension (implementation, comprehension).

INFRASTRUCTURE CHECK (issue #1120)
==================================
This task uses run_solution to plant a complete golden solution. It tests
whether the execution infrastructure works correctly -- NOT whether the
agent can independently synthesize a multiplication-table algorithm.

Builds an ``n x n`` multiplication table and flattens it row-major into a
single line. The golden solution is a genuine double list comprehension --
``[(i + 1) * (j + 1) for i in range(n) for j in range(n)]`` -- making the
``comprehension`` tag substantive rather than ornamental. The boundary
handling (``n == 0`` yields an empty result) probes edge-case reasoning
that goes beyond the ``easy`` transforms.

I/O contract:

- ``input.txt`` holds a single integer ``n``.
- ``output.txt`` contains the flattened ``n x n`` table
  (``entry[i][j] == (i+1)*(j+1)``), space-separated on one line.
  An empty file when ``n == 0``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="implementation_list_comprehension",
    description=(
        "Read integer n from input.txt; build an n-by-n multiplication "
        "table flattened row-major and write it space-separated to "
        "output.txt; empty when n is 0."
    ),
    prompt=(
        "input.txt contains a single integer n. Build an n-by-n "
        "multiplication table where entry (i, j) equals (i+1)*(j+1) for "
        "0-based i, j in range(n). Flatten the table row by row and write "
        "the values space-separated on a single line to output.txt. Write "
        "an empty output.txt when n is 0."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "output.txt contains the n*n space-separated products in row-major "
        "order, or is empty when n is 0."
    ),
    tags=["implementation", "comprehension", "infrastructure"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    n = int(Path("input.txt").read_text().strip())
    table = [(i + 1) * (j + 1) for i in range(n) for j in range(n)]
    Path("output.txt").write_text(" ".join(map(str, table)) + ("\\n" if table else ""))


if __name__ == "__main__":
    main()
"""

_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_implementation_list_comprehension(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #1052)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
