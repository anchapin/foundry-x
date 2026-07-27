"""Benchmark task: binary search (implementation, easy).

Exercises a single non-trivial algorithmic shape: the O(log n) halving
loop with index arithmetic and a termination condition. The complement
``lo <= hi`` boundary and the empty-list / not-found edges probe
single-function synthesis beyond pure transformation.

I/O contract:

- ``input.txt`` has two lines. Line 1 is the integer ``target``. Line 2
  is a space-separated list of integers sorted ascending.
- ``output.txt`` contains the 0-based index of ``target`` in the list,
  or ``-1`` when absent (including an empty list).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="implementation_binary_search",
    description=(
        "Read a target integer and a sorted list from input.txt; binary "
        "search for the target and write its 0-based index (or -1) to "
        "output.txt."
    ),
    prompt=(
        "input.txt has two lines. Line 1 is an integer target. Line 2 is a "
        "space-separated list of integers sorted ascending. Use binary "
        "search to find the target. Write its 0-based index to output.txt, "
        "or -1 if it is not present."
    ),
    difficulty_tier="easy",
    expected_outcome=(
        "output.txt contains the 0-based index of target in the list, or "
        "'-1' when the target is absent or the list is empty."
    ),
    tags=["implementation"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def binary_search(arr: list[int], target: int) -> int:
    lo, hi = 0, len(arr) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1


def main() -> None:
    lines = Path("input.txt").read_text().splitlines()
    target = int(lines[0].strip())
    arr = [int(x) for x in lines[1].split()] if len(lines) > 1 else []
    Path("output.txt").write_text(f"{binary_search(arr, target)}\\n")


if __name__ == "__main__":
    main()
"""

_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_implementation_binary_search(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #1052)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
