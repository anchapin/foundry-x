"""Benchmark task: sort a list of integers ascending."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="sort_a_list",
    description="Sort a space-separated list of integers into ascending order.",
    prompt=(
        "Read space-separated integers from input.txt, sort them ascending, "
        "and write the result space-separated to output.txt."
    ),
    tags=["sorting", "io"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    nums = sorted(int(x) for x in Path("input.txt").read_text().split())
    Path("output.txt").write_text(" ".join(map(str, nums)) + "\\n")


if __name__ == "__main__":
    main()
"""


_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_sort_a_list(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #112)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
