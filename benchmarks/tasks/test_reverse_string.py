"""Benchmark task: reverse a string."""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="reverse_string",
    description="Reverse the characters of the input string.",
    prompt=(
        "Read a single line from input.txt, reverse its characters, and write "
        "the result to output.txt."
    ),
    tags=["strings"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    text = Path("input.txt").read_text().rstrip("\\n")
    Path("output.txt").write_text(text[::-1] + "\\n")


if __name__ == "__main__":
    main()
"""


@pytest.mark.benchmark
def test_reverse_string(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text()
    expected = (fixture_dir / "expected.txt").read_text()
    assert actual == expected, f"task {TASK.name}: output mismatch"
