"""Benchmark task: reverse a string.

INFRASTRUCTURE CHECK (issue #1120)
==================================
This task uses run_solution to plant a complete golden solution. It tests
whether the execution infrastructure works correctly -- NOT whether the
agent can independently synthesize a string-reversal algorithm.

The golden-solution approach provides a deterministic pass/fail baseline
that locks the infrastructure green before the agent loop is wired.
"""

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
    difficulty_tier="easy",
    tags=["strings", "infrastructure"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    text = Path("input.txt").read_text().rstrip("\\n")
    Path("output.txt").write_text(text[::-1] + "\\n")


if __name__ == "__main__":
    main()
"""


_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_reverse_string(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #112)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
