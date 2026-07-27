"""Benchmark task: FizzBuzz (implementation, smoke).

INFRASTRUCTURE CHECK (issue #1120)
==================================
This task uses run_solution to plant a complete golden solution. It tests
whether the execution infrastructure works correctly -- NOT whether the
agent can independently synthesize a FizzBuzz implementation.

The canonical single-function synthesis task. Exercises the agent's ability
to map a divisibility spec to conditional branching -- the smallest
reasoning step beyond pure transformation. Covers the classic 3/5/15
mutual-exclusion surface that is HumanEval+'s ``fizz_buzz`` competency.

I/O contract:

- ``input.txt`` holds a single integer ``n``.
- ``output.txt`` contains the FizzBuzz sequence for ``1..n``,
  space-separated on a single line. An empty file when ``n == 0``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="implementation_fizzbuzz",
    description=(
        "Read integer n from input.txt and write the FizzBuzz sequence for "
        "1..n (space-separated) to output.txt; empty when n is 0."
    ),
    prompt=(
        "input.txt contains a single integer n. For each i from 1 to n: "
        "output 'FizzBuzz' if divisible by 15, 'Fizz' if divisible by 3, "
        "'Buzz' if divisible by 5, otherwise str(i). Write the tokens "
        "space-separated on one line to output.txt. Write an empty "
        "output.txt when n is 0."
    ),
    difficulty_tier="easy",
    expected_outcome=(
        "output.txt contains the space-separated FizzBuzz tokens for 1..n, or is empty when n is 0."
    ),
    tags=["implementation", "infrastructure"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def fizzbuzz(n: int) -> list[str]:
    out: list[str] = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            out.append("FizzBuzz")
        elif i % 3 == 0:
            out.append("Fizz")
        elif i % 5 == 0:
            out.append("Buzz")
        else:
            out.append(str(i))
    return out


def main() -> None:
    n = int(Path("input.txt").read_text().strip())
    tokens = fizzbuzz(n)
    Path("output.txt").write_text(" ".join(tokens) + ("\\n" if tokens else ""))


if __name__ == "__main__":
    main()
"""

_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_implementation_fizzbuzz(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #1052)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
