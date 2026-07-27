"""Benchmark task: sort a list of integers ascending.

INFRASTRUCTURE CHECK (issue #1120)
==================================
This task uses run_solution to plant a complete golden solution. It tests
whether the execution infrastructure (workspace staging, script execution,
file I/O) works correctly -- NOT whether the agent can independently
synthesize a sorting algorithm.

The golden-solution approach provides a deterministic pass/fail baseline
that locks the infrastructure green before the agent loop is wired.

Non-golden variant: test_sort_a_list__agent_output (below) validates
agent-produced output against the same I/O contract and fixture set.
"""

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
    difficulty_tier="easy",
    tags=["sorting", "io", "infrastructure"],
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


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_sort_a_list__agent_output(benchmark_workspace: Path, case: str) -> None:
    """Non-golden agent-capability variant (issue #1120).

    Unlike test_sort_a_list which plants GOLDEN_SOLUTION via run_solution,
    this variant seeds ONLY the input and asserts on agent-produced output.
    It requires the Runner to produce a solution.py in the workspace before
    the Critic evaluates it.

    The I/O contract and fixture set are identical; only the planted-solution
    path is different.
    """
    import subprocess
    import sys

    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    solution_path = benchmark_workspace / "solution.py"
    if not solution_path.exists():
        pytest.skip("no agent-produced solution.py found -- run the full agent loop first")

    result = subprocess.run(
        [sys.executable, "solution.py"],
        cwd=benchmark_workspace,
        capture_output=True,
        text=True,
        check=False,
    )

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert result.returncode == 0, f"task {TASK.name}/{case}: solution.py exited {result.returncode}: {result.stderr}"
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
