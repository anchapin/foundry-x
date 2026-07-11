"""Benchmark task: two-sum (find two distinct indices whose values sum to target).

Medium-tier algorithmic benchmark (issue #179). Exercises the smallest
non-trivial algorithmic shape beyond the O(n) linear transforms that
dominate the deterministic suite (``nth_fibonacci``, ``reverse_string``,
``sort_a_list``): a single-pass hash-map lookup. The complement check
``target - value`` against a running ``value -> index`` map is a real
reasoning step that probes a single-file solve beyond pure transformation.

The five seeded edge cases cover the standard two-sum surface:

- ``basic``          -- positive values, one pair, canonical 9 = 2 + 7.
- ``no_pair``        -- positive values, no pair sums to the target.
- ``single_element`` -- one element, no pair possible (only one index).
- ``negatives``      -- negative target and negative values exercise the
                        sign-handling branch of the hash-map lookup.
- ``duplicates``     -- duplicate values exercise the map key-collision
                        path (first occurrence stored, second looked up).

I/O contract (mirrors ``sort_a_list`` / ``reverse_string``):

- ``input.txt`` has two lines. Line 1 is the target integer. Line 2 is a
  space-separated list of integers.
- ``output.txt`` contains either ``"<i> <j>\\n"`` with ``i < j`` (sorted
  ascending) when a pair exists, or an empty file when no pair sums to
  the target.

The golden solution is a textbook one-pass hash-map scan; it returns
``None`` when no pair exists and sorts the returned indices before
printing so the post-condition is canonical regardless of discovery order.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="two_sum",
    description=(
        "Read a target integer and a space-separated list of integers from "
        "input.txt; find two distinct indices whose values sum to the target "
        "and write them space-separated (sorted ascending) to output.txt. "
        "Write an empty output.txt if no such pair exists."
    ),
    prompt=(
        "input.txt has two lines. Line 1 is an integer target. "
        "Line 2 is a space-separated list of integers. Find two distinct "
        "indices i, j with i < j such that nums[i] + nums[j] == target. "
        "Write 'i j' (space-separated, ascending order) on a single line to "
        "output.txt. If no such pair exists, write an empty output.txt."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "output.txt contains either '<i> <j>\\n' with i < j on a single line "
        "when a pair sums to the target, or is empty when no pair exists."
    ),
    tags=["algorithms", "hashing"],
)

GOLDEN_SOLUTION = """\
from __future__ import annotations

from pathlib import Path


def two_sum(nums: list[int], target: int) -> tuple[int, int] | None:
    seen: dict[int, int] = {}
    for j, value in enumerate(nums):
        complement = target - value
        if complement in seen:
            i = seen[complement]
            return (min(i, j), max(i, j))
        seen[value] = j
    return None


def main() -> None:
    lines = Path("input.txt").read_text().splitlines()
    target = int(lines[0].strip())
    nums = [int(x) for x in lines[1].split()] if len(lines) > 1 else []

    pair = two_sum(nums, target)
    if pair is None:
        Path("output.txt").write_text("")
    else:
        i, j = pair
        Path("output.txt").write_text(f"{i} {j}\\n")


if __name__ == "__main__":
    main()
"""


_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_two_sum(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #179)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
