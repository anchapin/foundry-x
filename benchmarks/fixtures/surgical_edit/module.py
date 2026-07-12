"""Multi-function module for the surgical_edit benchmark (seeded/broken).

Four independent helper functions live here. Exactly one -- ``median``
-- is buggy: it indexes two past the middle of the sorted list, so
calling it on any non-empty input raises ``IndexError`` and the
``__main__`` block aborts with a non-zero exit code. The other three
(``square``, ``is_even``, ``average``) are correct and MUST remain
byte-identical after a surgical fix.

This file is the *seeded* (broken) state. The golden targeted edit
lives in ``benchmarks/tasks/test_surgical_edit.py`` (``GOLDEN_EDIT_*``).
"""


def square(n: int) -> int:
    """Return the square of *n*."""
    return n * n


def is_even(n: int) -> bool:
    """Return True if *n* is even."""
    return n % 2 == 0


def average(nums: list[int]) -> float:
    """Return the arithmetic mean of *nums* (empty list -> 0.0)."""
    if not nums:
        return 0.0
    return sum(nums) / len(nums)


def median(nums: list[int]) -> float:
    """Return the median of a sorted copy of *nums*."""
    ordered = sorted(nums)
    mid = len(ordered) // 2
    # BUG: off-by-two index -> IndexError on any non-empty input.
    return ordered[mid + 2]


if __name__ == "__main__":
    print(square(5))
    print(is_even(4))
    print(average([1, 2, 3]))
    print(median([1, 2, 3]))
