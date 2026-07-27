"""Benchmark task: string operations (implementation, string-manipulation).

A Caesar-shift-by-one cipher: each alphabetic character advances one
position in the alphabet with wrap-around (``z`` -> ``a``, ``Z`` -> ``A``);
non-alphabetic characters are unchanged. Exercises character-class
branching, modular arithmetic over ASCII offsets, and case preservation --
the core of single-function string manipulation.

I/O contract:

- ``input.txt`` holds a single line of text.
- ``output.txt`` contains the shifted string. An empty file when the
  input line is empty.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="implementation_string_ops",
    description=(
        "Read a line from input.txt, apply a Caesar +1 shift to every "
        "alphabetic character (wrapping z->a, Z->A, non-alpha unchanged), "
        "and write the result to output.txt."
    ),
    prompt=(
        "input.txt contains a single line of text. Shift every alphabetic "
        "character forward by one position in the alphabet, wrapping so "
        "that 'z' becomes 'a' and 'Z' becomes 'A'. Leave digits, spaces, "
        "punctuation, and any other non-letter characters unchanged. Write "
        "the shifted string to output.txt. Write an empty output.txt when "
        "the input line is empty."
    ),
    difficulty_tier="easy",
    expected_outcome=(
        "output.txt contains the Caesar-shifted string (each letter "
        "advanced by one with wrap-around), or is empty for empty input."
    ),
    tags=["implementation", "string-manipulation"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def shift_char(c: str) -> str:
    if "a" <= c <= "z":
        return chr((ord(c) - ord("a") + 1) % 26 + ord("a"))
    if "A" <= c <= "Z":
        return chr((ord(c) - ord("A") + 1) % 26 + ord("A"))
    return c


def caesar_shift(text: str) -> str:
    return "".join(shift_char(c) for c in text)


def main() -> None:
    text = Path("input.txt").read_text().rstrip("\\n")
    shifted = caesar_shift(text)
    Path("output.txt").write_text(shifted + ("\\n" if shifted else ""))


if __name__ == "__main__":
    main()
"""

_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_implementation_string_ops(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #1052)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    (benchmark_workspace / "input.txt").write_text((fixture_dir / "input.txt").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
