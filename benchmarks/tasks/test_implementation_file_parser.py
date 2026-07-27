"""Benchmark task: aggregate key-value pairs across files (implementation, io, multi-file).

Parses every ``*.txt`` data file in the workspace, aggregating
``name:score`` lines into per-name totals, and writes the result sorted by
name. This is the suite's first ``implementation`` task that genuinely
exercises the ``multi-file`` surface: the agent must discover and read an
arbitrary set of input files rather than a single ``input.txt``.

Each fixture case directory contains one or more ``data*.txt`` files plus
an ``expected.txt``. The test copies every file except ``expected.txt``
into the workspace before running the golden solution.

I/O contract:

- The workspace contains one or more ``data*.txt`` files, each with lines
  of the form ``name:score`` (blank lines ignored).
- ``output.txt`` contains one ``name total`` line per unique name, sorted
  alphabetically. An empty file when there is no data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="implementation_file_parser",
    description=(
        "Read name:score pairs from every data*.txt file in the workspace, "
        "sum scores per name, and write 'name total' lines sorted by name "
        "to output.txt."
    ),
    prompt=(
        "The workspace contains one or more files named data*.txt. Each "
        "line in those files has the form 'name:score' where score is an "
        "integer (blank lines are ignored). Sum the scores for each name "
        "across all data files. Write one line 'name total' per unique "
        "name to output.txt, sorted alphabetically by name. Write an "
        "empty output.txt when there is no data."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "output.txt contains one 'name total' line per unique name in "
        "alphabetical order, or is empty when no data files contain "
        "name:score lines."
    ),
    tags=["implementation", "io", "multi-file"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    totals: dict[str, int] = {}
    for p in sorted(Path(".").glob("*.txt")):
        if p.name == "output.txt":
            continue
        for line in p.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            name, _, score = line.partition(":")
            totals[name] = totals.get(name, 0) + int(score)
    lines = [f"{name} {totals[name]}" for name in sorted(totals)]
    Path("output.txt").write_text("\\n".join(lines) + ("\\n" if lines else ""))


if __name__ == "__main__":
    main()
"""

_CASES = sorted(
    p.name for p in (Path(__file__).parent.parent / "fixtures" / TASK.name).iterdir() if p.is_dir()
)


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.benchmark
def test_implementation_file_parser(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for TASK across edge-case fixtures (issue #1052)."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case
    for src in sorted(fixture_dir.iterdir()):
        if src.name == "expected.txt":
            continue
        (benchmark_workspace / src.name).write_text(src.read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    actual = (benchmark_workspace / "output.txt").read_text().rstrip("\n")
    expected = (fixture_dir / "expected.txt").read_text().rstrip("\n")
    assert actual == expected, f"task {TASK.name}/{case}: output mismatch"
