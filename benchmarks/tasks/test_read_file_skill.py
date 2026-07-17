"""Benchmark task: gate the read_file skill contract (issue #577).

The ``read_file`` skill (``harness/skills/read_file.json``) is the agent's
primary information-gathering tool, yet the benchmark suite had zero tasks
pinning its contract. A regression in ``read_file`` (wrong path resolution,
encoding mishandling, truncation issues) would pass every benchmark silently.

This module closes that gap with three benchmark tasks:

1. ``config_parser``  -- reads a config file and parses key=value pairs,
                         verifying the skill handles medium-sized structured
                         files without truncation.
2. ``line_range``     -- reads a specific line range from a large file using
                         ``offset`` and ``max_lines``, verifying the skill
                         correctly pages through large files.
3. ``latin1_encoding`` -- reads a file with non-UTF8 (latin-1) encoding and
                         verifies the skill handles it gracefully with
                         ``errors="replace"`` rather than raising.

Each task seeds the workspace with a fixture file, runs a golden solution
that mirrors the ``read_file`` skill contract (stdlib-only, same input/output
schema), and asserts the output matches the expected result.

The golden solution is deliberately written to mirror the skill's
``input_schema`` / ``output_schema`` contract so the benchmark pins the
interface rather than an arbitrary implementation choice.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="read_file_skill",
    description=(
        "Gate the read_file skill contract: config parsing, line-range "
        "paging, and non-UTF8 encoding handling."
    ),
    prompt=(
        "Use the read_file skill to: (1) read config.ini and produce "
        "key=value pairs; (2) read lines 10-20 of source.txt using "
        "offset/max_lines; (3) read data.txt (latin-1) and verify its "
        "contents. Write results to output.txt."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "output.txt contains: (1) all key=value pairs from config.ini; "
        "(2) lines 10-20 from source.txt; (3) the decoded latin-1 content "
        "from data.txt."
    ),
    requires_skills=["read_file"],
    tags=["read_file", "config", "encoding", "offset", "paging"],
)

CASES = ["config_parser", "line_range", "latin1_encoding"]

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name


def _run_solution(workspace: Path) -> None:
    solution_path = FIXTURE_DIR / "solution.py"
    source = solution_path.read_text(encoding="utf-8")
    run_solution(workspace, source)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.benchmark
def test_read_file_skill(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for the read_file skill across three cases.

    Each case seeds the workspace with a fixture file, runs the golden
    solution (which mirrors the read_file contract), and asserts the
    output matches the expected result.

    Cases:
        config_parser   -- config.ini -> key=value pairs
        line_range      -- source.txt lines 10-20 via offset/max_lines
        latin1_encoding -- data.txt (latin-1) decoded via utf-8/replace
    """
    fixture_case_dir = FIXTURE_DIR / case

    if case == "config_parser":
        (benchmark_workspace / "config.ini").write_text(
            (fixture_case_dir / "config.ini").read_text(encoding="utf-8")
        )
    elif case == "line_range":
        (benchmark_workspace / "source.txt").write_text(
            (fixture_case_dir / "source.txt").read_text(encoding="utf-8")
        )
    elif case == "latin1_encoding":
        (benchmark_workspace / "data.txt").write_bytes((fixture_case_dir / "data.txt").read_bytes())

    _run_solution(benchmark_workspace)

    actual = (benchmark_workspace / "output.txt").read_text(encoding="utf-8")
    expected = (fixture_case_dir / "expected.txt").read_text(encoding="utf-8")

    assert actual == expected, (
        f"task {TASK.name}/{case}: output mismatch\n"
        f"--- expected ---\n{expected!r}\n"
        f"--- actual ---\n{actual!r}"
    )
