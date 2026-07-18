"""Benchmark task: gate the read_multiple_files skill contract (issue #873).

The ``read_multiple_files`` skill (``harness/skills/read_multiple_files.json``)
lets the agent read N files in a single batched call. The benchmark suite had
no task pinning its contract; issue #617 / PR #688 explicitly scoped the
multi-file case out when adding the singular ``read_file`` benchmark. A
regression in batch handling, per-file error semantics, or the aggregate
top-level ``truncated`` flag would pass every existing benchmark silently.

This module closes that gap with three benchmark tasks:

1. ``multi_file_basic``       -- reads three files of mixed content types
                                 (Python / JSON / Markdown) in a single batch
                                 and verifies the per-file result entries
                                 preserve order and content.
2. ``missing_file_in_batch``  -- one of the listed paths does not exist;
                                 verifies the call itself succeeds (top-level
                                 ``error == None``) while the missing entry
                                 carries the per-file ``error`` and the
                                 other entries still resolve correctly.
3. ``truncation_aggregate``   -- one small file plus one file large enough
                                 to exceed the per-file ``max_bytes`` cap;
                                 verifies the large entry is truncated, the
                                 small entry is not, and the top-level
                                 ``truncated`` flag is true (since the
                                 aggregate is the OR over all entries).

Each task seeds the workspace with the fixture files it expects to exist,
runs a golden solution that mirrors the ``read_multiple_files`` skill
contract (stdlib-only, same input/output schema), and asserts the output
matches the expected result. The golden solution deliberately pins the
interface (per-file result shape, per-file ``error`` location, aggregate
``truncated`` semantics) rather than an arbitrary implementation choice.

See ADR-0004 (Critic gate), ADR-0005 (pytest as evaluation framework), and
PR #688 (the singular benchmark this module parallels).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="read_multiple_files_skill",
    description=(
        "Gate the read_multiple_files skill contract: multi-file batches, "
        "per-file errors, and aggregate top-level truncation."
    ),
    prompt=(
        "Use the read_multiple_files skill to: (1) read three mixed-content "
        "files (app.py, data.json, notes.md) in a single batch and emit a "
        "per-file digest; (2) read a batch where one path (missing.py) does "
        "not exist and verify only that entry reports an error; (3) read a "
        "batch with one small and one large file and verify the top-level "
        "truncated flag is true. Write results to output.txt."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "output.txt contains the per-case output: ordered per-file digests "
        "for case 1; per-file errors isolated to missing paths for case 2; "
        "and the aggregate top-level truncated flag set true when any file "
        "is truncated for case 3."
    ),
    requires_skills=["read_multiple_files"],
    tags=["read_multiple_files", "batch", "errors", "truncation"],
)

CASES = ["multi_file_basic", "missing_file_in_batch", "truncation_aggregate"]

FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name


def _seed(workspace: Path, fixture_case_dir: Path, files: list[str]) -> None:
    """Copy each named fixture file into ``workspace`` as UTF-8 text."""
    for name in files:
        (workspace / name).write_text(
            (fixture_case_dir / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _run_solution(workspace: Path) -> None:
    solution_path = FIXTURE_DIR / "solution.py"
    source = solution_path.read_text(encoding="utf-8")
    run_solution(workspace, source)


@pytest.mark.parametrize("case", CASES)
@pytest.mark.benchmark
def test_read_multiple_files_skill(benchmark_workspace: Path, case: str) -> None:
    """Deterministic pass/fail check for the read_multiple_files skill.

    Cases:
        multi_file_basic       -- 3 mixed-content files, all present.
        missing_file_in_batch  -- 2 files present + 1 referenced but absent;
                                   the absent entry must report a per-file
                                   error and the call itself must succeed.
        truncation_aggregate   -- 1 small + 1 large file; the large file
                                   must be truncated and the top-level
                                   ``truncated`` flag must be true.
    """
    fixture_case_dir = FIXTURE_DIR / case

    if case == "multi_file_basic":
        _seed(benchmark_workspace, fixture_case_dir, ["app.py", "data.json", "notes.md"])
    elif case == "missing_file_in_batch":
        # ``missing.py`` is intentionally NOT seeded -- the call must report
        # a per-file error for it without failing the whole batch.
        _seed(benchmark_workspace, fixture_case_dir, ["present.py", "settings.ini"])
    elif case == "truncation_aggregate":
        _seed(benchmark_workspace, fixture_case_dir, ["small.txt", "large_module.py"])
    else:
        pytest.fail(f"unknown case: {case}")

    _run_solution(benchmark_workspace)

    actual = (benchmark_workspace / "output.txt").read_text(encoding="utf-8")
    expected = (fixture_case_dir / "expected.txt").read_text(encoding="utf-8")

    assert actual == expected, (
        f"task {TASK.name}/{case}: output mismatch\n"
        f"--- expected ---\n{expected!r}\n"
        f"--- actual ---\n{actual!r}"
    )
