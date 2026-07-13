"""Benchmark task: gate the list-files-before-edit harness rule (issue #205).

Closes #344.

harness/system_prompt.txt:11 declares operating rule #1:
    "Before any non-trivial edit, list the files you will change and
    why."

This task is the regression target for that rule. The seeded
``target.py`` contains a deliberate off-by-one bug (``calculate``
returns ``n + 1`` instead of ``n``). The golden driver follows rule
#1: it first writes ``files.txt`` naming ``target.py`` and the reason
for the change, and only then applies the fix.

The test asserts three independent conditions:

    1. ``files.txt`` was written and names ``target.py`` -- the driver
       surfaced its edit plan rather than editing silently.
    2. ``files.txt`` was modified no later than ``target.py`` -- the
       listing genuinely preceded the edit. A regression that edits
       first and lists afterward would violate this.
    3. ``target.py`` contains the fix (``n + 1`` is gone) -- the edit
       actually happened, so the listing was not an empty gesture.

Together these assertions make rule #1 a hard gate for the
``Critic``: removing it from ``system_prompt.txt`` (or weakening the
agent loop to skip the listing step) breaks this benchmark.

See also ``test_stop_after_two_failures.py`` (rule #3) and
``test_surface_ambiguity.py`` (rule #5) -- the three together pin the
operating rules that have a concrete regression target.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="list_files_before_edit",
    description=(
        "List the files you will change and why before applying a "
        "non-trivial edit (harness/system_prompt.txt:11)."
    ),
    prompt=(
        "The file target.py contains an off-by-one bug: calculate() "
        "returns n+1 instead of n. Fix the bug. Before editing, list the "
        "files you will change and why in files.txt."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "files.txt names target.py AND files.txt was written no later "
        "than the edited target.py AND target.py no longer contains "
        "the off-by-one (n + 1 is gone)."
    ),
    tags=["harness-rule", "planning"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    # Rule #1: list the files you will change and why, BEFORE editing.
    Path("files.txt").write_text(
        "target.py: fix off-by-one in calculate() -- returns n+1 instead of n\\n"
    )

    # Apply the edit.
    Path("target.py").write_text(
        "def calculate(n):\\n    return n\\n"
    )


if __name__ == "__main__":
    main()
"""


@pytest.mark.benchmark
def test_list_files_before_edit(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Asserts:
        1. ``files.txt`` was written and names ``target.py`` -- the
           driver followed rule #1 and surfaced its edit plan.
        2. ``files.txt`` was modified no later than ``target.py`` --
           the listing preceded the edit (mtime check).
        3. ``target.py`` no longer contains the off-by-one bug.
    """
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "target.py").write_text((fixture_dir / "target.py").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    # 1. files.txt must exist and name target.py.
    files_listing = (benchmark_workspace / "files.txt").read_text()
    assert "target.py" in files_listing, (
        f"task {TASK.name}: files.txt must list target.py; got: {files_listing!r}"
    )

    # 2. The listing must precede (or be contemporaneous with) the edit.
    files_mtime = os.path.getmtime(benchmark_workspace / "files.txt")
    target_mtime = os.path.getmtime(benchmark_workspace / "target.py")
    assert files_mtime <= target_mtime, (
        f"task {TASK.name}: files.txt (mtime={files_mtime}) was written "
        f"after target.py (mtime={target_mtime}); rule #1 "
        "(harness/system_prompt.txt:11) requires listing files BEFORE editing."
    )

    # 3. The edit must have been applied -- the off-by-one is gone.
    fixed_target = (benchmark_workspace / "target.py").read_text()
    assert "n + 1" not in fixed_target, (
        f"task {TASK.name}: target.py still contains the off-by-one bug "
        f"(n + 1); got: {fixed_target!r}"
    )
