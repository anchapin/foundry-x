"""Benchmark task: gate the surface-ambiguity harness rule (issue #205).

harness/system_prompt.txt:15 declares operating rule #5:
    "When a task is ambiguous, surface the ambiguity explicitly instead
    of guessing."

This task is the regression target for that rule. The task prompt asks
the agent to "sort the data" in ``data.csv`` and write
``sorted.csv`` -- but the request does not specify the sort key (name
or age?) or the direction (ascending or descending?). The only correct
behaviour under rule #5 is to surface the ambiguity rather than
silently picking an interpretation.

The golden driver follows rule #5: it writes ``ambiguity.txt``
containing the surfaced question and does **not** produce
``sorted.csv``. The test asserts both sides of that contract:

    1. ``ambiguity.txt`` was written and contains a question marker
       (``?``) -- the driver surfaced the ambiguity explicitly.
    2. ``sorted.csv`` does **not** exist -- the driver refrained from
       guessing; a regression that silently picks a sort order would
       produce ``sorted.csv`` and skip ``ambiguity.txt``.

Together these assertions make rule #5 a hard gate for the
``Critic``: removing it from ``system_prompt.txt`` (or weakening the
agent loop to guess on ambiguous input) breaks this benchmark.

See also ``test_stop_after_two_failures.py`` (rule #3) and
``test_list_files_before_edit.py`` (rule #1).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="surface_ambiguity",
    description=(
        "Surface an ambiguous requirement explicitly instead of guessing "
        "(harness/system_prompt.txt:15)."
    ),
    prompt=(
        "The file data.csv contains user records with name and age "
        "columns. Sort the data and write the result to sorted.csv."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "ambiguity.txt contains a question surfacing the ambiguity "
        "(sort key and/or direction unspecified) AND sorted.csv does "
        "not exist (the driver did not guess)."
    ),
    tags=["harness-rule", "ambiguity"],
)

GOLDEN_SOLUTION = """\
from pathlib import Path


def main() -> None:
    # Rule #5: the task is ambiguous -- "sort the data" does not specify
    # the sort key (name or age) or the direction (ascending/descending).
    # Surface the ambiguity instead of guessing.
    Path("ambiguity.txt").write_text(
        "Ambiguity surfaced: 'sort the data' does not specify:\\n"
        "  1. Which column is the sort key (name or age)?\\n"
        "  2. Ascending or descending order?\\n"
        "Please clarify before I proceed.\\n"
    )


if __name__ == "__main__":
    main()
"""


@pytest.mark.benchmark
def test_surface_ambiguity(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Asserts:
        1. ``ambiguity.txt`` was written and contains a question marker
           (``?``) -- the driver surfaced the ambiguity explicitly.
        2. ``sorted.csv`` does **not** exist -- the driver refrained from
           guessing; a regression that silently picks a sort order would
           produce ``sorted.csv`` and skip the ambiguity file.
    """
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "data.csv").write_text((fixture_dir / "data.csv").read_text())

    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    # 1. ambiguity.txt must exist and contain a question.
    ambiguity = (benchmark_workspace / "ambiguity.txt").read_text()
    assert ambiguity.strip(), (
        f"task {TASK.name}: ambiguity.txt is empty; rule #5 "
        "(harness/system_prompt.txt:15) requires surfacing the ambiguity."
    )
    assert "?" in ambiguity, (
        f"task {TASK.name}: ambiguity.txt must contain a question marker "
        f"(surfaces the ambiguity rather than guessing); got: {ambiguity!r}"
    )

    # 2. sorted.csv must NOT exist -- the driver must not have guessed.
    assert not (benchmark_workspace / "sorted.csv").exists(), (
        f"task {TASK.name}: sorted.csv exists -- the driver guessed "
        "instead of surfacing the ambiguity (rule #5 violation)."
    )
