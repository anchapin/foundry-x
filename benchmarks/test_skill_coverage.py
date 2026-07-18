"""Meta-test: every harness skill has benchmark coverage via ``requires_skills``.

The harness advertises skills via ``harness/skills/*.json``. The Critic uses
``BenchmarkTask.requires_skills`` to flag a benchmark as "not yet evaluable"
when a required skill is absent from the harness (issue #104, ADR-0004).
But there is no inverse check: if a new skill is added and no benchmark
lists it in ``requires_skills``, that skill silently has zero regression
coverage -- a gap that is invisible until someone audits the task
definitions by hand.

This module closes that gap at CI time. It loads every skill file, extracts
the skill ``name``, then loads every ``BenchmarkTask`` via
``registry.load_all_tasks()`` and asserts that each skill name (excluding
the ``example_skill`` template) appears in at least one task's
``requires_skills`` list.

Expected state
--------------
Once a benchmark task is added for every non-excluded harness skill, the
test is green and acts as a regression guard: a new skill shipped without
a covering benchmark task will fail CI. The ``@pytest.mark.xfail`` marker
that wrapped this test in earlier revisions was removed (issue #873) once
the last uncovered skill (``read_multiple_files``) gained coverage, per
the test's original self-documenting "Expected initial state" note that
called for marker removal at full coverage.

Exclusion rationale
-------------------
``example_skill.json`` declares ``name = "read_file"`` but is the canonical
template skill (ADR-0004, issue #19), not a production tool. It is excluded
by filename stem so the assertion focuses on real skills only.

See issue #264, ADR-0009, ADR-0005.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.registry import load_all_tasks

#: Absolute path to the repository root (parent of ``benchmarks/``).
REPO_ROOT = Path(__file__).resolve().parents[1]

#: Directory containing the harness skill definitions.
SKILLS_DIR = REPO_ROOT / "harness" / "skills"

#: Skill files excluded from the coverage assertion. ``example_skill.json``
#: is the canonical template skill (its internal ``name`` is ``read_file``),
#: not a production tool, so requiring benchmark coverage for it would be
#: noise. Future template/reference skills should be added here.
_EXCLUDED_STEMS: frozenset[str] = frozenset({"example_skill"})


def _load_skill_names() -> set[str]:
    """Return the ``name`` field of every non-excluded ``harness/skills/*.json``.

    The ``name`` field is the identifier used in ``BenchmarkTask.requires_skills``
    (e.g. ``bash``), not the filename stem. For every production skill the two
    coincide, but extracting the ``name`` field is the contract-correct source
    of truth.
    """
    names: set[str] = set()
    for skill_file in sorted(SKILLS_DIR.glob("*.json")):
        if skill_file.stem in _EXCLUDED_STEMS:
            continue
        data = json.loads(skill_file.read_text(encoding="utf-8"))
        names.add(data["name"])
    return names


def _load_covered_skills() -> set[str]:
    """Return the union of every ``requires_skills`` entry across all tasks."""
    covered: set[str] = set()
    for task in load_all_tasks():
        covered.update(task.requires_skills)
    return covered


@pytest.mark.benchmark
def test_every_harness_skill_has_benchmark_coverage() -> None:
    """Assert every non-excluded harness skill appears in some ``requires_skills``.

    Fails with a descriptive message listing the uncovered skill names so
    that follow-up PRs know exactly which benchmarks to add. Once green,
    this test becomes a regression guard: a new skill shipped without a
    covering benchmark task will fail CI.
    """
    skill_names = _load_skill_names()
    covered = _load_covered_skills()

    uncovered = skill_names - covered
    assert not uncovered, (
        "The following harness skills have no benchmark coverage -- no "
        "BenchmarkTask lists them in requires_skills: "
        f"{sorted(uncovered)}. Add a benchmark task that declares these "
        "skills in its requires_skills list."
    )
