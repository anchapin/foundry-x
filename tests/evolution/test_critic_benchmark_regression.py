"""Benchmark regression-check tests for the Critic (issue #93).

ADR-0004 item 3 promises: no previously-passing benchmark task may newly fail
after a proposed harness edit. This module pins the contract by exercising
``Critic.evaluate`` against a synthetic baseline where one task passes pre-edit
and fails post-edit.

Per issue #93 acceptance criteria:

- ``test_critic_rejects_when_baseline_task_regresses_post_edit`` -- pre-edit
  baseline passes; post-edit the diff breaks task_A; ``approved is False``.
- ``test_critic_approves_when_all_baseline_tasks_still_pass`` -- pre-edit
  baseline passes; post-edit the diff is benign; ``approved is True``.
- ``test_critic_surfaces_regressed_task_name_in_verdict`` -- post-edit the
  regressed task name is identifiable from the verdict so the Evolver can
  attribute the regression (regression_report.py pairs ``failed_checks`` /
  ``passed_checks`` task names across sessions to detect regressions).

The Critic currently surfaces regressed task names in ``verdict.notes`` (the
pytest output tail). The regression_report.py contract treats
``passed_checks`` and ``failed_checks`` as lists of task names; that contract
is honoured here by exercising the verdict's combined evidence surface
(``failed_checks`` + ``notes``) so the regressed task name is recoverable from
the verdict regardless of which field it surfaces in today.
"""

from __future__ import annotations

import difflib
from pathlib import Path

from benchmarks.models import BenchmarkTask
from foundry_x.evolution.critic import Critic
from tests._harness_fixture import install_load_check_prerequisites


_BASELINE_SOURCE = """
def test_task_A():
    assert True

def test_task_B():
    assert True
""".lstrip()


_REGRESSED_SOURCE = """
def test_task_A():
    assert False

def test_task_B():
    assert True
""".lstrip()


def _write_harness(tmp_path: Path, test_source: str) -> Path:
    harness_dir = tmp_path / "harness"
    tests_dir = harness_dir / "tests"
    tests_dir.mkdir(parents=True)
    (harness_dir / "system_prompt.txt").write_text("original\n")
    (tests_dir / "test_benchmarks.py").write_text(test_source)
    # load_check prerequisites (issue #187): the Critic gates on
    # harness/scripts/load_check.py before pytest.
    install_load_check_prerequisites(harness_dir)
    return harness_dir


def _diff(relative_path: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )


def _baseline_critic(harness_dir: Path) -> Critic:
    task_a = BenchmarkTask(name="task_A", description="synthetic baseline task A")
    task_b = BenchmarkTask(name="task_B", description="synthetic baseline task B")
    return Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_benchmarks.py"],
        benchmark_tasks=[task_a, task_b],
        use_sandbox=False,
    )


def test_critic_rejects_when_baseline_task_regresses_post_edit(tmp_path: Path) -> None:
    harness_dir = _write_harness(tmp_path, _BASELINE_SOURCE)

    baseline = _baseline_critic(harness_dir).evaluate("")
    assert baseline.approved is True
    assert baseline.failed_checks == []

    regress_diff = _diff(
        "tests/test_benchmarks.py",
        _BASELINE_SOURCE,
        _REGRESSED_SOURCE,
    )

    verdict = _baseline_critic(harness_dir).evaluate(regress_diff)

    assert verdict.approved is False
    assert "pytest" in verdict.failed_checks


def test_critic_approves_when_all_baseline_tasks_still_pass(tmp_path: Path) -> None:
    harness_dir = _write_harness(tmp_path, _BASELINE_SOURCE)

    benign_diff = _diff("system_prompt.txt", "original\n", "patched\n")

    verdict = _baseline_critic(harness_dir).evaluate(benign_diff)

    assert verdict.approved is True
    assert "pytest" in verdict.passed_checks
    assert verdict.failed_checks == []


def test_critic_surfaces_regressed_task_name_in_verdict(tmp_path: Path) -> None:
    harness_dir = _write_harness(tmp_path, _BASELINE_SOURCE)

    regress_diff = _diff(
        "tests/test_benchmarks.py",
        _BASELINE_SOURCE,
        _REGRESSED_SOURCE,
    )

    verdict = _baseline_critic(harness_dir).evaluate(regress_diff)

    assert verdict.approved is False
    evidence = " ".join(verdict.failed_checks) + " " + verdict.notes
    assert "task_A" in evidence
    assert "task_B" not in evidence
