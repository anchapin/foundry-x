"""Benchmark task: Critic.evaluate() enforces the diff-size cap (issue #333).

Regression target: ``src/foundry_x/evolution/critic.py`` (ADR-0004).

SECURITY.md §"Rate limits" names "max M lines of harness diff per proposal"
as a guardrail against resource-exhaustion attacks.  The ``Critic`` enforces
this cap at the gate so that even a diff that bypasses ``Evolver._validate_edit``
(issue #333) is caught before the sandbox copy is mutated.

A diff exceeding ``max_diff_lines`` (default 200, mirroring the Evolver
default) must cause ``CriticVerdict.verdict=False`` with
``failed_checks=["diff_size_cap"]`` before git apply is attempted.
"""

from __future__ import annotations

import difflib
import shutil
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.evolution.critic import Critic

TASK = BenchmarkTask(
    name="critic_diff_size_cap",
    description=(
        "Critic.evaluate() rejects a unified_diff exceeding max_diff_lines "
        "(default 200) with failed_checks=['diff_size_cap'] before git apply "
        "is attempted, preventing resource-exhaustion via oversized patches."
    ),
    prompt=(
        "Inspect src/foundry_x/evolution/critic.py: confirm that "
        "Critic.evaluate() counts the lines in the proposed diff and rejects "
        "it with failed_checks=['diff_size_cap'] when the count exceeds "
        "max_diff_lines; the check must run before git apply is attempted."
    ),
    difficulty_tier="easy",
    expected_outcome=(
        "Critic.evaluate() returns verdict=False with 'diff_size_cap' in "
        "failed_checks and notes mentioning the line count when the diff "
        "exceeds max_diff_lines."
    ),
    tags=["security", "benchmark"],
)


def _make_minimal_harness(root: Path) -> Path:
    """Minimal harness that satisfies load_check prerequisites (issue #187)."""
    (root / "skills").mkdir(parents=True, exist_ok=True)
    (root / "system_prompt.txt").write_text("You are a helpful agent.\n")
    hooks_dir = root / "hooks"
    hooks_dir.mkdir(parents=True, exist_ok=True)
    (hooks_dir / "__init__.py").write_text(
        "class HookRegistry:\n"
        "    def __init__(self):\n"
        "        self._hooks = []\n"
        "\n"
        "    def register(self, hook):\n"
        "        self._hooks.append(hook)\n"
        "\n"
        "\n"
        "def get_registry():\n"
        "    return HookRegistry()\n"
    )
    scripts_dir = root / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    repo_root = Path(__file__).resolve().parents[2]
    shutil.copyfile(
        repo_root / "harness" / "scripts" / "load_check.py",
        scripts_dir / "load_check.py",
    )
    (root / "manifest.json").write_text(
        '{"version": "0.0.0", "model_target": "test", "hooks": [], "skills": []}'
    )
    tests_dir = root / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test_smoke.py").write_text(
        "import pytest\n\n@pytest.mark.benchmark\ndef test_smoke():\n    assert True\n"
    )
    return root


_SYSTEM_PROMPT_LINES = ["You are a helpful agent.\n"]


@pytest.mark.benchmark
def test_critic_rejects_oversized_diff(benchmark_workspace: Path) -> None:
    """A diff exceeding max_diff_lines is rejected before git apply is attempted.

    The diff has 250 lines against a cap of 200 (default).  The gate must
    return ``verdict=False`` with ``'diff_size_cap'`` in failed_checks
    immediately, without running git apply or pytest.
    """
    harness = _make_minimal_harness(benchmark_workspace / "harness")

    cap = 200
    old_lines = _SYSTEM_PROMPT_LINES[:]
    new_lines = _SYSTEM_PROMPT_LINES[:] + [f"extra line {i}\n" for i in range(240)]
    oversized_diff = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="a/system_prompt.txt",
            tofile="b/system_prompt.txt",
            lineterm="\n",
        )
    )
    actual_lines = len(oversized_diff.splitlines())
    assert actual_lines > cap, f"test precondition: diff must exceed cap ({actual_lines} vs {cap})"

    critic = Critic(harness, max_diff_lines=cap)
    verdict = critic.evaluate(oversized_diff)

    assert verdict.verdict is False, (
        f"task {TASK.name}: Critic.evaluate() must return verdict=False "
        f"for a {actual_lines}-line diff exceeding cap={cap}; "
        f"got verdict={verdict.verdict}"
    )
    assert "diff_size_cap" in verdict.failed_checks, (
        f"task {TASK.name}: expected 'diff_size_cap' in failed_checks, got {verdict.failed_checks}"
    )
    assert "git apply" not in verdict.passed_checks, (
        f"task {TASK.name}: 'git apply' must not appear in passed_checks; "
        f"the cap check must run before apply (issue #333)"
    )


@pytest.mark.benchmark
def test_critic_accepts_rightsized_diff(benchmark_workspace: Path) -> None:
    """A diff within the size cap passes the gate.

    A diff of exactly max_diff_lines lines must be accepted; the boundary
    is inclusive (cap=200 means 200-line diffs are valid).
    """
    harness = _make_minimal_harness(benchmark_workspace / "harness")

    cap = 200
    old_lines = _SYSTEM_PROMPT_LINES[:]
    new_lines = _SYSTEM_PROMPT_LINES[:] + [f"extra line {i}\n" for i in range(186)]
    rightsized_diff = "".join(
        difflib.unified_diff(
            old_lines,
            new_lines,
            fromfile="a/system_prompt.txt",
            tofile="b/system_prompt.txt",
            lineterm="\n",
        )
    )
    actual_lines = len(rightsized_diff.splitlines())
    assert actual_lines <= cap, (
        f"test precondition: diff must be within cap ({actual_lines} vs {cap})"
    )

    critic = Critic(harness, max_diff_lines=cap)
    verdict = critic.evaluate(rightsized_diff)

    assert verdict.verdict is True, (
        f"task {TASK.name}: Critic.evaluate() must return verdict=True "
        f"for a {actual_lines}-line diff within cap={cap}; "
        f"got verdict={verdict.verdict}, failed_checks={verdict.failed_checks}"
    )


@pytest.mark.benchmark
def test_critic_diff_size_cap_default_matches_evolver(benchmark_workspace: Path) -> None:
    """The Critic default max_diff_lines matches the Evolver default (200).

    SECURITY.md §"Rate limits" calls for the same M for both the Evolver
    and the Critic; a regression that changes only one side creates a gap
    the other side cannot close.
    """
    from foundry_x.evolution.evolver import Evolver

    harness = _make_minimal_harness(benchmark_workspace / "harness")
    evolver = Evolver()
    critic = Critic(harness)

    assert critic.max_diff_lines == evolver.max_diff_lines == 200, (
        f"task {TASK.name}: Critic.max_diff_lines ({critic.max_diff_lines}) "
        f"and Evolver.max_diff_lines ({evolver.max_diff_lines}) must both "
        f"equal 200 (SECURITY.md default)"
    )
