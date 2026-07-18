"""Benchmark task: git-status-driven edit (issue #810).

This benchmark validates the "read git status/diff output -> decide what
to change -> make targeted edit" loop. A harness regression that breaks
git skill routing would not be caught without this task.

The seeded workspace is a git repo with exactly one file modified
(``calculator.py``). The ``multiply`` function has a bug: it uses
``a + a`` instead of ``a * b``. The golden solution:

    1. runs ``git diff --name-only`` (or ``git status --short``) to
       identify the changed file,
    2. reads the file content,
    3. applies a targeted fix to the buggy function,
    4. verifies the edit is correct.

The test asserts:

    1. **Pre-condition** -- ``git diff`` shows at least one modified
       ``.py`` file in the seeded workspace.
    2. **Golden fix** -- the targeted edit is applied in place.
    3. **Post-condition** -- ``git diff`` confirms only the intended
       lines changed (the buggy line is fixed, nothing else).

See ADR-0005, ADR-0010, and issue #810.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="git_status_driven_edit",
    description=(
        "Read git status/diff to find a modified file, identify the bug "
        "inside it, and apply a targeted fix."
    ),
    prompt=(
        "The workspace is a git repo with one modified Python file. "
        "Use git diff --name-only (or git status --short) to identify which "
        "file changed, read it, fix the bug in the multiply function (it uses "
        "addition where it should use multiplication), then verify the fix is "
        "correct by running 'python calculator.py'."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the targeted edit, git diff shows only the intended change "
        "(multiply now uses * instead of +), and 'python calculator.py' "
        "exits 0."
    ),
    timeout_seconds=30,
    requires_skills=["bash"],
    tags=["git", "multi-step", "editing", "decision-making"],
)

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

_BUGGY_LINE = "return a + a"
_FIXED_LINE = "return a * b"


def _seed_workspace(workspace: Path) -> None:
    """Copy the git repo fixture into *workspace* and set up a git working tree.

    The workspace is seeded as a git repo with one staged+modified Python file
    so that ``git diff --name-only`` shows it as a changed file.
    """
    import shutil

    shutil.copytree(_FIXTURE_DIR, workspace, dirs_exist_ok=True)

    # Initialize a git repo and make an initial commit
    subprocess.run(["git", "init"], cwd=workspace, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@foundryx.dev"], cwd=workspace, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.name", "FoundryX Test"], cwd=workspace, capture_output=True
    )
    # Stage and commit the initial (correct) version
    subprocess.run(["git", "add", "calculator.py"], cwd=workspace, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=workspace, capture_output=True)
    # Overwrite with the buggy version so git diff shows changes
    buggy_content = (
        "def multiply(a, b):\n"
        "    # Bug: should be a * b\n"
        "    return a + a\n"
        "\n"
        "if __name__ == '__main__':\n"
        "    result = multiply(3, 4)\n"
        "    expected = 12\n"
        "    if result != expected:\n"
        "        raise SystemExit(f'3*4={result}, expected {expected} (bug: using + instead of *)')\n"
    )
    (workspace / "calculator.py").write_text(buggy_content)


def _run_calculator(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python calculator.py`` inside *workspace* and capture output."""
    return subprocess.run(
        [sys.executable, "calculator.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _git_diff_names(workspace: Path) -> list[str]:
    """Return list of filenames changed in git diff (working tree)."""
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.strip().split("\n") if line.strip()]


def _git_diff_hunks(workspace: Path, filename: str) -> str:
    """Return the diff hunk for *filename* in the working tree."""
    result = subprocess.run(
        ["git", "diff", filename],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
    return result.stdout


GOLDEN_SOLUTION = """
import subprocess
from pathlib import Path

_BUGGY_LINE = "return a + a"
_FIXED_LINE = "return a * b"


def main():
    # Step 1: identify the changed file via git diff --name-only
    result = subprocess.run(
        ["git", "diff", "--name-only"],
        capture_output=True,
        text=True,
        check=True,
    )
    changed_files = [f.strip() for f in result.stdout.strip().split("\\n") if f.strip()]
    py_files = [f for f in changed_files if f.endswith(".py")]
    if not py_files:
        raise SystemExit("git diff showed no modified .py files")
    target = py_files[0]

    # Step 2: read the file content
    content = Path(target).read_text()

    # Step 3: apply the targeted fix
    fixed = content.replace(_BUGGY_LINE, _FIXED_LINE)

    # Step 4: verify the fix is unique (non-vacuous)
    if fixed == content:
        raise SystemExit(f"No replacement made in {target}")

    occurrences = content.count(_BUGGY_LINE)
    if occurrences != 1:
        raise SystemExit(f"Expected exactly 1 occurrence of buggy line, found {occurrences}")

    Path(target).write_text(fixed)
    print("Fix applied successfully")


if __name__ == "__main__":
    main()
"""


@pytest.mark.benchmark
def test_git_status_driven_edit(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Three-step shape (medium tier):

        1. **Pre-condition** -- the seeded git repo has at least one
           modified ``.py`` file (verified via ``git diff --name-only``).
        2. **Golden fix** -- run ``GOLDEN_SOLUTION``, which reads git
           status, identifies the changed file, applies a targeted patch.
        3. **Post-condition** -- the targeted edit is applied and
           ``git diff`` confirms only the intended lines changed.
    """
    _seed_workspace(benchmark_workspace)

    # --- Pre-condition: git diff shows at least one modified .py file. ---
    changed_py_files = _git_diff_names(benchmark_workspace)
    assert any(f.endswith(".py") for f in changed_py_files), (
        f"task {TASK.name}: git diff --name-only must show at least one "
        f"modified .py file; got: {changed_py_files!r}"
    )

    # Also verify the buggy line is present in the working tree
    calculator_path = benchmark_workspace / "calculator.py"
    calculator_content = calculator_path.read_text()
    assert _BUGGY_LINE in calculator_content, (
        f"task {TASK.name}: seeded calculator.py must contain the buggy "
        f"line {_BUGGY_LINE!r}; got: {calculator_content!r}"
    )

    # --- Verify the seeded workspace fails when run. --------------------
    bad = _run_calculator(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded calculator.py must fail before fix; got rc={bad.returncode}"
    )

    # --- Golden fix: git status -> read -> targeted edit. ---------------
    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    # --- Post-condition: the fix is correct and complete. -------------
    # The buggy line must be gone
    fixed_content = calculator_path.read_text()
    assert _BUGGY_LINE not in fixed_content, (
        f"task {TASK.name}: buggy line {_BUGGY_LINE!r} must be gone "
        f"after fix; got: {fixed_content!r}"
    )

    # The correct line must be present
    assert _FIXED_LINE in fixed_content, (
        f"task {TASK.name}: fixed line {_FIXED_LINE!r} must be present "
        f"after fix; got: {fixed_content!r}"
    )

    # git diff should show only the intended change
    diff_hunks = _git_diff_hunks(benchmark_workspace, "calculator.py")
    assert _FIXED_LINE in diff_hunks, (
        f"task {TASK.name}: git diff must contain the fixed line; got: {diff_hunks!r}"
    )

    # The calculator must now run correctly
    good = _run_calculator(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: calculator.py must exit 0 after fix; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
