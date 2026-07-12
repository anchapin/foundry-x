"""Benchmark task: navigate a nested directory tree to find and fix a bug.

This is the first benchmark task that exercises the ``list_dir`` skill
(issue #263). Until now every task in the suite either names the target
file in the prompt (``fix_import_error``, ``cross_file_refactor``) or
operates on a flat workspace (``sort_a_list``, ``two_sum``). Real-world
coding requires the agent to **discover workspace structure** -- listing
directories to find the right file to edit -- before it can act. That
discovery step is a prerequisite for every non-trivial coding task, and
without a benchmark that exercises it the Critic has no regression target
for filesystem navigation.

The fixture seeds a nested tree (``src/`` with ``utils/`` and ``services/``
sub-packages) where one module deep in the tree --
``src/utils/formatting.py`` -- exports a ``greet`` function that is broken
(references an undefined name). ``main.py`` at the workspace root imports
and calls ``greet``. The agent must:

    1. use ``list_dir`` to explore the directory tree and discover which
       file defines ``greet``,
    2. read the broken file and diagnose the bug,
    3. edit the file so ``python main.py`` exits 0.

The golden solution mirrors the ``list_dir`` skill contract
(``harness/skills/list_dir.json``): it walks the tree with ``os.scandir``
(the only filesystem-introspection primitive the skill allows), skips
dotfiles, and reads each ``.py`` file until it finds the one containing
the broken pattern. It then applies the fix and the test independently
verifies the post-condition.

See ADR-0010 (Runner agent loop -- skill surface) and issue #168 (closed:
seed ``list_dir`` skill, zero benchmark coverage until now).
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from benchmarks.support import run_solution

TASK = BenchmarkTask(
    name="list_dir_navigation",
    description=(
        "Navigate a nested directory tree using list_dir to find the module "
        "that defines a broken greet() function, then fix it so "
        "'python main.py' exits 0."
    ),
    prompt=(
        "The workspace contains a nested directory tree under src/ with "
        "several Python modules. main.py at the workspace root imports a "
        "greet() function from somewhere inside the tree and calls it, but "
        "greet() is broken -- running 'python main.py' raises a NameError. "
        "Use list_dir to explore the directory tree, find which file "
        "defines greet(), fix the bug, and ensure 'python main.py' exits 0 "
        "and prints 'Hello world!'."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the fix, 'python main.py' returns rc=0 with stdout equal to "
        "fixtures/list_dir_navigation/expected_stdout.txt."
    ),
    timeout_seconds=30,
    requires_skills=["bash", "list_dir"],
    tags=["navigation", "filesystem", "multi-file", "debugging"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: Golden solution -- walks the tree via os.scandir (mirroring the
#: list_dir skill contract), locates the file containing the broken
#: f-string ``{nam}``, and corrects it to ``{name}``.
GOLDEN_SOLUTION = """\
import os
from pathlib import Path

#: This driver's own path -- excluded from the search so it never matches
#: itself (the broken snippet appears here as a literal in _BROKEN).
_SELF = Path(__file__).resolve()

#: The broken snippet and its fix.  The full expression ``"Hello {nam}"``
#: is used (not just ``"{nam}"``) so the search is as precise as possible.
_BROKEN = "Hello {" + "nam}"
_FIXED = "Hello {" + "name}"


def list_dir(path):
    \"""Mirror harness/skills/list_dir.json: bounded os.scandir walk.

    Returns sorted, non-hidden entries -- the same contract the list_dir
    skill exposes to the agent. The agent path would call list_dir on
    '.' then recurse into sub-directories; this golden driver does the
    same to demonstrate the navigation is genuinely required.
    \"""
    entries = []
    with os.scandir(path) as it:
        for entry in sorted(it, key=lambda e: e.name):
            if entry.name.startswith("."):
                continue
            entries.append(entry)
    return entries


def find_target(root):
    \"""Walk the tree to locate the .py file containing the broken f-string.

    Returns the Path to the file or None if no file matches. The search
    is a depth-first walk -- exactly what an agent issuing repeated
    list_dir calls would do to discover the workspace structure.
    \"""
    stack = [Path(root)]
    while stack:
        current = stack.pop()
        for entry in list_dir(current):
            if entry.is_dir():
                stack.append(Path(entry))
            elif entry.is_file() and entry.name.endswith(".py"):
                candidate = Path(entry).resolve()
                if candidate == _SELF:
                    continue
                text = Path(entry).read_text()
                if _BROKEN in text:
                    return Path(entry)
    return None


def main():
    target = find_target(".")
    if target is None:
        raise SystemExit("navigation failed: broken file not found in tree")
    fixed = target.read_text().replace(_BROKEN, _FIXED)
    target.write_text(fixed)


if __name__ == "__main__":
    main()
"""


def _seed_workspace(workspace: Path) -> None:
    """Copy the nested fixture tree into *workspace*.

    The fixture directory holds the full run-time layout (``src/`` package,
    ``main.py``, ``expected_stdout.txt``); a straight ``copytree`` with
    ``dirs_exist_ok=True`` layers it on top of the already-created
    workspace root.
    """
    shutil.copytree(_FIXTURE_DIR, workspace, dirs_exist_ok=True)


def _run_main(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python main.py`` inside *workspace* and capture all output."""
    return subprocess.run(
        [sys.executable, "main.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


@pytest.mark.benchmark
def test_list_dir_navigation(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Three-step shape (medium tier):

        1. **Pre-condition** -- seed the workspace with the fixture tree
           and run ``python main.py``.  It MUST fail: ``greet`` raises
           ``NameError`` because it references the undefined name ``nam``.
           This proves the bug is real and not silently masked.
        2. **Golden fix** -- run ``GOLDEN_SOLUTION``, which walks the tree
           with ``os.scandir`` (the ``list_dir`` skill contract) to locate
           the broken file, then corrects ``{nam}`` to ``{name}``.
        3. **Post-condition** -- re-run ``python main.py``.  It MUST exit
           rc=0 and print the expected stdout.
    """
    _seed_workspace(benchmark_workspace)

    expected_stdout = (_FIXTURE_DIR / "expected_stdout.txt").read_text()

    # --- Pre-condition: the seeded tree is genuinely broken. ------------
    bad = _run_main(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    assert (
        "NameError" in bad.stderr
    ), f"task {TASK.name}: expected NameError in stderr; got stderr={bad.stderr!r}"

    # --- Golden fix: navigate the tree and correct the broken file. -----
    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    # --- Post-condition: the corrected workspace runs end-to-end. -------
    good = _run_main(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
    assert good.stdout == expected_stdout, (
        f"task {TASK.name}: stdout mismatch " f"(got {good.stdout!r}, expected {expected_stdout!r})"
    )
