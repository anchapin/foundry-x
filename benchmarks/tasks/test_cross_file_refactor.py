"""Benchmark task: multi-file cross-reference rename.

This is a ``difficulty_tier='medium'`` task that exercises the
read-edit loop across **two** files with an import relationship:

    1. read ``utils.py`` (library) -- discover it defines ``normalize_string``
       (the OLD name),
    2. read ``main.py`` (CLI entry point) -- discover it imports and calls
       ``normalize_string`` (the OLD name),
    3. edit ``utils.py`` -- rename ``normalize_string`` -> ``sanitize_string``,
    4. edit ``main.py`` -- update the import and call site to use
       ``sanitize_string``,
    5. re-run ``python main.py`` -- must exit 0 with expected stdout.

Both files start with the OLD name ``normalize_string``.  The agent must
rename consistently across both files.  This is the core shape of a real
refactor: identify a symbol, rename it, update all references.

The ``requires_skills`` list carries both ``bash`` (to run the script and
observe the traceback) and ``edit_file`` (to apply the targeted string
replacements), so the Critic (ADR-0004) can flag the task as
"not-yet-evaluable" when either skill is absent from the harness.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="cross_file_refactor",
    description=(
        "Rename the function ``normalize_string`` to ``sanitize_string`` "
        "consistently across utils.py and main.py: both the library "
        "definition and the call site must be updated so that "
        "'python main.py' runs without import or attribute errors."
    ),
    prompt=(
        "Two files are in the workspace:\n"
        "  - utils.py: a library that defines ``normalize_string`` (the OLD name).\n"
        "  - main.py: a CLI entry point that imports and calls ``normalize_string``.\n"
        "\n"
        "Rename the function from ``normalize_string`` to ``sanitize_string``:\n"
        "  1. In utils.py, rename the function definition from ``normalize_string`` "
        "to ``sanitize_string``.\n"
        "  2. In main.py, update the import to bind the new name directly "
        "('from utils import sanitize_string') and update the call site.\n"
        "\n"
        "After both edits, 'python main.py' must exit 0 and print:\n"
        "  result=hello world\n"
        "\n"
        "Both files must reflect the rename with no import errors."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the rename, 'python main.py' returns rc=0 with stdout:\n" "  result=hello world"
    ),
    timeout_seconds=30,
    requires_skills=["bash", "edit_file"],
    tags=["refactoring", "rename", "multi-file"],
)

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

GOLDEN_UTILS = '''\
"""Utility functions for string manipulation.

Rename complete: ``normalize_string`` has been renamed to ``sanitize_string``.
"""


def sanitize_string(text: str) -> str:
    """Return a trimmed, lowercased copy of *text* stripped of punctuation."""
    import re

    text = text.strip().lower()
    return re.sub(r"[^\w\s]", "", text)
'''

GOLDEN_MAIN = '''\
"""CLI entry point that uses the string utility functions."""

from utils import sanitize_string


def main() -> None:
    """Print the sanitized version of a sample string."""
    sample = "  Hello, World!  "
    result = sanitize_string(sample)
    print(f"result={result}")


if __name__ == "__main__":
    main()
'''


def _run_main(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python main.py`` inside *workspace* and capture all output."""
    return subprocess.run(
        [sys.executable, "main.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _seed_workspace(workspace: Path) -> None:
    """Copy the static fixture files into *workspace*."""
    (workspace / "utils.py").write_text((_FIXTURE_DIR / "utils.py").read_text())
    (workspace / "main.py").write_text((_FIXTURE_DIR / "main.py").read_text())


@pytest.mark.benchmark
def test_cross_file_refactor(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Multi-step shape (medium tier):

        1. **Pre-condition** -- seed the workspace with the initial fixture
           files and run ``python main.py``.  The run MUST fail with
           ``ImportError`` or ``AttributeError``: main.py tries to import
           ``normalize_string`` which does not exist (only ``sanitize_string``
           will exist after the fix).  This proves the broken state is real.
        2. **Gold fix** -- apply the golden corrections to **both** files:
           rename the def in ``utils.py`` and update the import and call in
           ``main.py``.
        3. **Post-condition** -- re-run ``python main.py``.  It MUST exit 0
           and print ``result=hello world``, proving the rename is
           consistent end-to-end.
    """
    _seed_workspace(benchmark_workspace)

    expected_stdout = (_FIXTURE_DIR / "expected_stdout.txt").read_text()

    # --- Pre-condition: the seeded workspace is genuinely broken. ------------
    bad = _run_main(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the rename; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} "
        f"stderr={bad.stderr!r}"
    )

    # --- Gold fix: edit BOTH the utils (def rename) and main --------------
    # (import + call update) so both files agree on the new name.
    (benchmark_workspace / "utils.py").write_text(GOLDEN_UTILS)
    (benchmark_workspace / "main.py").write_text(GOLDEN_MAIN)

    # --- Post-condition: the renamed workspace runs end-to-end. -------------
    good = _run_main(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} "
        f"stderr={good.stderr!r}"
    )
    assert (
        good.stdout == expected_stdout
    ), f"task {TASK.name}: stdout mismatch (got {good.stdout!r}, expected {expected_stdout!r})"
