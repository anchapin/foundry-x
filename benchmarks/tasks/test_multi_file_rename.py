"""Benchmark task: coordinated multi-file function rename.

This is a ``difficulty_tier='medium'`` task that exercises the
read-edit-test loop across **two** files -- the core shape of a real
refactor (issue #267).  Unlike ``fix_import_error`` (#110), which edits
a single file, and ``cross_file_refactor`` (#176), which adds a helper
plus fixes a test expectation, this task requires the agent to rename a
symbol consistently in two coordinated files:

    1. read ``callee.py`` (library) -- discover it still defines the OLD
       name ``compute_value``,
    2. read ``caller.py`` (entry point) -- discover it already calls the
       NEW name ``calculate_value`` via ``callee.calculate_value()``,
    3. edit ``callee.py`` -- rename ``compute_value`` -> ``calculate_value``
       so the def matches the caller's already-migrated reference,
    4. edit ``caller.py`` -- update the import to bind the new name
       directly (``from callee import calculate_value``) and call it,
    5. re-run ``python -m caller`` -- must exit 0 with the expected stdout.

The seeded workspace is deliberately *half-renamed*: the caller has been
migrated to the new name but the callee has not, so ``python -m caller``
fails with ``AttributeError``.  The agent must complete the rename in the
callee and reconcile the caller's import -- the smallest credible shape
that proves the harness can drive a coordinated multi-file edit and then
verify it end-to-end.  This gives the Digester (ADR-0010, Phase 2) a
richer failure surface to classify once real traces flow.

The ``requires_skills`` list carries both ``bash`` (to run the module and
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
    name="multi_file_rename",
    description=(
        "Rename a function consistently across callee.py and caller.py: the "
        "caller already references the new name but the callee still defines "
        "the old name, so 'python -m caller' raises AttributeError until both "
        "files are reconciled."
    ),
    prompt=(
        "Two files are in the workspace:\n"
        "  - callee.py: a library that defines compute_value (the OLD name).\n"
        "  - caller.py: an entry point that calls callee.calculate_value() "
        "(the NEW name) and is run with 'python -m caller'.\n"
        "\n"
        "The workspace is half-renamed: caller.py already uses the new name "
        "calculate_value, but callee.py still defines compute_value, so "
        "'python -m caller' currently fails with AttributeError.\n"
        "\n"
        "Complete the coordinated rename so both files agree on the name "
        "calculate_value:\n"
        "  1. In callee.py, rename the function compute_value -> "
        "calculate_value (keep its body unchanged).\n"
        "  2. In caller.py, update the import to bind the new name directly "
        "('from callee import calculate_value') and call it.\n"
        "\n"
        "After both edits, 'python -m caller' must exit 0 and print the "
        "expected output."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the rename, 'python -m caller' returns rc=0 with stdout equal "
        "to fixtures/multi_file_rename/expected_stdout.txt."
    ),
    timeout_seconds=30,
    requires_skills=["bash", "edit_file"],
    tags=["refactoring", "rename", "multi-file"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: Golden ``callee.py`` -- ``compute_value`` renamed to ``calculate_value``.
#: Only the definition name changes; the body is identical to the seed so the
#: rename is a pure symbol rename (no behavioural drift to reason about).
GOLDEN_CALLEE = '''\
"""Callee module for the multi_file_rename benchmark (golden).

``compute_value`` has been renamed to ``calculate_value`` to match the
caller's already-migrated reference.
"""


def calculate_value() -> str:
    """Return the canonical value string (renamed from compute_value)."""
    return "value=7"
'''

#: Golden ``caller.py`` -- import updated from module-attribute access
#: (``import callee`` / ``callee.calculate_value()``) to a direct ``from``-import
#: of the new name.  This is the second coordinated edit: the caller must be
#: reconciled with the renamed callee, not merely re-pointed at it.
GOLDEN_CALLER = '''\
"""Caller module for the multi_file_rename benchmark (golden).

Import updated to bind the new name directly.
"""

from callee import calculate_value


def main() -> None:
    """Print the value produced by the callee (new name)."""
    print(calculate_value())


if __name__ == "__main__":
    main()
'''


def _run_caller(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python -m caller`` inside *workspace* and capture all output."""
    return subprocess.run(
        [sys.executable, "-m", "caller"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _seed_workspace(workspace: Path) -> None:
    """Copy the static fixture files into *workspace* under their run-time names."""
    (workspace / "callee.py").write_text((_FIXTURE_DIR / "callee.py").read_text())
    (workspace / "caller.py").write_text((_FIXTURE_DIR / "caller.py").read_text())


@pytest.mark.benchmark
def test_multi_file_rename(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Multi-step shape (medium tier):

        1. **Pre-condition** -- seed the workspace with the half-renamed
           fixture files and run ``python -m caller``.  The call MUST fail
           with ``AttributeError``: the caller references
           ``callee.calculate_value`` but the callee still defines
           ``compute_value``.  This proves the inconsistency is real and
           not silently masked by the fixture.
        2. **Gold fix** -- apply the golden corrections to **two** files:
           rename the def in ``callee.py`` and update the import in
           ``caller.py`` so both agree on ``calculate_value``.
        3. **Post-condition** -- re-run ``python -m caller``.  It MUST
           exit 0 and print the expected stdout, proving the rename is
           consistent end-to-end.
    """
    _seed_workspace(benchmark_workspace)

    expected_stdout = (_FIXTURE_DIR / "expected_stdout.txt").read_text()

    # --- Pre-condition: the seeded workspace is genuinely broken. --------
    bad = _run_caller(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the rename; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} "
        f"stderr={bad.stderr!r}"
    )
    assert (
        "AttributeError" in bad.stderr
    ), f"task {TASK.name}: expected AttributeError in stderr; got stderr={bad.stderr!r}"

    # --- Gold fix: edit BOTH the callee (def rename) and the caller -------
    # (import update) so the two files agree on the new name.
    (benchmark_workspace / "callee.py").write_text(GOLDEN_CALLEE)
    (benchmark_workspace / "caller.py").write_text(GOLDEN_CALLER)

    # --- Post-condition: the renamed workspace runs end-to-end. ----------
    good = _run_caller(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} "
        f"stderr={good.stderr!r}"
    )
    assert (
        good.stdout == expected_stdout
    ), f"task {TASK.name}: stdout mismatch (got {good.stdout!r}, expected {expected_stdout!r})"
