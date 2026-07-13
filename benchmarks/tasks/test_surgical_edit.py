"""Benchmark task: surgical edit precision (issue #266).

Every other fix-shaped benchmark (``fix_syntax_error``, ``fix_import_error``,
``multi_file_rename``) validates its golden solution by **overwriting the
entire target file** via ``run_solution`` (``benchmarks/support.py:19-28``).
No task tests whether an edit **preserves unrelated code in the same file**.

This task seeds a multi-function ``module.py`` where exactly one function
(``median``) has a bug and the golden solution patches only that function
body -- leaving the other three functions (``square``, ``is_even``,
``average``) byte-identical. It exercises ``edit_file`` precision
(``harness/skills/edit_file.json``) and gives the Critic a regression
target for "the agent clobbered unrelated code."

The golden fix is modelled as a targeted ``old_string`` -> ``new_string``
replacement -- exactly the ``edit_file`` contract -- NOT a full-file
overwrite. The test then asserts:

    1. **Pre-condition** -- the seeded module crashes (rc != 0) with
       ``IndexError`` from the buggy ``median``.
    2. **Surgical edit** -- the ``old_string`` matches exactly once
       (the patch is unique and targeted) and the replacement is
       applied in place.
    3. **Post-condition** -- the patched module runs rc=0 with the
       expected stdout.
    4. **Precision** -- the three untouched functions are byte-identical
       before and after the edit (sha256 of their AST source segments
       match), and the buggy function's hash changed (non-vacuous guard).

See also ADR-0010 (Runner agent loop) and ``harness/skills/edit_file.json``.
"""

from __future__ import annotations

import ast
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="surgical_edit",
    description=(
        "Patch only the buggy function body in a multi-function module so "
        "the module runs rc=0, leaving the other functions byte-identical "
        "(exercises edit_file precision)."
    ),
    prompt=(
        "The file module.py contains four functions. Exactly one -- "
        "median() -- has a bug that raises IndexError, so 'python "
        "module.py' exits non-zero. Fix the bug by editing ONLY the "
        "median() function body. Do not rewrite the whole file and do "
        "not modify square(), is_even(), or average(); they must remain "
        "byte-identical. After the fix, 'python module.py' must exit 0 "
        "and print the expected output."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After a targeted edit to median() only, 'python module.py' "
        "returns rc=0 with stdout equal to "
        "fixtures/surgical_edit/expected_stdout.txt, and the other three "
        "functions are unchanged (byte-identical source)."
    ),
    timeout_seconds=30,
    requires_skills=["edit_file"],
    tags=["editing", "precision", "surgical"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: Functions that MUST NOT change during the surgical edit. The golden
#: patch touches only the buggy function; these three are the regression
#: target for "unrelated code was preserved."
UNTOUCHED_FUNCTIONS: tuple[str, ...] = ("square", "is_even", "average")

#: The single buggy function the golden edit patches.
BUGGY_FUNCTION = "median"

#: Targeted ``edit_file``-shaped patch for the buggy ``median`` body.
#: ``old_string`` is the exact substring present in the seeded module;
#: ``new_string`` is the corrected body. Only these lines change -- the
#: function signature, docstring, and the ``ordered``/``mid`` setup lines
#: are left intact, as are all three other functions.
GOLDEN_EDIT_OLD = (
    "    # BUG: off-by-two index -> IndexError on any non-empty input.\n"
    "    return ordered[mid + 2]\n"
)
GOLDEN_EDIT_NEW = (
    "    if len(ordered) % 2 == 0:\n"
    "        return (ordered[mid - 1] + ordered[mid]) / 2\n"
    "    return float(ordered[mid])\n"
)


def _run_module(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python module.py`` inside *workspace* and capture all output."""
    return subprocess.run(
        [sys.executable, "module.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _function_hashes(source: str) -> dict[str, str]:
    """Return ``{function_name: sha256(source_segment)}`` for top-level funcs.

    Uses :func:`ast.get_source_segment` to extract each top-level
    ``FunctionDef``'s exact source text (signature + docstring + body),
    then hashes it. Two identical source segments yield the same hash, so
    a function that was not touched by the edit hashes identically before
    and after -- the byte-identical precision check.
    """
    tree = ast.parse(source)
    return {
        node.name: hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
    }


@pytest.mark.benchmark
def test_surgical_edit(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Asserts:
        1. The seeded module is genuinely broken (rc != 0, IndexError).
        2. The golden ``old_string`` matches exactly once (surgical/unique).
        3. After the in-place edit, ``python module.py`` exits 0 with the
           expected stdout.
        4. The three untouched functions are byte-identical (sha256 match)
           and the buggy function changed (non-vacuous).
    """
    seeded_source = (_FIXTURE_DIR / "module.py").read_text()
    expected_stdout = (_FIXTURE_DIR / "expected_stdout.txt").read_text()
    module_path = benchmark_workspace / "module.py"
    module_path.write_text(seeded_source)

    # --- Pre-condition: the seeded module is genuinely broken. --------
    bad = _run_module(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded module.py must fail before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} "
        f"stderr={bad.stderr!r}"
    )
    assert "IndexError" in bad.stderr, (
        f"task {TASK.name}: expected IndexError in stderr; got stderr={bad.stderr!r}"
    )

    # --- Capture pre-edit function hashes. ----------------------------
    before = _function_hashes(seeded_source)

    # --- Apply the golden surgical edit (edit_file contract). ---------
    # Exactly one match must occur: the patch is unique and targeted,
    # not a full-file rewrite.
    occurrences = seeded_source.count(GOLDEN_EDIT_OLD)
    assert occurrences == 1, (
        f"task {TASK.name}: golden old_string must match exactly once "
        f"in the seeded module; found {occurrences}"
    )
    patched_source = seeded_source.replace(GOLDEN_EDIT_OLD, GOLDEN_EDIT_NEW, 1)
    module_path.write_text(patched_source)

    # --- Post-condition: the patched module runs end-to-end. ----------
    good = _run_module(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: patched module.py must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} "
        f"stderr={good.stderr!r}"
    )
    assert good.stdout == expected_stdout, (
        f"task {TASK.name}: stdout mismatch (got {good.stdout!r}, expected {expected_stdout!r})"
    )

    # --- Precision: untouched functions are byte-identical. -----------
    after = _function_hashes(patched_source)
    for fname in UNTOUCHED_FUNCTIONS:
        assert before[fname] == after[fname], (
            f"task {TASK.name}: untouched function {fname}() changed "
            f"during the surgical edit; the fix must patch only "
            f"{BUGGY_FUNCTION}()."
        )
    # Non-vacuous guard: the buggy function must actually have changed.
    assert before[BUGGY_FUNCTION] != after[BUGGY_FUNCTION], (
        f"task {TASK.name}: {BUGGY_FUNCTION}() hash unchanged after the "
        f"golden edit; the patch did not alter the buggy function."
    )
