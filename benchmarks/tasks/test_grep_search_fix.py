"""Benchmark task: grep-search-and-fix a stale symbol reference (issue #262).

This is the first benchmark task that exercises the ``grep_search``
skill. Until now no task in the suite listed ``grep_search`` in
``requires_skills`` -- the harness advertises the skill
(``harness/skills/grep_search.json``) but the Critic had no regression
target proving the agent path can actually drive a grep-locate-edit-test
loop. This task closes that gap.

The fixture seeds a small package of Python files where the lookup
function in ``models.py`` has been renamed (``get_user`` ->
``fetch_user``) but exactly one caller -- ``services.py`` -- still
imports the OLD name, so ``python -m pytest`` fails collection with
``ImportError``. A decoy file (``helpers.py``) already uses the new
name, so a grep for the stale symbol must discriminate which file
still holds the broken reference rather than blindly rewriting every
file. The agent must:

    1. use ``grep_search`` to locate the stale ``get_user`` reference
       across the workspace (it lives only in ``services.py``),
    2. edit that file so both the import and the call use
       ``fetch_user``,
    3. re-run ``python -m pytest`` so it exits 0.

The golden solution mirrors the ``grep_search`` skill contract
(``harness/skills/grep_search.json``): a bounded regex walk over
``*.py`` files using ``re.compile`` + ``Path.rglob`` + a
utf-8/replace read -- the same stdlib-only surface the skill exposes to
the agent. It excludes itself from the walk (the stale symbol appears
here as a literal in ``_OLD``) exactly as ``list_dir_navigation``'s
golden driver excludes its own path.

See ADR-0010 (Runner agent loop -- skill surface), ADR-0005, issue #110
(closed: ``fix_import_error`` multi-step task), and issue #264 (closed:
skill-coverage meta-test that ``grep_search`` previously failed).
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
    name="grep_search_fix",
    description=(
        "Grep across a multi-file workspace to locate the one caller still "
        "referencing a renamed symbol, edit it so 'python -m pytest' collects "
        "cleanly, and confirm the test run exits 0."
    ),
    prompt=(
        "The workspace contains several Python files. In models.py a lookup "
        "function was renamed, but one caller still imports the OLD name, so "
        "'python -m pytest' currently fails during collection with an "
        "ImportError. Use grep_search to find which file still references the "
        "stale symbol, edit that file so it uses the new name on both the "
        "import and the call site, then ensure 'python -m pytest' exits 0."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the fix, 'python -m pytest' returns rc=0 (the single "
        "test_describe_user test passes), and only the file holding the stale "
        "reference has been edited."
    ),
    timeout_seconds=30,
    requires_skills=["bash", "grep_search"],
    tags=["grep", "search", "rename", "multi-file", "debugging"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: The stale symbol (renamed away in models.py) and its replacement. Only
#: ``services.py`` contains this literal in the seeded workspace, so the
#: grep uniquely pinpoints the file to edit.
_OLD_SYMBOL = "get_user"
_NEW_SYMBOL = "fetch_user"

#: Golden solution -- mirrors the ``grep_search`` skill contract
#: (``re.compile`` + ``Path.rglob`` + utf-8/replace read, the stdlib-only
#: surface the skill exposes). It excludes its own path so the literal
#: ``_OLD`` value does not match the driver itself, then locates the file
#: still referencing the stale symbol and renames both the import and the
#: call site in place.
GOLDEN_SOLUTION = """\
import re
from pathlib import Path

# This driver's own path -- excluded from the search so it never matches
# itself (the stale symbol appears here as a literal in _OLD).
_SELF = Path(__file__).resolve()

# The stale symbol (renamed away in models.py) and its replacement.
_OLD = "get_user"
_NEW = "fetch_user"
_PATTERN = re.compile(r"\\bget_user\\b")


def grep_search(root, pattern):
    # Mirror harness/skills/grep_search.json: bounded regex walk over
    # *.py files using re.compile + Path.rglob + utf-8/replace read --
    # the same stdlib-only contract the skill exposes to the agent.
    hits = []
    for path in sorted(Path(root).rglob("*.py")):
        if path.resolve() == _SELF:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if pattern.search(text):
            hits.append(path)
    return hits


def main():
    targets = grep_search(".", _PATTERN)
    if not targets:
        raise SystemExit("grep failed: no file references the stale symbol")
    for target in targets:
        text = target.read_text(encoding="utf-8", errors="replace")
        target.write_text(text.replace(_OLD, _NEW))


if __name__ == "__main__":
    main()
"""


def _seed_workspace(workspace: Path) -> None:
    """Copy the fixture package into *workspace*.

    The fixture directory holds the full run-time layout (source modules
    plus the pytest entry point); ``copytree`` with ``dirs_exist_ok=True``
    layers it on top of the already-created workspace root.
    """
    shutil.copytree(_FIXTURE_DIR, workspace, dirs_exist_ok=True)


def _run_pytest(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python -m pytest`` inside *workspace* and capture all output."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprover"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


@pytest.mark.benchmark
def test_grep_search_fix(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Three-step shape (medium tier):

        1. **Pre-condition** -- seed the workspace with the fixture
           package and run ``python -m pytest``. It MUST fail during
           collection: ``services`` imports the stale ``get_user`` name
           from ``models``, which no longer defines it (renamed to
           ``fetch_user``). This proves the broken reference is real and
           not silently masked.
        2. **Golden fix** -- run ``GOLDEN_SOLUTION``, which greps the
           workspace for the stale symbol (locating it uniquely in
           ``services.py``) and renames both the import and the call site
           to the new name in place.
        3. **Post-condition** -- re-run ``python -m pytest``. It MUST
           exit rc=0 with the single ``test_describe_user`` test passing.
    """
    _seed_workspace(benchmark_workspace)

    # --- Pre-condition: the seeded workspace is genuinely broken. --------
    bad = _run_pytest(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    combined_bad = bad.stdout + bad.stderr
    assert "ImportError" in combined_bad, (
        f"task {TASK.name}: expected ImportError during collection; "
        f"got stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )

    # --- Golden fix: grep to locate, then edit the stale file in place. --
    run_solution(benchmark_workspace, GOLDEN_SOLUTION)

    # --- Post-condition: the corrected workspace runs end-to-end. --------
    good = _run_pytest(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
    assert "1 passed" in good.stdout, (
        f"task {TASK.name}: expected '1 passed' in pytest summary; got stdout={good.stdout!r}"
    )
