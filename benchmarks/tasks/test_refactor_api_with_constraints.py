"""Benchmark task: keyword-only API refactor with a test constraint (ADR-0028 H2).

This is the second ``difficulty_tier='hard'`` task in the suite.  It
implements Archetype H2 ("multi-file coordinated refactor with
constraint") from ADR-0028 §3:

    A library exposes a public API that is used across multiple packages.
    The agent must change the signature ... then update every caller to
    match the new contract.  The agent is not told which files use the
    API; it must discover them.  Additionally, a constraint applies: a
    test in one of the caller packages must still pass after the
    refactor, demonstrating the agent did not simply delete the test.

The seeded workspace is half-migrated (mirroring the
``refactor_across_three_files`` medium task, but at hard-tier scale):

    * ``lib/math_utils.py`` exposes ``clamp(value, *, lo, hi)`` -- the
      NEW keyword-only signature is already in place.
    * ``app/calc.py``, ``app/stats.py`` and ``app/ui.py`` still call
      ``clamp`` with two positional arguments (the OLD signature), so
      importing the app and exercising it raises ``TypeError``.
    * ``tests/test_stats.py::test_clamp_mean`` also contains a direct
      positional ``clamp`` call -- the agent must fix that call too.

The agent must:

    1. Read ``lib/math_utils.py`` to learn the new keyword-only contract.
    2. Use grep/search to discover every positional ``clamp(...)`` caller
       across the workspace (the prompt does not name them).
    3. Update all three application callers and the direct call inside
       ``test_clamp_mean`` to the keyword-only form.
    4. Leave ``test_clamp_mean`` present and passing (the constraint).
    5. Confirm ``python -m pytest`` exits 0.

This satisfies the four hard-tier criteria in ADR-0028 §2:

    * Multi-phase reasoning: the agent must first identify the failure
      mode (TypeError from keyword-only), then systematically locate
      every caller rather than fixing only the one in the traceback.
    * Cross-module scope: six files across two packages plus tests
      (``lib/math_utils.py``, ``lib/__init__.py``, ``app/calc.py``,
      ``app/stats.py``, ``app/ui.py``, ``tests/test_stats.py``).
    * Non-trivial state: the new keyword-only contract is the state that
      must be propagated to every caller; a single missed caller leaves
      the suite red, so the agent must track completion across files.
    * Precise outcome: ``python -m pytest`` return code plus an explicit
      check that ``test_clamp_mean`` is still defined.

The prompt deliberately does not name ``calc.py``/``stats.py``/``ui.py``
(ADR-0028 §4 "No hint leakage").
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="refactor_api_with_constraints",
    description=(
        "lib/math_utils.py exposes clamp(value, *, lo, hi) -- a "
        "keyword-only signature -- but the three application callers in "
        "app/ and a direct call in tests/test_stats.py still invoke it "
        "positionally, so 'python -m pytest' fails with TypeError. "
        "Discover every positional caller, update them to the keyword-only "
        "form, and ensure tests/test_stats.py::test_clamp_mean still "
        "exists and passes (do not delete it)."
    ),
    prompt=(
        "The workspace contains a small library (lib/) and an application "
        "(app/) with tests (tests/). The public function "
        "lib/math_utils.py::clamp has been migrated to a keyword-only "
        "signature: clamp(value, *, lo, hi). However, several callers "
        "still call it positionally, e.g. clamp(x, 0, 100), so "
        "'python -m pytest' currently fails with TypeError.\n"
        "\n"
        "Find every positional caller of clamp across the workspace and "
        "update each one to the keyword-only form (e.g. clamp(x, lo=0, "
        "hi=100)). Do not change the clamp signature itself, and do not "
        "delete or skip any test -- every existing test, including "
        "tests/test_stats.py::test_clamp_mean, must still exist and "
        "pass after your edits.\n"
        "\n"
        "After your changes, 'python -m pytest' must exit 0."
    ),
    difficulty_tier="hard",
    expected_outcome=(
        "After updating all positional callers to keyword-only, "
        "'python -m pytest' exits 0 with 2 passed, and "
        "tests/test_stats.py still defines test_clamp_mean (it was "
        "corrected, not deleted)."
    ),
    timeout_seconds=60,
    requires_skills=["bash", "grep_search", "edit_file"],
    tags=["refactoring", "api-migration", "keyword-only", "multi-file", "cross-module"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: Golden callers -- every positional call migrated to keyword-only.
GOLDEN_CALC = '''\
"""calc module for refactor_api_with_constraints (golden).

The positional clamp call has been migrated to the keyword-only form.
"""

from lib.math_utils import clamp


def bounded_sum(a: float, b: float) -> float:
    """Return ``a + b`` clamped to [0, 100]."""
    return clamp(a + b, lo=0, hi=100)
'''

GOLDEN_STATS = '''\
"""stats module for refactor_api_with_constraints (golden).

The positional clamp call has been migrated to the keyword-only form.
"""

from lib.math_utils import clamp


def clamp_mean(values: list[float]) -> float:
    """Return the mean of *values* clamped to [0, 1000]."""
    m = sum(values) / len(values)
    return clamp(m, lo=0, hi=1000)
'''

GOLDEN_UI = '''\
"""ui module for refactor_api_with_constraints (golden).

The positional clamp call has been migrated to the keyword-only form.
"""

from lib.math_utils import clamp


def display_bar(value: float, width: int = 5) -> str:
    """Render *value* clamped to [1, width] as a row of '#' characters."""
    v = clamp(value, lo=1, hi=width)
    return "#" * int(v)
'''

#: Golden test file -- the direct positional clamp call inside
#: ``test_clamp_mean`` is migrated to keyword-only.  The test still EXISTS
#: (the constraint) and still asserts the same behaviour.
GOLDEN_TEST_STATS = '''\
"""Stats tests for refactor_api_with_constraints (golden).

The direct positional clamp call has been migrated to keyword-only.
test_clamp_mean still exists and still asserts clamp_mean + clamp.
"""

from lib.math_utils import clamp
from app.stats import clamp_mean


def test_clamp_mean() -> None:
    """clamp_mean clamps the computed mean; the direct clamp call below
    also exercises the keyword-only contract."""
    assert clamp_mean([50, 150]) == 100
    assert clamp(100, lo=0, hi=1000) == 100


def test_clamp_bounds() -> None:
    """Direct keyword-only clamp call (already migrated)."""
    assert clamp(5, lo=0, hi=10) == 5
    assert clamp(-3, lo=0, hi=10) == 0
    assert clamp(99, lo=0, hi=10) == 10
'''

#: Marker string that must still be present in the test file after the
#: refactor, proving test_clamp_mean was corrected rather than deleted
#: (ADR-0028 §3, Archetype H2 constraint).
_REQUIRED_TEST_MARKER = (_FIXTURE_DIR / "required_test_marker.txt").read_text().strip()

#: Number of tests the golden workspace must report.
_EXPECTED_PASS_COUNT = (_FIXTURE_DIR / "expected_test_count.txt").read_text().strip()


def _seed_workspace(workspace: Path) -> None:
    """Copy the fixture lib/, app/ and tests/ into *workspace*."""
    skip_dirs = {"__pycache__", ".git", ".venv"}
    for subdir in ("lib", "app", "tests"):
        src = _FIXTURE_DIR / subdir
        dst = workspace / subdir
        for file in src.rglob("*"):
            if file.is_file() and not any(part in skip_dirs for part in file.parts):
                rel = file.relative_to(src)
                dst_file = dst / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                dst_file.write_text(file.read_text())


def _clear_caches(workspace: Path) -> None:
    """Strip ``__pycache__`` and pytest caches so re-runs are deterministic."""
    for cache in workspace.rglob("__pycache__"):
        if cache.is_dir():
            shutil.rmtree(cache, ignore_errors=True)
    pcache = workspace / ".pytest_cache"
    if pcache.exists():
        shutil.rmtree(pcache, ignore_errors=True)


def _run_pytest(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python -m pytest -q`` inside *workspace* and capture output."""
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=workspace,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _test_clamp_mean_is_defined(workspace: Path) -> bool:
    """Return True iff ``test_clamp_mean`` is a top-level function in the test file.

    Uses ``ast`` (not a string grep) so a commented-out or docstring
    mention does not masquerade as a definition -- the constraint is that
    the test is genuinely still present and collectable.
    """
    test_file = workspace / "tests" / "test_stats.py"
    try:
        tree = ast.parse(test_file.read_text())
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "test_clamp_mean"
        for node in ast.walk(tree)
    )


@pytest.mark.benchmark
def test_refactor_api_with_constraints(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Hard-tier shape (ADR-0028 H2):

        1. **Pre-condition** -- seed the half-migrated workspace and run
           ``python -m pytest``.  It MUST fail with ``TypeError`` because
           the three app callers (and the direct call in
           ``test_clamp_mean``) still use positional arguments while
           ``clamp`` is keyword-only.
        2. **Gold fix** -- migrate every positional caller to the
           keyword-only form: ``app/calc.py``, ``app/stats.py``,
           ``app/ui.py`` and the direct call in ``tests/test_stats.py``.
           The ``clamp`` signature itself is unchanged.
        3. **Post-condition** -- re-run ``python -m pytest``.  It MUST
           exit 0 with exactly the expected passing-test count, AND
           ``test_clamp_mean`` must still be defined (proving the test
           was corrected, not deleted to green the suite).
    """
    _seed_workspace(benchmark_workspace)

    # --- Pre-condition: the seeded workspace fails with TypeError. --------
    bad = _run_pytest(benchmark_workspace)
    combined = bad.stdout + bad.stderr
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    assert "TypeError" in combined, (
        f"task {TASK.name}: expected TypeError from the keyword-only "
        f"signature; got stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )

    # --- Gold fix: migrate every positional caller to keyword-only. -------
    (benchmark_workspace / "app" / "calc.py").write_text(GOLDEN_CALC)
    (benchmark_workspace / "app" / "stats.py").write_text(GOLDEN_STATS)
    (benchmark_workspace / "app" / "ui.py").write_text(GOLDEN_UI)
    (benchmark_workspace / "tests" / "test_stats.py").write_text(GOLDEN_TEST_STATS)
    _clear_caches(benchmark_workspace)

    # --- Post-condition: suite is green and the constrained test remains. --
    good = _run_pytest(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
    assert f"{_EXPECTED_PASS_COUNT} passed" in good.stdout, (
        f"task {TASK.name}: expected {_EXPECTED_PASS_COUNT} passing tests; "
        f"got stdout={good.stdout!r}"
    )
    assert _test_clamp_mean_is_defined(benchmark_workspace), (
        f"task {TASK.name}: the constraint test test_clamp_mean must still "
        f"be defined after the refactor (it must be corrected, not deleted). "
        f"Marker was {_REQUIRED_TEST_MARKER!r}."
    )
