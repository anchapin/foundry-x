"""Benchmark task: debug a circular import across two packages (ADR-0028 H1).

This is a ``difficulty_tier='medium'`` task in the suite.  It
implements Archetype H1 ("complex debugging across modules") from
ADR-0028 §3:

    A bug lives at the intersection of two modules.  The symptom is
    visible only when the full stack is exercised, but the root cause is
    in a different module than the one that raises the exception.

The seeded workspace contains two packages, ``pkg_a`` and ``pkg_b``.  Each
imports a name from the other at module top level, forming a circular
import.  The symptom is an ``ImportError`` raised during pytest collection
of ``tests/test_integration.py`` -- but the file the agent must edit to
break the cycle is ``pkg_b/serializers.py``, not the test file and not
``pkg_a/models.py`` (the module whose name appears in the error message).

The agent must:

    1. Run ``python -m pytest`` and read the ``ImportError`` traceback.
    2. Recognise the "partially initialised module" phrasing as a circular
       import, then locate the cycle by reading ``pkg_a/models.py`` and
       ``pkg_b/serializers.py``.
    3. Break the cycle with the deferred-import pattern: move the
       ``from pkg_a.models import DataModel`` line inside ``serialize()``
       so no import-time back-reference exists.
    4. Re-run ``python -m pytest`` and confirm it exits 0.

This satisfies the four hard-tier criteria in ADR-0028 §2:

    * Multi-phase reasoning: the agent forms an initial hypothesis
      ("the test file is wrong"), observes the real cause in the
      traceback, and revises to "mutual top-level import".
    * Cross-module scope: five files across two packages
      (``pkg_a/__init__.py``, ``pkg_a/models.py``, ``pkg_b/__init__.py``,
      ``pkg_b/serializers.py``, ``tests/test_integration.py``).
    * Non-trivial state: the import-time ordering constraint is not
      visible in any single file -- it emerges from the bidirectional
      edge between the two packages.
    * Precise outcome: ``python -m pytest`` return code is the pass/fail
      oracle.

The prompt deliberately does not name ``pkg_b/serializers.py``; the agent
must locate the cycle via the traceback (ADR-0028 §4 "No hint leakage").
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="debug_import_cycle",
    description=(
        "A workspace with two packages pkg_a and pkg_b has a circular "
        "import: pkg_a.models and pkg_b.serializers import from each "
        "other at module top level, so 'python -m pytest' fails with "
        "ImportError during collection. Identify the cycle from the "
        "traceback and break it with the deferred-import pattern so "
        "'python -m pytest' exits 0."
    ),
    prompt=(
        "The workspace contains two Python packages, pkg_a and pkg_b, "
        "plus a test suite under tests/. Running 'python -m pytest' "
        "currently fails during collection with an ImportError mentioning "
        "a 'partially initialised module'.\n"
        "\n"
        "Diagnose the root cause from the traceback, locate the cycle, "
        "and break it so that no import-time back-reference remains "
        "(the standard fix is to move one of the offending top-level "
        "imports inside the function body that actually needs it). "
        "Do not rename or delete any module, and do not modify the test "
        "files.\n"
        "\n"
        "After your fix, 'python -m pytest' must exit 0 with all tests "
        "passing."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After breaking the circular import, 'python -m pytest' exits 0 "
        "and both tests in tests/test_integration.py pass."
    ),
    timeout_seconds=60,
    requires_skills=["bash", "edit_file"],
    tags=["debugging", "import-cycle", "cross-module", "multi-file"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: Golden ``pkg_b/serializers.py`` -- the top-level import is deferred into
#: ``serialize()`` to break the cycle.  No other file changes.
GOLDEN_SERIALIZERS = '''\
"""Serializer for the debug_import_cycle hard-tier fixture (golden).

The circular import is broken by deferring the ``DataModel`` import to
function scope: at import time this module no longer touches
``pkg_a.models``, so there is no import-time back-reference and Python
can initialise both packages in either order.
"""


def serialize(model) -> dict[str, int]:
    """Return a dict representation of *model*.

    The ``DataModel`` import is deferred to runtime to break the
    pkg_a <-> pkg_b import cycle.
    """
    from pkg_a.models import DataModel  # noqa: F401 -- deferred to break cycle

    return {"value": model.value}
'''

#: Number of tests the golden workspace must report (test_round_trip +
#: test_to_serialized).  Used as a secondary assertion so a solution that
#: accidentally skips collection cannot pass.
_EXPECTED_PASS_COUNT = (_FIXTURE_DIR / "expected_test_count.txt").read_text().strip()


def _seed_workspace(workspace: Path) -> None:
    """Copy the fixture packages and tests into *workspace*."""
    skip_dirs = {"__pycache__", ".git", ".venv"}
    for subdir in ("pkg_a", "pkg_b", "tests"):
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


@pytest.mark.benchmark
def test_debug_import_cycle(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Hard-tier shape (ADR-0028 H1):

        1. **Pre-condition** -- seed the two-package workspace and run
           ``python -m pytest``.  It MUST fail during collection with an
           ``ImportError`` caused by the pkg_a <-> pkg_b circular import.
           This proves the symptom is real and reproducible.
        2. **Gold fix** -- apply the golden correction to ONE file:
           ``pkg_b/serializers.py``.  The top-level
           ``from pkg_a.models import DataModel`` is moved inside
           ``serialize()``, breaking the import-time cycle.  No other
           file changes.
        3. **Post-condition** -- re-run ``python -m pytest``.  It MUST
           exit 0 and report exactly the expected number of passing
           tests, proving the cycle is genuinely broken rather than
           masked (e.g. by skipping collection).
    """
    _seed_workspace(benchmark_workspace)

    # --- Pre-condition: the seeded workspace fails with a circular import. ---
    bad = _run_pytest(benchmark_workspace)
    combined = bad.stdout + bad.stderr
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    assert "ImportError" in combined, (
        f"task {TASK.name}: expected ImportError during collection; "
        f"got stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    assert "partially initialized" in combined.lower() or "circular import" in combined.lower(), (
        f"task {TASK.name}: expected a circular-import symptom "
        f"('partially initialized module' / 'circular import'); got combined={combined!r}"
    )

    # --- Gold fix: defer the offending import in pkg_b/serializers.py. -----
    (benchmark_workspace / "pkg_b" / "serializers.py").write_text(GOLDEN_SERIALIZERS)
    _clear_caches(benchmark_workspace)

    # --- Post-condition: the cycle is broken and the suite is green. -------
    good = _run_pytest(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
    assert f"{_EXPECTED_PASS_COUNT} passed" in good.stdout, (
        f"task {TASK.name}: expected {_EXPECTED_PASS_COUNT} passing tests; "
        f"got stdout={good.stdout!r}"
    )
