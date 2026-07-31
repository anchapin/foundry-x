"""Benchmark task: refactor dict config to dataclass across packages (ADR-0028 H2).

This is a ``difficulty_tier='hard'`` task in the suite.  It extends
Archetype H2 ("multi-file coordinated refactor with constraint") from
ADR-0028 §3 with an additional multi-phase twist: a renamed field that
is masked by the first failure.

The seeded workspace is half-migrated:

    * ``cfglib/config.py`` exposes ``get_config() -> Config`` where
      ``Config`` is a frozen dataclass — the NEW typed contract is in
      place.  The field formerly known as ``max_connections`` has been
      renamed to ``connection_limit``.
    * ``apisvc/client.py``, ``apisvc/router.py`` and ``wrkpool/pool.py``
      still access the config with dict-style ``cfg["key"]`` syntax,
      raising ``TypeError`` at call time.
    * ``cfglib/models.py`` already uses the correct attribute-access
      pattern (reference for the agent).
    * ``tests/test_config.py::test_config_usage`` contains a direct
      dict-style ``cfg["timeout"]`` call — the agent must fix this too.

The multi-phase reasoning twist: after migrating all dict-style callers
to attribute access, ``apisvc/router.py`` STILL fails because it uses
the old field name ``max_connections`` (now ``connection_limit``).  The
initial ``TypeError`` masks this deeper issue, so the agent must observe
a *second* failure after the first round of edits and revise its plan.

This satisfies the four hard-tier criteria in ADR-0028 §2:

    * Multi-phase reasoning: the agent fixes dict->attribute access
      (phase 1), re-runs the suite, discovers ``AttributeError`` from
      the renamed field (phase 2), and must revise its fix.
    * Cross-module scope: six files across four packages
      (``cfglib/config.py``, ``cfglib/models.py``,
      ``apisvc/client.py``, ``apisvc/router.py``,
      ``wrkpool/pool.py``, ``tests/test_config.py``).
    * Non-trivial state: the renamed ``connection_limit`` field is the
      cross-package state that must be discovered and propagated to
      ``apisvc/router.py`` — it is not visible in any single file until
      the ``TypeError`` masking it is resolved.
    * Precise outcome: ``python -m pytest`` return code is the pass/fail
      oracle, plus an AST check that ``test_config_usage`` still exists.

The prompt deliberately does not name ``router.py`` or
``connection_limit`` (ADR-0028 §4 "No hint leakage").
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
    name="refactor_config_to_dataclass",
    description=(
        "cfglib/config.py exposes get_config() -> Config (a frozen "
        "dataclass), but three application callers in apisvc/ and "
        "wrkpool/ still use dict-style cfg['key'] access, and a direct "
        "call in tests/test_config.py does too, so 'python -m pytest' "
        "fails with TypeError. Migrate every dict-style caller to "
        "attribute access, discover the renamed field, and ensure "
        "tests/test_config.py::test_config_usage still exists and passes."
    ),
    prompt=(
        "The workspace contains a config library (cfglib/), an API "
        "service (apisvc/), a worker pool (wrkpool/), and tests "
        "(tests/). The function cfglib/config.py::get_config() has been "
        "migrated from returning a plain dict to returning a typed "
        "Config dataclass. However, several callers still access the "
        "result with dict-style syntax, e.g. cfg['timeout'], so "
        "'python -m pytest' currently fails with TypeError.\n"
        "\n"
        "Find every dict-style caller of get_config() across the "
        "workspace and update each one to attribute access (e.g. "
        "cfg.timeout). Do not change the get_config() function or the "
        "Config class, and do not delete or skip any test -- every "
        "existing test, including tests/test_config.py::test_config_usage, "
        "must still exist and pass after your edits.\n"
        "\n"
        "After your changes, 'python -m pytest' must exit 0."
    ),
    difficulty_tier="hard",
    expected_outcome=(
        "After migrating all dict-style callers to attribute access "
        "(including fixing any renamed fields), 'python -m pytest' "
        "exits 0 with 3 passed, and tests/test_config.py still defines "
        "test_config_usage (it was corrected, not deleted)."
    ),
    timeout_seconds=300,
    requires_skills=["bash", "grep_search", "edit_file"],
    tags=["refactoring", "config-migration", "dataclass", "multi-file", "cross-module"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "hard" / TASK.name

#: Golden ``apisvc/client.py`` -- dict access migrated to attribute access.
GOLDEN_CLIENT = '''\
"""API client for refactor_config_to_dataclass (golden).

Dict-style config access migrated to attribute access.
"""

from cfglib.config import get_config


def fetch_with_timeout() -> tuple[int, int]:
    """Return (timeout, retries) from config."""
    cfg = get_config()
    return cfg.timeout, cfg.retries
'''

#: Golden ``apisvc/router.py`` -- dict access migrated to attribute access
#: AND the old field name ``max_connections`` updated to ``connection_limit``.
GOLDEN_ROUTER = '''\
"""API router for refactor_config_to_dataclass (golden).

Dict-style access migrated to attribute access; old field name
``max_connections`` updated to ``connection_limit``.
"""

from cfglib.config import get_config


def max_connections() -> int:
    """Return the connection limit from config."""
    cfg = get_config()
    return cfg.connection_limit
'''

#: Golden ``wrkpool/pool.py`` -- dict access migrated to attribute access.
GOLDEN_POOL = '''\
"""Worker pool for refactor_config_to_dataclass (golden).

Dict-style config access migrated to attribute access.
"""

from cfglib.config import get_config


def pool_settings() -> tuple[int, int]:
    """Return (timeout, batch_size) from config."""
    cfg = get_config()
    return cfg.timeout, cfg.batch_size
'''

#: Golden test file -- the direct dict-style ``cfg["timeout"]`` call inside
#: ``test_config_usage`` is migrated to attribute access.  The test still
#: EXISTS (the constraint) and still asserts the same behaviour.
GOLDEN_TEST_CONFIG = '''\
"""Config integration tests for refactor_config_to_dataclass (golden).

Direct dict-style ``cfg["timeout"]`` migrated to ``cfg.timeout``.
test_config_usage still exists and passes (the constraint).
"""

from cfglib.config import get_config
from apisvc.client import fetch_with_timeout
from apisvc.router import max_connections
from wrkpool.pool import pool_settings


def test_config_usage() -> None:
    """Integration test exercising all config consumers."""
    timeout, retries = fetch_with_timeout()
    assert timeout == 30
    assert retries == 3

    assert max_connections() == 10

    pool_timeout, batch_size = pool_settings()
    assert pool_timeout == 30
    assert batch_size == 100

    # Migrated to attribute access
    cfg = get_config()
    assert cfg.timeout == 30


def test_config_defaults() -> None:
    """Config dataclass has the expected default values."""
    cfg = get_config()
    assert cfg.timeout == 30
    assert cfg.retries == 3
    assert cfg.connection_limit == 10
    assert cfg.batch_size == 100


def test_config_types() -> None:
    """Config fields have the expected types."""
    cfg = get_config()
    assert isinstance(cfg.timeout, int)
    assert isinstance(cfg.connection_limit, int)
'''

#: Partial router -- dict access fixed to attribute but OLD field name kept.
#: Used to demonstrate the multi-phase failure: after the first round of
#: edits (dict -> attribute) the suite STILL fails because
#: ``cfg.max_connections`` raises ``AttributeError`` (field renamed to
#: ``connection_limit``).
_PARTIAL_ROUTER = '''\
"""API router for refactor_config_to_dataclass (partial fix).

Dict-style access migrated to attribute, but OLD field name
``max_connections`` not yet updated to ``connection_limit``.
"""

from cfglib.config import get_config


def max_connections() -> int:
    """Return the connection limit from config."""
    cfg = get_config()
    return cfg.max_connections  # AttributeError: renamed to connection_limit
'''

#: Marker string that must still be present in the test file after the
#: refactor, proving test_config_usage was corrected rather than deleted
#: (ADR-0028 §3, Archetype H2 constraint).
_REQUIRED_TEST_MARKER = (_FIXTURE_DIR / "required_test_marker.txt").read_text().strip()

#: Number of tests the golden workspace must report.
_EXPECTED_PASS_COUNT = (_FIXTURE_DIR / "expected_test_count.txt").read_text().strip()


def _seed_workspace(workspace: Path) -> None:
    """Copy the fixture cfglib/, apisvc/, wrkpool/ and tests/ into *workspace*."""
    skip_dirs = {"__pycache__", ".git", ".venv"}
    for subdir in ("cfglib", "apisvc", "wrkpool", "tests"):
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


def _test_config_usage_is_defined(workspace: Path) -> bool:
    """Return True iff ``test_config_usage`` is a top-level function in the test file.

    Uses ``ast`` (not a string grep) so a commented-out or docstring
    mention does not masquerade as a definition -- the constraint is that
    the test is genuinely still present and collectable.
    """
    test_file = workspace / "tests" / "test_config.py"
    try:
        tree = ast.parse(test_file.read_text())
    except (OSError, SyntaxError):
        return False
    return any(
        isinstance(node, ast.FunctionDef) and node.name == "test_config_usage"
        for node in ast.walk(tree)
    )


@pytest.mark.benchmark
def test_refactor_config_to_dataclass(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Hard-tier shape (ADR-0028 H2, extended with multi-phase twist):

        1. **Pre-condition** -- seed the half-migrated workspace and run
           ``python -m pytest``.  It MUST fail with ``TypeError`` because
           the three callers (and the direct call in ``test_config_usage``)
           still use dict-style access while ``get_config()`` returns a
           ``Config`` dataclass.
        2. **Phase-1 check** -- apply a partial fix (dict -> attribute
           for client, pool, and the test file, but keep the OLD field
           name in router).  Re-run the suite: it MUST STILL fail, now
           with ``AttributeError`` from the renamed ``max_connections``
           field.  This demonstrates the multi-phase reasoning
           requirement -- the initial ``TypeError`` masked a deeper issue.
        3. **Gold fix** -- apply the full golden correction to all four
           files.  ``connection_limit`` is used in ``router.py``.
        4. **Post-condition** -- re-run ``python -m pytest``.  It MUST
           exit 0 with exactly the expected passing-test count, AND
           ``test_config_usage`` must still be defined (proving the test
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
        f"task {TASK.name}: expected TypeError from dict-style access on "
        f"dataclass; got stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )

    # --- Phase 1: partial fix reveals the renamed field (multi-phase). ----
    (benchmark_workspace / "apisvc" / "client.py").write_text(GOLDEN_CLIENT)
    (benchmark_workspace / "wrkpool" / "pool.py").write_text(GOLDEN_POOL)
    (benchmark_workspace / "apisvc" / "router.py").write_text(_PARTIAL_ROUTER)
    (benchmark_workspace / "tests" / "test_config.py").write_text(GOLDEN_TEST_CONFIG)
    _clear_caches(benchmark_workspace)

    phase1 = _run_pytest(benchmark_workspace)
    phase1_combined = phase1.stdout + phase1.stderr
    assert phase1.returncode != 0, (
        f"task {TASK.name}: partial fix (dict->attribute, old field name) "
        f"must still fail; got rc={phase1.returncode} stdout={phase1.stdout!r}"
    )
    assert "AttributeError" in phase1_combined, (
        f"task {TASK.name}: expected AttributeError from renamed field "
        f"max_connections -> connection_limit; got stdout={phase1.stdout!r}"
    )

    # --- Gold fix: apply the full golden correction to all callers. -------
    (benchmark_workspace / "apisvc" / "router.py").write_text(GOLDEN_ROUTER)
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
    assert _test_config_usage_is_defined(benchmark_workspace), (
        f"task {TASK.name}: the constraint test test_config_usage must still "
        f"be defined after the refactor (it must be corrected, not deleted). "
        f"Marker was {_REQUIRED_TEST_MARKER!r}."
    )
