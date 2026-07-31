"""Benchmark task: debug an invalid state-machine transition (ADR-0028 H1).

This is a ``difficulty_tier='hard'`` task in the suite.  It implements
Archetype H1 ("complex debugging across modules") from ADR-0028 §3:

    A bug lives at the intersection of two modules.  The symptom is
    visible only when the full stack is exercised, but the root cause is
    in a different module than the one that raises the exception.

The seeded workspace contains a state-machine engine (``wfsm/``) and a
workflow runner (``wfrun/``).  The transition table in
``wfsm/machine.py`` incorrectly allows ``PENDING -> DONE``, bypassing
the ``RUNNING`` state.  The symptom is a ``ValueError`` raised by
``wfrun/reporter.py::report_duration`` because ``duration`` is ``None``
(the ``enter_running`` action that sets it was never fired).

The agent must:

    1. Run ``python -m pytest`` and read the ``ValueError`` traceback
       pointing to ``wfrun/reporter.py``.
    2. Form an initial hypothesis (e.g. "reporter should handle None")
       and observe that the real issue is upstream: ``duration`` is
       ``None`` because ``RUNNING`` was bypassed.
    3. Trace the execution path through ``wfrun/workflow.py`` back to
       the transition table in ``wfsm/machine.py``.
    4. Remove the invalid ``PENDING -> DONE`` edge so the fast-path
       workflow falls back to the normal ``PENDING -> RUNNING -> DONE``
       lifecycle.
    5. Re-run ``python -m pytest`` and confirm it exits 0.

This satisfies the four hard-tier criteria in ADR-0028 §2:

    * Multi-phase reasoning: the agent must revise its hypothesis from
      "fix the reporter" to "fix the transition table" after observing
      that the reporter's ``ValueError`` is a downstream symptom.
    * Cross-module scope: six files across three packages
      (``wfsm/states.py``, ``wfsm/machine.py``, ``wfsm/actions.py``,
      ``wfrun/workflow.py``, ``wfrun/reporter.py``,
      ``tests/test_workflow.py``).
    * Non-trivial state: the transition table is shared state defined in
      ``wfsm/machine.py`` but consumed by ``wfrun/workflow.py`` and
      validated indirectly by ``wfrun/reporter.py``.  The bug is an
      emergent property of the interaction between these modules.
    * Precise outcome: ``python -m pytest`` return code is the pass/fail
      oracle.

The prompt deliberately does not name ``wfsm/machine.py`` or the
``PENDING -> DONE`` edge (ADR-0028 §4 "No hint leakage").
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
    name="debug_state_machine_transitions",
    description=(
        "A workflow state machine has an invalid transition: PENDING -> "
        "DONE is allowed, bypassing RUNNING, so 'python -m pytest' fails "
        "with ValueError from the reporter (duration is None). Trace the "
        "error from the reporter through the workflow driver to the "
        "transition table, remove the invalid edge, and confirm the "
        "suite passes."
    ),
    prompt=(
        "The workspace contains a state-machine engine (wfsm/) and a "
        "workflow runner (wfrun/) with tests (tests/). Running "
        "'python -m pytest' currently fails: the fast-path workflow "
        "test raises a ValueError because 'duration is None (the RUNNING "
        "phase was never entered)'.\n"
        "\n"
        "Diagnose the root cause from the traceback. The error appears "
        "in the reporter, but the actual bug is in a different module -- "
        "an invalid state transition allows the workflow to skip a "
        "required state. Trace the execution path, identify the invalid "
        "transition, and fix it so that no state can bypass the required "
        "intermediate state.\n"
        "\n"
        "Do not modify the test files or the reporter's error handling. "
        "The fix must ensure that the fast-path workflow falls back to "
        "the normal lifecycle when the shortcut is correctly rejected.\n"
        "\n"
        "After your fix, 'python -m pytest' must exit 0."
    ),
    difficulty_tier="hard",
    expected_outcome=(
        "After removing the invalid PENDING -> DONE transition, the "
        "fast-path workflow falls back to PENDING -> RUNNING -> DONE, "
        "duration is recorded, and 'python -m pytest' exits 0 with "
        "3 passed."
    ),
    timeout_seconds=300,
    requires_skills=["bash", "grep_search", "edit_file"],
    tags=["debugging", "state-machine", "cross-module", "multi-file", "transitions"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / "hard" / TASK.name

#: Golden ``wfsm/machine.py`` -- the ``PENDING -> DONE`` edge is removed so
#: the only path to completion is ``PENDING -> RUNNING -> DONE``.
GOLDEN_MACHINE = '''\
"""Transition machine for debug_state_machine_transitions (golden).

The invalid ``PENDING -> DONE`` edge has been removed.  The only path
to ``DONE`` is now ``PENDING -> RUNNING -> DONE``, ensuring the
``enter_running`` action always fires and ``duration`` is recorded.
"""

from wfsm.states import State

#: Allowed state transitions.  FIXED: ``PENDING`` can only go to
#: ``RUNNING`` -- the ``DONE`` shortcut that bypassed the ``RUNNING``
#: state (and left ``duration`` unset) has been removed.
TRANSITIONS: dict[State, set[State]] = {
    State.IDLE: {State.PENDING},
    State.PENDING: {State.RUNNING},
    State.RUNNING: {State.DONE, State.FAILED},
    State.DONE: set(),
    State.FAILED: {State.IDLE},
}


def can_transition(current: State, target: State) -> bool:
    """Return True if ``current -> target`` is an allowed transition."""
    return target in TRANSITIONS.get(current, set())
'''

#: Number of tests the golden workspace must report.
_EXPECTED_PASS_COUNT = (_FIXTURE_DIR / "expected_test_count.txt").read_text().strip()


def _seed_workspace(workspace: Path) -> None:
    """Copy the fixture wfsm/, wfrun/ and tests/ into *workspace*."""
    skip_dirs = {"__pycache__", ".git", ".venv"}
    for subdir in ("wfsm", "wfrun", "tests"):
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
def test_debug_state_machine_transitions(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Hard-tier shape (ADR-0028 H1):

        1. **Pre-condition** -- seed the workspace and run
           ``python -m pytest``.  It MUST fail with ``ValueError``
           because the fast-path workflow bypasses ``RUNNING`` (the
           invalid ``PENDING -> DONE`` transition is allowed), leaving
           ``duration`` as ``None``, which ``report_duration`` rejects.
        2. **Gold fix** -- apply the golden correction to ONE file:
           ``wfsm/machine.py``.  The ``PENDING -> DONE`` edge is removed
           from the transition table so the fast-path workflow falls
           back to ``PENDING -> RUNNING -> DONE``.
        3. **Post-condition** -- re-run ``python -m pytest``.  It MUST
           exit 0 and report exactly the expected number of passing
           tests.
    """
    _seed_workspace(benchmark_workspace)

    # --- Pre-condition: the seeded workspace fails with ValueError. -------
    bad = _run_pytest(benchmark_workspace)
    combined = bad.stdout + bad.stderr
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the fix; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    assert "ValueError" in combined, (
        f"task {TASK.name}: expected ValueError from report_duration; "
        f"got stdout={bad.stdout!r} stderr={bad.stderr!r}"
    )
    assert "duration" in combined.lower(), (
        f"task {TASK.name}: expected 'duration' in the error message; got combined={combined!r}"
    )

    # --- Gold fix: remove the invalid PENDING -> DONE transition. ----------
    (benchmark_workspace / "wfsm" / "machine.py").write_text(GOLDEN_MACHINE)
    _clear_caches(benchmark_workspace)

    # --- Post-condition: the transition is fixed and the suite is green. ---
    good = _run_pytest(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} stderr={good.stderr!r}"
    )
    assert f"{_EXPECTED_PASS_COUNT} passed" in good.stdout, (
        f"task {TASK.name}: expected {_EXPECTED_PASS_COUNT} passing tests; "
        f"got stdout={good.stdout!r}"
    )
