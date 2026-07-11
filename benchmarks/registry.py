"""In-process registry of every ``BenchmarkTask`` declared under ``benchmarks/tasks/``.

Per issue #108 and ADR-0006, ``BenchmarkTask`` is the boundary model between
the benchmark suite and the Critic. This module lets the Critic enumerate
tasks in-process -- without spawning ``pytest`` and parsing JUnit XML -- so
a future Critic iteration can diff verdicts session-to-session and tighten
the cycle-time KPI loop (ADR-0004).

Why this exists
---------------
Before #108 the Critic's only way to enumerate benchmark tasks was to spawn
a ``pytest -m benchmark`` subprocess and parse its output. That made
regression detection O(pytest startup) per task and forced the Critic to
parse JUnit XML. The registry flips the contract: tasks are first-class
Python values, the Critic iterates them, and pytest becomes a downstream
executor rather than the source of truth.

The scan is dynamic: every file matching ``benchmarks/tasks/test_*.py`` is
imported, and its module-level ``TASK = BenchmarkTask(...)`` attribute is
harvested. Adding a new task requires only a new file under
``benchmarks/tasks/`` with the standard ``TASK`` attribute; the registry
picks it up automatically.

Pytest-free at the module boundary
----------------------------------
This module deliberately avoids ``import pytest`` at top level so the
Critic can use it in-process without dragging in pytest's plugin /
collection lifecycle. Task modules are still loaded via ``importlib`` --
which transitively imports pytest inside each task module -- but the
registry's own import graph stays clean. The Critic therefore pays the
pytest import cost only when it actually asks for the task list, not
when it merely imports ``foundry_x.evolution.critic``.
"""

from __future__ import annotations

import importlib
from pathlib import Path

from benchmarks.models import BenchmarkTask

#: Root directory containing the benchmark task modules
#: (``benchmarks/tasks/``). Exposed so callers (and tests) can iterate
#: the same files the registry scans without re-implementing the glob.
TASKS_DIR: Path = Path(__file__).resolve().parent / "tasks"


def load_all_tasks() -> list[BenchmarkTask]:
    """Return every ``BenchmarkTask`` declared under ``benchmarks/tasks/``.

    Each task module is expected to expose a module-level ``TASK`` attribute
    (a ``BenchmarkTask`` instance, ADR-0006). Modules without one are
    skipped silently: they might be utility helpers that happen to live
    under ``tasks/`` for organisational reasons.

    Returns:
        A fresh list of ``BenchmarkTask`` instances, sorted by the
        underlying filename (which matches the lexicographic order pytest
        uses to discover the same files via ``testpaths``).

    Notes:
        Importing a task module executes its top-level code -- including
        ``import pytest`` inside the module -- but Python caches the
        module in ``sys.modules``, so repeated calls are O(1) per module
        after the first.
    """
    tasks: list[BenchmarkTask] = []
    for task_file in sorted(TASKS_DIR.glob("test_*.py")):
        module_name = f"benchmarks.tasks.{task_file.stem}"
        module = importlib.import_module(module_name)
        task = getattr(module, "TASK", None)
        if task is None:
            continue
        tasks.append(task)
    return tasks


def get_task(name: str) -> BenchmarkTask | None:
    """Return the ``BenchmarkTask`` named *name*, or ``None`` if undeclared.

    Lookup walks ``load_all_tasks()`` linearly. The registry is small
    (single-digit to low-double-digit task count), so a sorted structure
    or hash would be premature optimisation -- the linear scan keeps the
    contract obvious and matches the human mental model of "the list of
    declared tasks".

    Args:
        name: The benchmark task id (snake_case; matches ``TASK.name``).

    Returns:
        The matching ``BenchmarkTask`` or ``None`` if no task declares
        that name. The match is exact -- partial, glob, and regex matching
        are out of scope for the Critic's regression loop.
    """
    for task in load_all_tasks():
        if task.name == name:
            return task
    return None
