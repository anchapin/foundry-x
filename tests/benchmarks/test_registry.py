"""Unit tests for the in-process ``BenchmarkTask`` registry (issue #108).

These pin the registry contract the Critic depends on for in-process
iteration and regression diffing:

- ``load_all_tasks()`` returns every ``BenchmarkTask`` declared under
  ``benchmarks/tasks/``.
- ``get_task(name)`` resolves by exact name; partial / glob / regex
  matches return ``None``.
- The registry's import graph is pytest-free at module level so the
  Critic can use it in-process without forcing pytest's plugin /
  collection lifecycle.
- The registry's enumeration agrees with the set of files pytest would
  collect under ``-m benchmark`` -- a drift between the two surfaces
  here.
"""

from __future__ import annotations

import ast
from pathlib import Path

from benchmarks.models import BenchmarkTask
from benchmarks.registry import TASKS_DIR, get_task, load_all_tasks

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_load_all_tasks_returns_one_entry_per_task_file() -> None:
    """``load_all_tasks()`` returns exactly one entry per ``test_*.py`` file.

    The registry's design invariant: every file under
    ``benchmarks/tasks/test_*.py`` declares one ``TASK = BenchmarkTask(...)``
    and contributes exactly one entry to the registry. This test asserts
    the invariant holds -- if a future contributor adds a helper file to
    ``tasks/`` without a ``TASK`` attribute, the registry's silent skip
    would shrink the count and this test would catch the drift.
    """
    task_files = sorted(TASKS_DIR.glob("test_*.py"))
    assert task_files, "no benchmark task files found under benchmarks/tasks/"
    assert len(load_all_tasks()) == len(task_files)


def test_load_all_tasks_returns_only_benchmark_task_instances() -> None:
    """Every entry is a ``BenchmarkTask`` instance with a non-empty name.

    The Critic iterates the list assuming pydantic accessors like
    ``.difficulty_tier`` and ``.name`` work. A future drift that puts a
    non-BenchmarkTask sentinel in the list would explode at first use;
    this test surfaces the mistake at registry-test time instead.
    """
    tasks = load_all_tasks()
    assert tasks, "registry must contain at least one task"
    for task in tasks:
        assert isinstance(task, BenchmarkTask), (
            f"registry entry {task!r} is not a BenchmarkTask instance"
        )
        assert task.name, "every registry entry must declare a non-empty name"


def test_get_task_returns_easy_task_by_name() -> None:
    """``get_task('sort_a_list')`` returns the easy-tier BenchmarkTask.

    Pinned by the issue acceptance criterion: ``sort_a_list`` is the
    canonical ``difficulty_tier='easy'`` task and the simplest end-to-end
    example of the registry's name-based lookup contract.
    """
    task = get_task("sort_a_list")
    assert task is not None
    assert task.name == "sort_a_list"
    assert task.difficulty_tier == "easy"


def test_get_task_returns_smoke_task_by_name() -> None:
    """``get_task('smoke_marker_and_fixture_resolve')`` returns the smoke-tier BenchmarkTask.

    The smoke canary is the cheapest end-to-end check (issue #27). It
    must be discoverable through the same registry path as every other
    task so the Critic's regression diff treats it uniformly.
    """
    task = get_task("smoke_marker_and_fixture_resolve")
    assert task is not None
    assert task.difficulty_tier == "smoke"


def test_get_task_returns_none_for_unknown_name() -> None:
    """``get_task`` returns ``None`` for unknown / empty names -- never raises.

    A regression that turned "name not found" into a ``KeyError`` or
    ``IndexError`` would break every Critic caller that gracefully
    handles missing tasks.
    """
    assert get_task("does_not_exist") is None
    assert get_task("") is None
    assert get_task("Sort_A_List") is None  # case-sensitive


def test_get_task_returns_none_when_only_substring_matches() -> None:
    """``get_task`` is exact-match; partial matches return ``None``.

    Substring / glob / regex matching is out of scope for the Critic's
    regression loop -- callers that want fuzzy matching should iterate
    ``load_all_tasks()`` themselves. This test pins the exact-match
    contract so a future convenience wrapper does not quietly change it.
    """
    assert get_task("sort") is None  # would substring-match "sort_a_list"
    assert get_task("a_list") is None
    assert get_task("*") is None


def test_registry_module_does_not_import_pytest_at_top_level() -> None:
    """``benchmarks/registry.py`` must not ``import pytest`` at module top level.

    The Critic uses this registry in-process (issue #108); pulling pytest
    in as a side effect of importing ``foundry_x.evolution.critic`` would
    force pytest's plugin / collection lifecycle on every Critic
    instantiation. This is a source-level (AST) check so it survives
    refactors that move imports around within the module.
    """
    registry_path = REPO_ROOT / "benchmarks" / "registry.py"
    tree = ast.parse(registry_path.read_text(encoding="utf-8"))

    offenders: list[str] = []
    for node in ast.iter_child_nodes(tree):
        # ``from __future__ import annotations`` is the only top-level
        # import the registry is allowed in addition to stdlib +
        # ``benchmarks.models``.
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "pytest" or alias.name.startswith("pytest."):
                    offenders.append(f"line {node.lineno}: import {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module == "pytest" or node.module.startswith("pytest."):
                offenders.append(f"line {node.lineno}: from {node.module} import ...")

    assert not offenders, (
        "benchmarks/registry.py must not import pytest at module top level "
        "(so the Critic can use it in-process). Offending imports:\n  - " + "\n  - ".join(offenders)
    )


def test_every_registry_entry_maps_to_a_benchmark_test_file() -> None:
    """Every registry entry comes from a file with ``@pytest.mark.benchmark``.

    Pins the invariant: the registry and pytest's ``-m benchmark``
    collection agree on the SET of benchmark tasks. A future contributor
    who deletes the ``@pytest.mark.benchmark`` decorator from a task
    would otherwise see pytest silently drop the task while the registry
    still returns it.
    """
    files_with_benchmark_marker: set[Path] = set()
    files_with_task_attribute: set[Path] = set()

    for task_file in sorted(TASKS_DIR.glob("test_*.py")):
        tree = ast.parse(task_file.read_text(encoding="utf-8"))
        has_benchmark_marker = False
        has_task_attribute = False
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
                "test_"
            ):
                if any(_is_pytest_mark_benchmark(deco) for deco in node.decorator_list):
                    has_benchmark_marker = True
            elif isinstance(node, ast.Assign):
                if (
                    len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == "TASK"
                ):
                    has_task_attribute = True

        if has_benchmark_marker:
            files_with_benchmark_marker.add(task_file)
        if has_task_attribute:
            files_with_task_attribute.add(task_file)

    assert files_with_benchmark_marker == files_with_task_attribute, (
        "Registry and pytest collection disagree on benchmark task files.\n"
        f"  Files with @pytest.mark.benchmark but no TASK: "
        f"{sorted(files_with_benchmark_marker - files_with_task_attribute)}\n"
        f"  Files with TASK but no @pytest.mark.benchmark: "
        f"{sorted(files_with_task_attribute - files_with_benchmark_marker)}"
    )


def test_load_all_tasks_names_are_unique() -> None:
    """``load_all_tasks()`` returns tasks with distinct ``name`` values.

    ``get_task(name)`` resolves by exact match; duplicates would make the
    first match win and silently mask a later redeclaration. The
    pydantic ``name`` validator does not (yet) enforce global uniqueness
    -- this test catches it at the registry boundary instead.
    """
    names = [task.name for task in load_all_tasks()]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    assert not duplicates, f"duplicate task names in registry: {duplicates}"


def _is_pytest_mark_benchmark(node: ast.expr) -> bool:
    """Return True if ``node`` is ``@pytest.mark.benchmark`` (with or without call)."""
    # @pytest.mark.benchmark (attribute chain, no call)
    if isinstance(node, ast.Attribute) and node.attr == "benchmark":
        cur = node.value
        if isinstance(cur, ast.Attribute) and cur.attr == "mark":
            if isinstance(cur.value, ast.Name) and cur.value.id == "pytest":
                return True
    # @pytest.mark.benchmark(...) (call wrapping the chain)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "benchmark":
            cur = func.value
            if isinstance(cur, ast.Attribute) and cur.attr == "mark":
                if isinstance(cur.value, ast.Name) and cur.value.id == "pytest":
                    return True
    return False
