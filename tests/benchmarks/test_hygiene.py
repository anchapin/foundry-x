"""Hygiene tests for the benchmark suite (issue #113).

The Critic gate (ADR-0004) selects and runs benchmarks via
``@pytest.mark.benchmark`` and assumes three contract invariants the suite
previously did not assert. A drift in any of them would silently break the
gate without a single test failing:

1. **Marker coverage.** A new task forgets ``@pytest.mark.benchmark`` and
   stops being selected. ``pyproject.toml:40-42`` registers the marker but
   emits only a ``PytestUnknownMarkWarning`` on missing use, not a failure
   (benchmarks/README.md:43-57).
2. **No network.** A task imports ``requests`` / ``httpx`` / ``urllib`` and
   violates the local-first, model-agnostic principle
   (benchmarks/README.md:85-89, PHILOSOPHY.md §5).
3. **Fixture existence.** A task references a fixture directory that does
   not exist. ``benchmarks/conftest.py:63-69`` does raise at task-run time,
   but only after the task body executes; surface the mistake at collection
   time instead.

Scope notes:

- The marker-coverage check scopes to ``benchmarks/tasks/`` (issue #108
  moved the smoke canary here so the in-process registry can discover it
  via its ``TASK`` attribute). The infrastructure tests under
  ``benchmarks/test_workspace_fixture.py`` deliberately lack the marker
  because they are not benchmark tasks -- they exercise the
  ``benchmark_workspace`` fixture contract itself.
- The hygiene tests carry ``@pytest.mark.benchmark`` per AGENTS.md hard
  rule, but the marker-coverage assertion does not apply to them because
  they live under ``tests/`` (out of scope per the issue acceptance).
"""

# ruff: noqa: E402  -- pytest_plugins must precede imports (pytest contract).
from __future__ import annotations

# pytester is an opt-in plugin starting in pytest 8.x; declare it locally so
# the marker-coverage test can spawn a subprocess collection without leaking
# the fixture into the rest of the suite (tests/benchmarks/test_hygiene.py is
# the only consumer).
pytest_plugins = ["pytester"]

import ast
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS_DIR = REPO_ROOT / "benchmarks"
TASKS_DIR = BENCHMARKS_DIR / "tasks"
FIXTURES_DIR = BENCHMARKS_DIR / "fixtures"

#: Modules that perform network I/O. Tasks must not import them
#: (benchmarks/README.md "No network"; PHILOSOPHY.md §5).
FORBIDDEN_IMPORT_MODULES: tuple[str, ...] = ("requests", "httpx", "urllib")

#: Matches ``import <forbidden>`` and ``from <forbidden>[.x] import ...``
#: at the top of a line. ``from foo.urllib import x`` does NOT match
#: because the top-level module must be forbidden itself.
_FORBIDDEN_IMPORT_RE = re.compile(
    r"^\s*(?:import\s+({mods})|from\s+({mods})(?:\.\w+)?\s+import)\b".format(
        mods="|".join(re.escape(m) for m in FORBIDDEN_IMPORT_MODULES),
    ),
    re.MULTILINE,
)


# --- helpers ---------------------------------------------------------------


def _is_benchmark_decorator(node: ast.expr) -> bool:
    """Return True if ``node`` is ``@pytest.mark.benchmark`` (with or without a call)."""
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


def _function_markers(py_file: Path) -> dict[str, set[str]]:
    """Return ``{function_name: {marker_name, ...}}`` for ``test_*`` functions in ``py_file``.

    AST parsing keeps the assertion independent of pytest's report
    formatting -- we verify the source carries the marker, not the
    collected Item's runtime marker set.
    """
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith(
            "test_"
        ):
            markers: set[str] = set()
            for deco in node.decorator_list:
                if _is_benchmark_decorator(deco):
                    markers.add("benchmark")
            out[node.name] = markers
    return out


def _benchmark_target_files() -> list[Path]:
    """Files that contain benchmark tasks: every ``benchmarks/tasks/test_*.py``.

    The smoke canary lives under ``benchmarks/tasks/test_smoke.py`` since
    issue #108 moved it there so the in-process registry can harvest its
    ``TASK`` attribute; no separate path is required.
    """
    return sorted(TASKS_DIR.glob("test_*.py"))


def _declared_benchmark_tasks() -> list[tuple[str, Path]]:
    """Import each task module and return ``(task.name, source_path)`` for every ``TASK``."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    out: list[tuple[str, Path]] = []
    for task_file in sorted(TASKS_DIR.glob("test_*.py")):
        module_name = f"benchmarks.tasks.{task_file.stem}"
        module = __import__(module_name, fromlist=["TASK"])
        task = getattr(module, "TASK", None)
        if task is None:
            continue
        out.append((task.name, task_file))
    return out


# --- the three hygiene tests ----------------------------------------------


@pytest.mark.benchmark
def test_every_benchmark_test_carries_marker(pytester: pytest.Pytester) -> None:
    """Marker coverage: every benchmark task carries ``@pytest.mark.benchmark``.

    Uses pytester's ``runpytest_subprocess`` to collect the task set the same
    way ``uv run pytest benchmarks/`` would, then walks each collected
    nodeid back to its source file and parses the AST to verify the marker
    is present on the matching function. AST parsing keeps the assertion
    independent of pytest's report formatting.
    """
    targets = _benchmark_target_files()
    assert targets, "no benchmark task files found under benchmarks/tasks/"

    # ``--collect-only -q`` emits one nodeid per line:
    #   benchmarks/tasks/test_sort_a_list.py::test_sort_a_list
    result = pytester.runpytest_subprocess(
        *[str(p) for p in targets],
        "--collect-only",
        "-q",
        "--no-header",
    )
    nodeids = [line.strip() for line in result.stdout.lines if "::" in line and "<" not in line]
    assert nodeids, (
        "pytester collected no tests under the benchmark task files.\n"
        f"Targets: {[str(p) for p in targets]}\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )

    missing: list[str] = []
    for nodeid in nodeids:
        path_str, _, test_name = nodeid.partition("::")
        py_file = Path(path_str)
        if not py_file.is_absolute():
            py_file = (REPO_ROOT / path_str).resolve()
        # Parametrised tests generate nodeids like
        # ``test_fix_import_error[missing_module]``; strip the bracketed
        # suffix so the AST lookup finds the bare function definition.
        bare_name = test_name.split("[", 1)[0]
        markers = _function_markers(py_file).get(bare_name, set())
        if "benchmark" not in markers:
            missing.append(f"{nodeid} (markers present: {sorted(markers) or 'none'})")

    assert not missing, (
        "Benchmark tasks must each carry @pytest.mark.benchmark. Missing:\n  - "
        + "\n  - ".join(missing)
    )


@pytest.mark.benchmark
def test_benchmark_tasks_have_no_network_imports() -> None:
    """No-network invariant: ``benchmarks/tasks/*.py`` must not import network clients.

    Greps each task module for ``import requests`` / ``import httpx`` /
    ``import urllib`` (and the ``from`` variants). A match fails the suite
    immediately -- a benchmark task must be self-contained and reproducible
    offline (benchmarks/README.md "No network"; PHILOSOPHY.md §5).

    Scope: this check is restricted to ``benchmarks/tasks/``. Shared helpers
    (``benchmarks/support.py``, ``benchmarks/conftest.py``, the
    ``BenchmarkTask`` schema) are out of scope and may evolve to wrap a
    remote endpoint provided the Runner mediates it.
    """
    offenders: list[str] = []
    for task_file in sorted(TASKS_DIR.glob("*.py")):
        text = task_file.read_text(encoding="utf-8")
        for match in _FORBIDDEN_IMPORT_RE.finditer(text):
            forbidden = match.group(1) or match.group(2)
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{task_file.name}:{line_no}: {forbidden}")

    assert not offenders, (
        "Benchmark tasks must not import network clients "
        "(benchmarks/README.md 'No network'). Forbidden imports found:\n  - "
        + "\n  - ".join(offenders)
    )


@pytest.mark.benchmark
def test_every_benchmark_task_has_matching_fixture_directory() -> None:
    """Fixture-existence invariant: every non-smoke ``BenchmarkTask`` has a fixture dir.

    Imports each ``benchmarks/tasks/test_*.py`` module, reads its
    ``TASK = BenchmarkTask(...)`` instance, and asserts the named fixture
    directory exists under ``benchmarks/fixtures/``. A missing dir would
    produce a confusing ``FileNotFoundError`` deep inside ``conftest.py``
    at task-run time (conftest.py:63-69); surfacing the mismatch at
    collection time keeps the failure mode local and unambiguous.

    The smoke canary (``difficulty_tier="smoke"``) is excluded: by design
    it requires no fixture data and no agent invocation -- it is a
    static check that the plumbing is wired up. The ``difficulty_tier``
    field is the right discriminator because the smoke tier was added
    for exactly this shape of task (benchmarks/models.py:36-37).
    """
    declared = _declared_benchmark_tasks()
    assert declared, "no BenchmarkTask instances declared under benchmarks/tasks/"

    missing: list[str] = []
    for task_name, source_file in declared:
        # Import the module to read the task's ``difficulty_tier``.
        # Cheap: module is already cached in sys.modules by
        # ``_declared_benchmark_tasks``.
        module = __import__(f"benchmarks.tasks.{source_file.stem}", fromlist=["TASK"])
        task = module.TASK
        if task.difficulty_tier == "smoke":
            # Smoke canary -- no fixture data by design (issue #27).
            continue
        fixture_path = FIXTURES_DIR / task_name
        if not fixture_path.is_dir():
            missing.append(
                f"{source_file.relative_to(REPO_ROOT)}: "
                f"fixture directory missing: {fixture_path.relative_to(REPO_ROOT)}"
            )

    assert not missing, (
        "Benchmark tasks reference fixture directories that do not exist under "
        "benchmarks/fixtures/:\n  - "
        + "\n  - ".join(missing)
        + f"\n\nDeclared tasks: {[name for name, _ in declared]}"
    )
