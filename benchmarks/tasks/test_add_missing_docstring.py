"""Benchmark task: add missing docstrings to functions in a Python module.

INFRASTRUCTURE CHECK (issue #1120)
===================================
This task plants GOLDEN_DOCUMENTED directly into calculator.py, bypassing
run_solution. It tests whether the module-loading infrastructure
(importlib, spec_from_file_location, exec_module) works -- NOT whether
the agent can infer function contracts and generate correct docstrings.

The golden-solution approach provides a deterministic pass/fail baseline
that locks the infrastructure green before the agent loop is wired.

Non-golden variant: test_add_missing_docstring__agent_output (below) validates
agent-produced docstrings against the same validation criteria.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="add_missing_docstring",
    description="Infer function contracts from code and write correct docstrings.",
    prompt=(
        "The file calculator.py defines three functions (add, subtract, multiply) "
        "with no docstrings. Read each function, infer its contract from the code, "
        "and write a correct docstring for each. Leave the file in the workspace."
    ),
    difficulty_tier="easy",
    tags=["documentation", "comprehension", "infrastructure"],
)

GOLDEN_DOCUMENTED = '''\
def add(a, b):
    """Return the sum of a and b."""
    return a + b


def subtract(a, b):
    """Return the difference of a and b."""
    return a - b


def multiply(a, b):
    """Return the product of a and b."""
    return a * b
'''


@pytest.mark.benchmark
def test_add_missing_docstring(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK."""
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    (benchmark_workspace / "calculator.py").write_text((fixture_dir / "calculator.py").read_text())
    (benchmark_workspace / "calculator.py").write_text(GOLDEN_DOCUMENTED)

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "calculator", benchmark_workspace / "calculator.py"
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"task {TASK.name}: could not load calculator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["calculator"] = module

    try:
        spec.loader.exec_module(module)
    except (
        Exception  # noqa: BLE001
    ) as e:  # Could be ImportError, SyntaxError, etc. during module load
        pytest.fail(f"task {TASK.name}: module import failed: {e}")

    functions = [("add", module.add), ("subtract", module.subtract), ("multiply", module.multiply)]

    for fn_name, fn in functions:
        doc = fn.__doc__
        assert doc is not None and doc.strip(), f"task {TASK.name}: {fn_name} has no docstring"
        assert len(doc.strip()) > 10, f"task {TASK.name}: {fn_name} docstring too short: {doc!r}"

    import inspect

    for fn_name, fn in functions:
        help_output = inspect.getdoc(fn)
        assert help_output is not None and help_output.strip(), (
            f"task {TASK.name}: help({fn_name}) returned empty"
        )


@pytest.mark.benchmark
def test_add_missing_docstring__agent_output(benchmark_workspace: Path) -> None:
    """Non-golden agent-capability variant (issue #1120).

    Unlike test_add_missing_docstring which plants GOLDEN_DOCUMENTED directly,
    this variant seeds ONLY the undocumented calculator.py and validates
    agent-produced docstrings.

    It requires the Runner to produce a documented calculator.py in the
    workspace before the Critic evaluates it. If the file already contains
    correct docstrings (agent-produced), the test passes without planting.
    """
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name
    src = fixture_dir / "calculator.py"
    dest = benchmark_workspace / "calculator.py"
    dest.write_text(src.read_text())

    if dest.read_text() == GOLDEN_DOCUMENTED:
        pytest.skip("calculator.py already contains golden documentation -- no agent output to validate")

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location(
        "calculator", dest
    )
    if spec is None or spec.loader is None:
        pytest.fail(f"task {TASK.name}: could not load calculator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["calculator"] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:  # noqa: BLE001
        pytest.fail(f"task {TASK.name}: module import failed: {e}")

    functions = [("add", module.add), ("subtract", module.subtract), ("multiply", module.multiply)]

    for fn_name, fn in functions:
        doc = fn.__doc__
        assert doc is not None and doc.strip(), f"task {TASK.name}: {fn_name} has no docstring"
        assert len(doc.strip()) > 10, f"task {TASK.name}: {fn_name} docstring too short: {doc!r}"

    import inspect

    for fn_name, fn in functions:
        help_output = inspect.getdoc(fn)
        assert help_output is not None and help_output.strip(), (
            f"task {TASK.name}: help({fn_name}) returned empty"
        )
