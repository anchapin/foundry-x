"""Benchmark task: add missing docstrings to functions in a Python module."""

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
    tags=["documentation", "comprehension"],
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
    (benchmark_workspace / "calculator.py").write_text(
        (fixture_dir / "calculator.py").read_text()
    )
    (benchmark_workspace / "calculator.py").write_text(GOLDEN_DOCUMENTED)

    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("calculator", benchmark_workspace / "calculator.py")
    if spec is None or spec.loader is None:
        pytest.fail(f"task {TASK.name}: could not load calculator module")
    module = importlib.util.module_from_spec(spec)
    sys.modules["calculator"] = module

    try:
        spec.loader.exec_module(module)
    except Exception as e:
        pytest.fail(f"task {TASK.name}: module import failed: {e}")

    functions = [("add", module.add), ("subtract", module.subtract), ("multiply", module.multiply)]

    for fn_name, fn in functions:
        doc = fn.__doc__
        assert doc is not None and doc.strip(), (
            f"task {TASK.name}: {fn_name} has no docstring"
        )
        assert len(doc.strip()) > 10, (
            f"task {TASK.name}: {fn_name} docstring too short: {doc!r}"
        )

    import inspect
    for fn_name, fn in functions:
        help_output = inspect.getdoc(fn)
        assert help_output is not None and help_output.strip(), (
            f"task {TASK.name}: help({fn_name}) returned empty"
        )
