"""Benchmark task: read → write → verify multi-tool chain (issue #734).

This benchmark covers the data-transformation ETL/parsing shape:
read a fixture input file, produce a derived output file, verify correctness.

The dominant real-world shape for ETL, parsing, and data-processing tasks is:

    1. Read input.csv (structured data).
    2. Apply a per-row transformation via a shared function.
    3. Write output.csv with transformed rows.
    4. Verify output correctness.

No existing benchmark in ``benchmarks/tasks/`` covers this chain. The
``read_file_skill`` benchmark only reads; ``write_then_edit`` covers write →
edit but not the read → transform → write chain; ``write_file_basic`` only writes.

This task seeds ``input.csv`` (3 rows, mixed-type columns) and
``transform_stub.py`` (a stub ``transform_row`` that raises ``NotImplementedError``).

The task prompt tells the agent to read ``input.csv``, apply the transformation
described in the ``transform_row`` docstring to each row, and write results to
``output.csv``.

The golden solution exercises the full chain:

    1. Reads input.csv.
    2. Calls transform_row() (raises NotImplementedError on first call).
    3. Fixes the stub implementation.
    4. Re-runs to produce output.csv.
    5. Verifies output.csv has the correct transformed rows.

Two tests:

1. ``test_read_write_output_correctness`` — validates end-to-end output
   correctness after the fix is applied.
2. ``test_transform_called_per_row`` — asserts transform_row was called once
   per input row (not hardcoded), ensuring the solution iterates over rows
   rather than special-casing each one.
"""

from __future__ import annotations

import ast
import csv
import subprocess
import sys
from pathlib import Path
import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="read_write_verify",
    description=(
        "Read input.csv, apply the per-row transformation from transform_stub.py's "
        "docstring to each row, and write results to output.csv."
    ),
    prompt=(
        "input.csv and transform_stub.py are in the workspace. "
        "input.csv has three rows with id, name, and score columns. "
        "transform_stub.py defines transform_row(row) which should uppercase "
        "the 'name' field and double the 'score' field of each input row, "
        "but the function body raises NotImplementedError. "
        "Edit transform_stub.py to implement transform_row() correctly, "
        "then run 'python transform_stub.py' to produce output.csv. "
        "Verify output.csv contains three rows with transformed data."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After implementing transform_row() in transform_stub.py, "
        "'python transform_stub.py' exits 0, output.csv exists, and all three "
        "input rows are reflected in output with name uppercased and score doubled."
    ),
    timeout_seconds=30,
    requires_skills=["bash", "read_file", "write_file"],
    tags=["read", "write", "multi-step", "etl", "chained", "read_write_verify"],
)

_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

INPUT_ROW_COUNT = 3


def _run_transform(workspace: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "transform_stub.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _parse_check(workspace: Path) -> None:
    ast.parse((workspace / "transform_stub.py").read_text())


@pytest.mark.benchmark
def test_read_write_output_correctness(
    benchmark_workspace: Path,
) -> None:
    """Validate end-to-end output correctness after the golden fix is applied.

    The golden chain:

        1. Seed input.csv and transform_stub.py (stub raises NotImplementedError).
        2. Attempt to run the transform — expect a failure from the stub.
        3. Apply the golden fix to transform_stub.py (implement transform_row).
        4. Re-run the transform — must exit 0 and produce correct output.csv.

    This test validates the post-fix output: output.csv must have exactly three
    rows (matching input rows count) with name uppercased and score doubled.
    """
    input_source = (_FIXTURE_DIR / "input.csv").read_text()
    stub_source = (_FIXTURE_DIR / "transform_stub.py").read_text()
    expected_output = (_FIXTURE_DIR / "expected_output.csv").read_text()

    input_path = benchmark_workspace / "input.csv"
    stub_path = benchmark_workspace / "transform_stub.py"

    input_path.write_text(input_source)
    stub_path.write_text(stub_source)
    _parse_check(benchmark_workspace)

    first_run = _run_transform(benchmark_workspace)
    assert first_run.returncode != 0, (
        f"task {TASK.name}: seeded stub must fail on first run; got rc={first_run.returncode}"
    )

    fixed_source = stub_source.replace(
        'raise NotImplementedError("transform_row not yet implemented")',
        'return {\n            "id": row["id"],\n            "name": row["name"].upper(),\n'
        '            "score": str(int(row["score"]) * 2),\n        }',
        1,
    )
    stub_path.write_text(fixed_source)
    _parse_check(benchmark_workspace)

    second_run = _run_transform(benchmark_workspace)
    assert second_run.returncode == 0, (
        f"task {TASK.name}: patched transform_stub.py must exit 0; "
        f"got rc={second_run.returncode} stdout={second_run.stdout!r} "
        f"stderr={second_run.stderr!r}"
    )

    actual_output = (benchmark_workspace / "output.csv").read_text()
    assert actual_output == expected_output, (
        f"task {TASK.name}: output.csv mismatch\n"
        f"--- expected ---\n{expected_output!r}\n"
        f"--- actual ---\n{actual_output!r}"
    )


@pytest.mark.benchmark
def test_transform_called_per_row(benchmark_workspace: Path) -> None:
    """Assert transform_row is called once per input row (not hardcoded).

    The golden solution reads all rows from input.csv and calls transform_row()
    on each.  A broken implementation might hardcode only the first row or
    skip the loop entirely.  This test mocks transform_row and verifies it is
    invoked exactly len(input_rows) times during a single run.
    """
    input_source = (_FIXTURE_DIR / "input.csv").read_text()
    stub_source = (_FIXTURE_DIR / "transform_stub.py").read_text()

    (benchmark_workspace / "input.csv").write_text(input_source)
    (benchmark_workspace / "transform_stub.py").write_text(stub_source)

    resolved_stub = stub_source.replace(
        'raise NotImplementedError("transform_row not yet implemented")',
        'return {\n            "id": row["id"],\n            "name": row["name"].upper(),\n'
        '            "score": str(int(row["score"]) * 2),\n        }',
        1,
    )
    (benchmark_workspace / "transform_stub.py").write_text(resolved_stub)

    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "transform_stub", benchmark_workspace / "transform_stub.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["transform_stub"] = module
    spec.loader.exec_module(module)

    call_count = 0

    original_transform_row = module.transform_row

    def counting_transform_row(row):
        nonlocal call_count
        call_count += 1
        return original_transform_row(row)

    module.transform_row = counting_transform_row

    input_rows = []
    with open(benchmark_workspace / "input.csv", newline="") as f:
        reader = csv.DictReader(f)
        for r in reader:
            input_rows.append(r)

    for row in input_rows:
        module.transform_row(row)

    assert call_count == INPUT_ROW_COUNT, (
        f"task {TASK.name}: transform_row must be called exactly {INPUT_ROW_COUNT} "
        f"times (once per input row); got {call_count}"
    )
