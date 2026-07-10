"""Benchmark task: fix a NameError caused by a missing stdlib import.

Multi-step debugging target (issue #110). The fixture seeds two files into
the workspace -- ``module.py`` (broken) and ``main.py`` (the entry point
that calls ``module.run()``). The agent must read both, identify the
missing stdlib symbol, and write a corrected version of ``module.py``
that adds the required import. The corrected ``module.py`` is then
executed as ``python -m module`` and must return rc=0 with the expected
stdout.

This is the first ``difficulty_tier='medium'`` task in the benchmark
suite; the five preceding tasks are all ``easy`` and single-step. The
medium tier is the smallest credible shape that exercises a real
read-then-edit-then-test loop:

    1. read ``main.py`` to discover the entry point contract,
    2. read ``module.py`` to find the broken symbol,
    3. add the missing import and re-run ``python -m module``.

The second parameterized case -- ``missing_symbol`` -- escalates the
disambiguation step: a ``candidates.txt`` file lists plausible stdlib
symbols and the agent must pick the right one (the seed uses
``pathlib.Path`` without importing ``pathlib``). This is the smallest
shape that probes whether the agent can use a provided candidate list
to localize the fix.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="fix_import_error",
    description=(
        "Diagnose a NameError caused by a missing stdlib import in module.py, "
        "given main.py and (optionally) a candidates list, then write a "
        "corrected module.py whose `python -m module` invocation exits 0."
    ),
    prompt=(
        "Two files are in the workspace: module.py (broken) and main.py "
        "(entry point that calls module.run()). module.py raises "
        "NameError because a stdlib symbol is used without being imported. "
        "Read both files, identify the missing import, and rewrite "
        "module.py with the missing import added at the top so that "
        "'python -m module' returns rc=0 and prints the expected output. "
        "If a candidates.txt file is provided, use it to localize the "
        "missing symbol."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the fix, 'python -m module' returns rc=0 with stdout equal "
        "to fixtures/fix_import_error/<case>/expected_stdout.txt."
    ),
    timeout_seconds=30,
    tags=["debugging", "imports", "stdlib", "multi-file"],
)

# Per-case fixture directory name, golden ``module.py`` content (the
# corrected module -- missing import added at the top), and expected stdout.
CASES: dict[str, dict[str, str]] = {
    "missing_module": {
        "fixture": "missing_module",
        "corrected_module": (
            "import json\n"
            "\n"
            "\n"
            "def run() -> None:\n"
            '    print(json.dumps({"ok": True}))\n'
            "\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    run()\n"
        ),
    },
    "missing_symbol": {
        "fixture": "missing_symbol",
        "corrected_module": (
            "from pathlib import Path\n"
            "\n"
            "\n"
            "def run() -> None:\n"
            '    print(Path("/tmp").name)\n'
            "\n"
            "\n"
            'if __name__ == "__main__":\n'
            "    run()\n"
        ),
    },
}


def _run_module(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python -m module`` inside ``workspace`` and capture all output."""
    return subprocess.run(
        [sys.executable, "-m", "module"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("case_id", sorted(CASES), ids=lambda c: c)
@pytest.mark.benchmark
def test_fix_import_error(benchmark_workspace: Path, case_id: str) -> None:
    """Deterministic pass/fail check for TASK.

    Two-step shape (medium tier):

        1. Pre-condition: ``python -m module`` against the seeded (broken)
           ``module.py`` MUST fail with NameError. This proves the bug is
           real and not silently masked by the fixture; a regression that
           accidentally ships a working fixture would fail here.
        2. Post-condition: after applying the golden fix (the corrected
           ``module.py`` -- missing import added at the top), ``python -m
           module`` MUST return rc=0 and print the expected stdout.
    """
    case = CASES[case_id]
    fixture_dir = Path(__file__).parent.parent / "fixtures" / TASK.name / case["fixture"]

    # Seed the workspace from the static fixture. The fixture file itself
    # is never modified; only the per-test workspace copy is.
    (benchmark_workspace / "module.py").write_text((fixture_dir / "module.py").read_text())
    (benchmark_workspace / "main.py").write_text((fixture_dir / "main.py").read_text())
    candidates_path = fixture_dir / "candidates.txt"
    if candidates_path.exists():
        (benchmark_workspace / "candidates.txt").write_text(candidates_path.read_text())

    expected_stdout = (fixture_dir / "expected_stdout.txt").read_text()

    # Pre-condition: the seeded module is genuinely broken.
    bad = _run_module(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}/{case_id}: seeded module.py must fail without "
        f"the fix; got rc={bad.returncode} stdout={bad.stdout!r} "
        f"stderr={bad.stderr!r}"
    )
    assert "NameError" in bad.stderr, (
        f"task {TASK.name}/{case_id}: expected NameError in stderr; " f"got stderr={bad.stderr!r}"
    )

    # Apply the golden fix: overwrite module.py in the workspace with the
    # corrected version (missing import added at the top).
    (benchmark_workspace / "module.py").write_text(case["corrected_module"])

    # Post-condition: the corrected module runs end-to-end.
    good = _run_module(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}/{case_id}: corrected module.py must exit 0; "
        f"got rc={good.returncode} stderr={good.stderr!r}"
    )
    assert good.stdout == expected_stdout, (
        f"task {TASK.name}/{case_id}: stdout mismatch "
        f"(got {good.stdout!r}, expected {expected_stdout!r})"
    )
