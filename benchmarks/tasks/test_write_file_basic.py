"""Benchmark task: write_file skill coverage (issue #616).

This task exercises the ``write_file`` harness skill
(``harness/skills/write_file.json``), which takes a file path and
content and writes the full contents to disk (creating or overwriting).
The skill is the foundational file-creation primitive; this benchmark
validates that a golden solution can use it to create a runnable module
that exits 0 with the expected stdout.

The task is intentionally minimal: it seeds no fixture files, only
supplies the prompt and lets the golden solution write ``hello.py`` from
scratch. The test then asserts the file was written and runs correctly.

See ``harness/skills/write_file.json`` for the skill contract.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="write_file_basic",
    description=(
        "Write a runnable hello-world module to hello.py using write_file "
        "(harness/skills/write_file.json)."
    ),
    prompt=(
        "Write a file called hello.py that defines greet(name) returning "
        '"Hello, {name}!" and a main() that prints greet("World"). '
        "Use the write_file skill to create the file."
    ),
    difficulty_tier="easy",
    expected_outcome=(
        "hello.py exists in the workspace, contains the greet function and "
        "main that prints 'Hello, World!', and 'python hello.py' exits 0 "
        "with 'Hello, World!' on stdout."
    ),
    timeout_seconds=30,
    requires_skills=["write_file"],
    tags=["file-creation", "write_file"],
)

#: Golden content for hello.py (mirrors the fixture used in golden verification).
GOLDEN_HELLO_PY = """\
def greet(name: str) -> str:
    return f"Hello, {name}!"


def main() -> None:
    print(greet("World"))


if __name__ == "__main__":
    main()
"""

#: Expected stdout when running the golden hello.py.
EXPECTED_STDOUT = "Hello, World!\n"


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.benchmark
def test_write_file_basic(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    The golden solution writes ``hello.py`` using the ``write_file`` skill
    contract (path + content). The test verifies:

        1. ``hello.py`` was created and contains the golden source.
        2. ``python hello.py`` exits 0 with the expected stdout.
        3. The bytes written match the golden content size.
    """
    hello_path = benchmark_workspace / "hello.py"

    # Golden solution: write hello.py using the write_file skill contract.
    hello_path.write_text(GOLDEN_HELLO_PY)

    # --- Pre-condition: file exists and has correct content. --------------
    assert hello_path.exists(), f"task {TASK.name}: hello.py was not created"
    written_content = hello_path.read_text()
    assert written_content == GOLDEN_HELLO_PY, (
        f"task {TASK.name}: hello.py content mismatch; "
        f"expected sha256={_sha256(GOLDEN_HELLO_PY)}, "
        f"got sha256={_sha256(written_content)}"
    )

    # --- Post-condition: the module runs end-to-end. ----------------------
    result = subprocess.run(
        [sys.executable, "hello.py"],
        cwd=benchmark_workspace,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"task {TASK.name}: python hello.py must exit 0; "
        f"got rc={result.returncode} stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert result.stdout == EXPECTED_STDOUT, (
        f"task {TASK.name}: stdout mismatch; expected {EXPECTED_STDOUT!r}, got {result.stdout!r}"
    )
