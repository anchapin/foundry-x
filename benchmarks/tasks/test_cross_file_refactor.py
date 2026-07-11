"""Benchmark task: cross-file comprehension with a multi-step fix.

This is the second ``difficulty_tier='medium'`` task in the suite (after
``fix_import_error``, #110).  Unlike the import task -- which edits a
single file -- this task requires the agent to **read** three files and
**edit** two of them in one session:

    1. read ``text_stats.py`` (library) -- discover that ``summarize``
       calls a missing ``_format_line`` helper,
    2. read ``driver.py`` (CLI entry point) -- understand the public
       contract and how ``summarize`` is consumed,
    3. read ``test_text_stats.py`` (tests) -- see that the failing test
       also carries a wrong expected value,
    4. edit ``text_stats.py`` -- add the ``_format_line`` helper whose
       format is documented in ``summarize``'s docstring,
    5. edit ``test_text_stats.py`` -- correct the expected string to
       match the documented ``"words=<N>, chars=<M>"`` format.

The improvement-rate KPI (PRD §5) weights multi-step capability above
trivial transforms.  Without a cross-file shape that exercises the
read-multiple / edit-multiple loop, a regression that breaks that loop
passes the suite silently (issue #176 motivation).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="cross_file_refactor",
    description=(
        "Read a library module, a CLI driver, and a test file; add the "
        "missing helper to the library and fix the test's expected value "
        "so the full test suite passes."
    ),
    prompt=(
        "Three files are in the workspace:\n"
        "  - text_stats.py: a library with word_count, char_count, and "
        "summarize. The summarize function calls _format_line, which does "
        "not exist yet.\n"
        "  - driver.py: a CLI entry point that imports summarize and prints "
        "a summary.\n"
        "  - test_text_stats.py: a test file with one passing test "
        "(test_word_count) and one failing test (test_summarize).\n"
        "\n"
        "Read all three files. Implement _format_line in text_stats.py so "
        "that summarize returns the format documented in its docstring. "
        "Then fix the expected value in test_text_stats.py so that the "
        "test matches the documented format. After both fixes, "
        "'python -m pytest -q' must exit 0."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the fix, 'python -m pytest -q' returns rc=0 (both tests "
        "pass) and 'python driver.py' prints the expected stdout."
    ),
    timeout_seconds=30,
    requires_skills=["bash"],
    tags=["comprehension", "multi-file"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: Golden ``text_stats.py`` -- the missing ``_format_line`` helper added.
GOLDEN_LIBRARY = """\
\"\"\"Text statistics helpers for summarising prose.\"\"\"


def word_count(text: str) -> int:
    \"\"\"Return the number of whitespace-separated words in *text*.\"\"\"
    return len(text.split())


def char_count(text: str) -> int:
    \"\"\"Return the number of non-whitespace characters in *text*.\"\"\"
    return sum(1 for ch in text if not ch.isspace())


def _format_line(words: int, chars: int) -> str:
    \"\"\"Format a summary line as 'words=<N>, chars=<M>'.\"\"\"
    return f\"words={words}, chars={chars}\"


def summarize(text: str) -> str:
    \"\"\"Return a one-line summary of *text*.

    Format: ``\"words=<N>, chars=<M>\"`` where *N* is the word count and
    *M* is the non-whitespace character count.
    \"\"\"
    return _format_line(word_count(text), char_count(text))
"""

#: Golden ``test_text_stats.py`` -- the expected value corrected from
#: the singular ``"word=2, char=10"`` to the plural ``"words=2, chars=10"``
#: to match the documented format.
GOLDEN_TEST = """\
\"\"\"Tests for the text statistics library.\"\"\"\n
from text_stats import summarize, word_count


def test_word_count():
    \"\"\"word_count splits on whitespace (passing case).\"\"\"
    assert word_count(\"hello world\") == 2
    assert word_count(\"one two three four\") == 4


def test_summarize():
    \"\"\"summarize produces the documented 'words=<N>, chars=<M>' format.\"\"\"
    result = summarize(\"hello world\")
    assert result == \"words=2, chars=10\"
"""


def _run_pytest(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python -m pytest -q`` inside *workspace* and capture output."""
    return subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _run_driver(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python driver.py`` inside *workspace* and capture output."""
    return subprocess.run(
        [sys.executable, "driver.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )


def _seed_workspace(workspace: Path) -> None:
    """Copy the static fixture files into *workspace* under their run-time names.

    The fixture test file is stored as ``unit_checks.py`` (not ``test_*``)
    so pytest does not collect it from ``benchmarks/fixtures/`` during the
    normal suite run.  At run-time it is written to the workspace as
    ``test_text_stats.py``.
    """
    (workspace / "text_stats.py").write_text((_FIXTURE_DIR / "library.py").read_text())
    (workspace / "driver.py").write_text((_FIXTURE_DIR / "driver.py").read_text())
    (workspace / "test_text_stats.py").write_text((_FIXTURE_DIR / "unit_checks.py").read_text())


@pytest.mark.benchmark
def test_cross_file_refactor(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Multi-step shape (medium tier):

        1. **Pre-condition** -- seed the workspace with the broken
           fixture files and run ``python -m pytest -q``.  The suite MUST
           fail: ``test_word_count`` passes but ``test_summarize`` errors
           with ``NameError`` (``_format_line`` is undefined).  This
           proves the bug is real and not silently masked.
        2. **Gold fix** -- apply the golden corrections to **two** files:
           add ``_format_line`` to ``text_stats.py`` and fix the expected
           value in ``test_text_stats.py``.
        3. **Post-condition** -- re-run ``python -m pytest -q``.  Both
           tests MUST pass (rc=0).  Additionally, ``python driver.py``
           MUST print the expected stdout, proving the library integrates
           end-to-end.
    """
    _seed_workspace(benchmark_workspace)

    expected_driver_stdout = (_FIXTURE_DIR / "expected_driver_stdout.txt").read_text()

    # --- Pre-condition: the seeded suite is genuinely broken. ------------
    bad = _run_pytest(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail pytest before the "
        f"fix; got rc={bad.returncode} stdout={bad.stdout!r} "
        f"stderr={bad.stderr!r}"
    )
    assert "1 passed" in bad.stdout, (
        f"task {TASK.name}: expected 'test_word_count' to pass in the "
        f"pre-condition; stdout={bad.stdout!r}"
    )

    # --- Gold fix: edit BOTH the library and the test file. -------------
    (benchmark_workspace / "text_stats.py").write_text(GOLDEN_LIBRARY)
    (benchmark_workspace / "test_text_stats.py").write_text(GOLDEN_TEST)

    # --- Post-condition: full suite passes + driver output matches. -----
    good = _run_pytest(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must pass pytest; "
        f"got rc={good.returncode} stdout={good.stdout!r} "
        f"stderr={good.stderr!r}"
    )
    assert "2 passed" in good.stdout, (
        f"task {TASK.name}: expected both tests to pass; " f"stdout={good.stdout!r}"
    )

    driver = _run_driver(benchmark_workspace)
    assert driver.returncode == 0, (
        f"task {TASK.name}: driver.py must exit 0 after the fix; "
        f"got rc={driver.returncode} stderr={driver.stderr!r}"
    )
    assert driver.stdout == expected_driver_stdout, (
        f"task {TASK.name}: driver stdout mismatch "
        f"(got {driver.stdout!r}, expected {expected_driver_stdout!r})"
    )
