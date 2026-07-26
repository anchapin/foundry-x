"""Benchmark task: coordinated three-file constant rename.

This is the third ``difficulty_tier='medium'`` multi-file refactor task
in the suite (after ``cross_file_refactor`` #176 and ``multi_file_rename``
#267).  Unlike those two -- which edit two files -- this task requires
the agent to **read** three files and **edit** one of them in one session:

    1. read ``constants.py`` -- discover it defines ``MAX_BUFFER_SIZE``
       (the OLD name),
    2. read ``processor.py`` -- see it imports and uses ``MAX_CHUNK_SIZE``
       (the NEW name),
    3. read ``server.py`` -- see it also imports and uses ``MAX_CHUNK_SIZE``
       (the NEW name),
    4. edit ``constants.py`` -- rename ``MAX_BUFFER_SIZE`` ->
       ``MAX_CHUNK_SIZE`` so both consumers resolve correctly,
    5. re-run ``python main.py`` -- must exit 0 with the expected stdout.

The seeded workspace is deliberately half-renamed: both consumer files
already use the new name ``MAX_CHUNK_SIZE`` but ``constants.py`` still
defines the old name ``MAX_BUFFER_SIZE``, so running ``python main.py``
raises ``NameError``.  The agent must identify the single file that needs
editing (not grep-blindly rewrite all three) -- the smallest credible
shape that exercises ``read_file`` + ``grep_search`` + ``edit_file`` in
a coordinated sequence, fulfilling the quantization-sensitivity requirement
in ADR-0020.

The ``requires_skills`` list carries ``read_file`` (to read all three
files), ``grep_search`` (to locate the stale symbol before editing), and
``edit_file`` (to apply the targeted rename), so the Critic (ADR-0004)
can flag the task as "not-yet-evaluable" when any of those skills is
absent from the harness.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

TASK = BenchmarkTask(
    name="refactor_across_three_files",
    description=(
        "Three files are in the workspace: constants.py defines the OLD "
        "constant name MAX_BUFFER_SIZE, while processor.py and server.py "
        "already import and use the NEW name MAX_CHUNK_SIZE.  "
        "Rename MAX_BUFFER_SIZE -> MAX_CHUNK_SIZE in constants.py so both "
        "consumers resolve correctly and 'python main.py' exits 0."
    ),
    prompt=(
        "Three files are in the workspace:\n"
        "  - constants.py: defines MAX_BUFFER_SIZE (the OLD name).\n"
        "  - processor.py: imports and uses MAX_CHUNK_SIZE (the NEW name).\n"
        "  - server.py: imports and uses MAX_CHUNK_SIZE (the NEW name).\n"
        "\n"
        "The workspace is half-renamed: both consumer files already use "
        "the new name MAX_CHUNK_SIZE, but constants.py still defines the "
        "old name MAX_BUFFER_SIZE, so 'python main.py' currently fails "
        "with NameError.\n"
        "\n"
        "Read all three files. Use grep_search to locate the stale symbol "
        "MAX_BUFFER_SIZE (it only appears in constants.py). Then edit "
        "constants.py to rename MAX_BUFFER_SIZE -> MAX_CHUNK_SIZE so both "
        "consumers resolve correctly.\n"
        "\n"
        "After the edit, 'python main.py' must exit 0 and print:\n"
        "  processed_size=11\n"
        "  port=8080, max_payload=4096"
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "After the rename, 'python main.py' returns rc=0 with stdout equal "
        "to fixtures/refactor_across_three_files/expected_stdout.txt."
    ),
    timeout_seconds=30,
    requires_skills=["read_file", "grep_search", "edit_file"],
    tags=["multi-file", "refactoring", "rename"],
)

#: Root of the static fixture data for this task.
_FIXTURE_DIR = Path(__file__).parent.parent / "fixtures" / TASK.name

#: Golden ``constants.py`` -- ``MAX_BUFFER_SIZE`` renamed to ``MAX_CHUNK_SIZE``.
#: Only the definition name changes; the value and all other content are
#: identical to the seed so the rename is a pure symbol rename (no
#: behavioural drift to reason about).
GOLDEN_CONSTANTS = '''\
"""Constants definition for refactor_across_three_files benchmark (golden).

MAX_BUFFER_SIZE has been renamed to MAX_CHUNK_SIZE to match the
consumers' already-migrated references.
"""

MAX_CHUNK_SIZE = 4096

DEFAULT_TIMEOUT = 30
'''

#: Golden ``processor.py`` -- identical to the seed (only constants.py changes).
GOLDEN_PROCESSOR = '''\
"""Processor module for refactor_across_three_files benchmark (seeded/broken).

References MAX_CHUNK_SIZE -- the NEW constant name.  Since constants.py still
defines MAX_BUFFER_SIZE (the OLD name), running main.py raises NameError
until constants.py is updated to define the new name.

This file is the *seeded* (broken) state.  The golden state is identical
to this file -- only constants.py changes.
"""

from constants import MAX_CHUNK_SIZE


def process_data(data: str) -> int:
    """Return the byte size of *data* capped to MAX_CHUNK_SIZE."""
    size = len(data.encode("utf-8"))
    return min(size, MAX_CHUNK_SIZE)


def main() -> None:
    """Print the process_data result for a sample string."""
    result = process_data("hello world")
    print(f"processed_size={result}")
'''

#: Golden ``server.py`` -- identical to the seed (only constants.py changes).
GOLDEN_SERVER = '''\
"""Server module for refactor_across_three_files benchmark (seeded/broken).

References MAX_CHUNK_SIZE -- the NEW constant name.  Since constants.py still
defines MAX_BUFFER_SIZE (the OLD name), running main.py raises NameError
until constants.py is updated to define the new name.

This file is the *seeded* (broken) state.  The golden state is identical
to this file -- only constants.py changes.
"""

from constants import MAX_CHUNK_SIZE

SERVER_PORT = 8080


def make_server_config() -> dict[str, int]:
    """Return a server config dict using MAX_CHUNK_SIZE as the max payload."""
    return {
        "port": SERVER_PORT,
        "max_payload": MAX_CHUNK_SIZE,
    }


def main() -> None:
    """Print the server config as a one-liner."""
    cfg = make_server_config()
    print(f"port={cfg['port']}, max_payload={cfg['max_payload']}")
'''


def _run_main(workspace: Path) -> subprocess.CompletedProcess[str]:
    """Run ``python main.py`` inside *workspace* and capture all output."""
    return subprocess.run(
        [sys.executable, "main.py"],
        cwd=workspace,
        capture_output=True,
        text=True,
        check=False,
    )


def _seed_workspace(workspace: Path) -> None:
    """Copy the static fixture files into *workspace* under their run-time names."""
    (workspace / "constants.py").write_text((_FIXTURE_DIR / "constants.py").read_text())
    (workspace / "processor.py").write_text((_FIXTURE_DIR / "processor.py").read_text())
    (workspace / "server.py").write_text((_FIXTURE_DIR / "server.py").read_text())
    (workspace / "main.py").write_text((_FIXTURE_DIR / "main.py").read_text())


@pytest.mark.benchmark
def test_refactor_across_three_files(benchmark_workspace: Path) -> None:
    """Deterministic pass/fail check for TASK.

    Multi-step shape (medium tier):

        1. **Pre-condition** -- seed the workspace with the half-renamed
           fixture files and run ``python main.py``.  It MUST fail with
           ``NameError``: ``constants.py`` defines ``MAX_BUFFER_SIZE``
           but both consumers reference ``MAX_CHUNK_SIZE``.  This proves
           the inconsistency is real and not silently masked by the fixture.
        2. **Gold fix** -- apply the golden correction to **one** file:
           rename the definition in ``constants.py`` from ``MAX_BUFFER_SIZE``
           to ``MAX_CHUNK_SIZE``.  The consumer files are already correct.
        3. **Post-condition** -- re-run ``python main.py``.  It MUST
           exit 0 and print the expected stdout, proving the rename is
           consistent end-to-end.
    """
    _seed_workspace(benchmark_workspace)

    expected_stdout = (_FIXTURE_DIR / "expected_stdout.txt").read_text()

    # --- Pre-condition: the seeded workspace is genuinely broken. --------
    bad = _run_main(benchmark_workspace)
    assert bad.returncode != 0, (
        f"task {TASK.name}: seeded workspace must fail before the rename; "
        f"got rc={bad.returncode} stdout={bad.stdout!r} "
        f"stderr={bad.stderr!r}"
    )
    assert "ImportError" in bad.stderr, (
        f"task {TASK.name}: expected ImportError in stderr; got stderr={bad.stderr!r}"
    )

    # --- Gold fix: rename the constant definition in constants.py. -------
    (benchmark_workspace / "constants.py").write_text(GOLDEN_CONSTANTS)

    # --- Post-condition: the renamed workspace runs end-to-end. ----------
    good = _run_main(benchmark_workspace)
    assert good.returncode == 0, (
        f"task {TASK.name}: corrected workspace must exit 0; "
        f"got rc={good.returncode} stdout={good.stdout!r} "
        f"stderr={good.stderr!r}"
    )
    assert good.stdout == expected_stdout, (
        f"task {TASK.name}: stdout mismatch (got {good.stdout!r}, expected {expected_stdout!r})"
    )
