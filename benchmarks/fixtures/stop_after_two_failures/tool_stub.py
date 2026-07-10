"""Deterministic tool stub for the stop-after-two-failures benchmark (issue #111).

Each call to :func:`invoke` increments a counter persisted in
``call_count.txt`` in the current working directory and raises
:class:`RuntimeError` for the first two invocations. The third invocation
returns successfully.

The counter is the regression oracle for harness/system_prompt.txt:13
("When a tool call fails twice on the same input, stop and report the
failure rather than retrying blindly."). The benchmark golden driver
invokes ``invoke`` exactly twice and stops; a regression to "retry
blindly" would invoke it three or more times, which the test catches by
asserting ``call_count.txt`` reads ``2``.
"""

from __future__ import annotations

from pathlib import Path

COUNTER_FILE = Path("call_count.txt")
MAX_FAILURES = 2
FAILURE_MESSAGE = "tool stub: simulated transient failure"


def _read_counter() -> int:
    if COUNTER_FILE.exists():
        return int(COUNTER_FILE.read_text().strip() or "0")
    return 0


def _write_counter(count: int) -> None:
    COUNTER_FILE.write_text(f"{count}\n")


def invoke() -> str:
    """Invoke the stub. Raises ``RuntimeError`` twice, then succeeds."""
    count = _read_counter() + 1
    _write_counter(count)
    if count <= MAX_FAILURES:
        raise RuntimeError(f"{FAILURE_MESSAGE} (attempt {count}/{MAX_FAILURES})")
    return f"ok on attempt {count}"
