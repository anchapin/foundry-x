"""Regression guard for the terminal FoundryAgent-lifecycle event rows.

Issue #788 (duplicate of #784) asked for ``task_completed`` and
``task_failed`` to be documented in the ``## Event kinds`` table of
:mod:`docs.CONTEXT.md` with the correct producer, payload contract,
and failure-signal classification. The rows already exist; this test
pins their *column content* so that a future documentation drift
cannot silently weaken or drop either row. The companion
:mod:`tests.docs.test_event_kinds` only pins kind-name enumeration;
this test extends coverage to the per-row contracts the issue's
acceptance criteria named explicitly.

See ADR-0007 (trace-driven development) and ADR-0010 (Runner agent
loop) for the terminal-event contract this test guards.
"""

from __future__ import annotations

import re
from pathlib import Path

# Repo root: tests/docs/test_event_kinds_terminal_events.py -> parents[2]
REPO_ROOT = Path(__file__).resolve().parents[2]
CONTEXT_MD = REPO_ROOT / "docs" / "CONTEXT.md"

# Scans from one "## " heading to the next so we only inspect the
# Event kinds section. Mirrors the helper in test_event_kinds.py.
_SECTION_HEADING_RE = re.compile(r"^##\s+", re.MULTILINE)


def _read_event_kinds_section() -> str:
    text = CONTEXT_MD.read_text(encoding="utf-8")
    match = re.search(r"^##\s+Event kinds\s*$", text, re.MULTILINE)
    assert match is not None, "docs/CONTEXT.md is missing the '## Event kinds' section"
    start = match.end()
    next_heading = _SECTION_HEADING_RE.search(text, pos=start)
    end = next_heading.start() if next_heading else len(text)
    return text[start:end]


def _row_for(kind: str) -> str:
    """Return the cells of the ``kind`` row as a single string.

    The captured text spans every column after the kind cell up to the
    row's trailing pipe, so producer, payload, and failure-signal
    assertions can all run against the same captured slice.
    """
    section = _read_event_kinds_section()
    pattern = re.compile(
        rf"^\|\s*\*\*`{kind}`\*\*\s*\|(?P<rest>.*?)\|\s*$",
        re.MULTILINE,
    )
    match = pattern.search(section)
    assert match is not None, (
        f"docs/CONTEXT.md 'Event kinds' table is missing the `{kind}` row (issue #788/#784)."
    )
    return match.group("rest")


def test_task_completed_row_names_runner_main_producer():
    """The success-path terminal marker must attribute its producer to Runner.main."""
    row = _row_for("task_completed")
    assert "Runner.main" in row, (
        f"task_completed row must name Runner.main as the producer; got: {row!r}"
    )


def test_task_completed_row_payload_contract_pins_duration_ms():
    """The task_completed payload contract must document the duration_ms field.

    ``Runner.main`` records ``{"duration_ms": int}`` on the success path
    (``src/foundry_x/execution/runner.py``); the documented contract must
    name the same key so a Digester reading the docs matches the live
    event stream.
    """
    row = _row_for("task_completed")
    assert "duration_ms" in row, (
        f"task_completed payload contract must include duration_ms; got: {row!r}"
    )


def test_task_completed_row_is_not_a_failure_signal():
    """The success-path terminal marker must read 'no' in the failure-signal column."""
    row = _row_for("task_completed")
    assert re.search(r"\bno\b\s*$", row), (
        f"task_completed failure-signal column must be 'no'; got: {row!r}"
    )


def test_task_failed_row_names_runner_main_producer():
    """The exception-path terminal marker must attribute its producer to Runner.main."""
    row = _row_for("task_failed")
    assert "Runner.main" in row, (
        f"task_failed row must name Runner.main as the producer; got: {row!r}"
    )


def test_task_failed_row_payload_contract_pins_error_fields():
    """The task_failed payload contract must document error_type, message, and duration_ms.

    ``Runner.main`` records ``{"error_type": str, "message": str,
    "duration_ms": int}`` on the exception path
    (``src/foundry_x/execution/runner.py``); the documented contract
    must name the same keys so a Digester reading the docs matches the
    live event stream. Stack frames are deliberately omitted
    (ADR-0007).
    """
    row = _row_for("task_failed")
    for key in ("error_type", "message", "duration_ms"):
        assert key in row, f"task_failed payload contract must include {key!r}; got: {row!r}"


def test_task_failed_row_is_a_terminal_failure_signal():
    """The exception-path terminal marker must read 'yes' (terminal) in the failure-signal column."""
    row = _row_for("task_failed")
    assert re.search(r"\byes\b.*terminal", row), (
        f"task_failed failure-signal column must mark 'yes' (terminal); got: {row!r}"
    )
