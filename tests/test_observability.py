from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from foundry_x.evolution.digester import FailureReport
from foundry_x.observability.render import render_failure_report
from foundry_x.observability.timeline import format_timeline
from foundry_x.trace.logger import TraceEvent


def _event(kind: str, offset: timedelta, payload: dict) -> TraceEvent:
    base = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
    return TraceEvent(
        event_id=f"evt-{kind}",
        session_id="sess-test",
        timestamp=(base + offset).isoformat(),
        kind=kind,
        payload=payload,
    )


def _sample_events() -> list[TraceEvent]:
    return [
        _event("user_prompt", timedelta(seconds=0.0), {"prompt": "Fix the bug in auth.py"}),
        _event("tool_call", timedelta(seconds=0.3), {"name": "read_file"}),
        _event("tool_result", timedelta(seconds=0.8), {"name": "read_file", "status": "ok"}),
        _event("error", timedelta(seconds=1.2), {"message": "permission denied"}),
        _event("outcome", timedelta(seconds=2.0), {"status": "failed"}),
    ]


def _sample_report() -> FailureReport:
    return FailureReport(
        session_id="sess-001",
        summary="Agent called the wrong tool for file deletion.",
        failed_steps=[
            {"step": 3, "kind": "tool_call", "detail": "called rm instead of edit"},
            {"step": 5, "kind": "state_leak", "detail": "temp file left behind"},
        ],
        suspected_causes=[
            "System prompt does not list the edit_file skill.",
            "No hook validates tool selection before execution.",
        ],
        proposed_class="bad-prompt",
    )


# --- timeline formatter tests ---


def test_format_timeline_has_five_step_lines():
    output = format_timeline(_sample_events())
    step_lines = [ln for ln in output.splitlines() if re.search(r"#\d+", ln)]
    assert len(step_lines) == 5


def test_format_timeline_relative_offsets():
    output = format_timeline(_sample_events())
    assert "+0.0s" in output
    assert "+0.3s" in output
    assert "+0.8s" in output
    assert "+1.2s" in output
    assert "+2.0s" in output


def test_format_timeline_error_marker():
    events = _sample_events()
    output = format_timeline(events, highlight_errors=True)
    # The 4th event is the error; its line carries the leading marker.
    error_line = next(ln for ln in output.splitlines() if "#4" in ln)
    assert error_line.startswith("!") or error_line.startswith("\u2717")


def test_format_timeline_tool_name_summary():
    output = format_timeline(_sample_events())
    tool_call_line = next(ln for ln in output.splitlines() if re.search(r"#2", ln))
    assert "read_file" in tool_call_line


def test_format_timeline_highlight_disabled():
    output = format_timeline(_sample_events(), highlight_errors=False)
    assert "\u2717" not in output
    # The error line carries no leading marker column.
    error_line = next(ln for ln in output.splitlines() if "#4" in ln)
    assert not error_line.startswith("!")


def test_format_timeline_empty_events():
    assert format_timeline([]) == ""


def test_format_timeline_truncates_long_prompt():
    long_prompt = "x" * 200
    events = [_event("user_prompt", timedelta(0), {"prompt": long_prompt})]
    output = format_timeline(events)
    # Summary is truncated to 60 characters.
    summary_part = output.split("user_prompt", 1)[1].strip()
    assert len(summary_part) == 60


# --- failure report render tests ---


def test_render_contains_session_id():
    md = render_failure_report(_sample_report())
    assert "sess-001" in md


def test_render_contains_summary_heading_and_text():
    report = _sample_report()
    md = render_failure_report(report)
    assert "## Summary" in md
    assert report.summary in md


def test_render_contains_suspected_causes():
    report = _sample_report()
    md = render_failure_report(report)
    assert "## Suspected Causes" in md
    for cause in report.suspected_causes:
        assert cause in md
    assert "1. " in md
    assert "2. " in md


def test_render_contains_failed_steps_table():
    report = _sample_report()
    md = render_failure_report(report)
    assert "## Failed Steps" in md
    assert "| step | kind | detail |" in md
    assert "| --- | --- | --- |" in md
    for step in report.failed_steps:
        assert str(step["step"]) in md
        assert step["kind"] in md
        assert step["detail"] in md


def test_render_contains_classification():
    report = _sample_report()
    md = render_failure_report(report)
    assert f"## Classification: {report.proposed_class}" in md


def test_render_empty_fields_uses_defaults():
    report = FailureReport(
        session_id="sess-002",
        summary="Edge case: no failures recorded.",
    )
    md = render_failure_report(report)
    assert "## Summary" in md
    assert "## Suspected Causes" in md
    assert "## Failed Steps" in md
    assert "## Classification: unknown" in md
