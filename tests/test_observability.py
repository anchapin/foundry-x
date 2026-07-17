from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.observability.cli import main as cli_main
from foundry_x.observability.render import render_failure_report
from foundry_x.observability.timeline import (
    build_timeline_records,
    format_timeline,
    render_timeline_json,
)
from foundry_x.trace.logger import TraceEvent, TraceLogger


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


def test_format_timeline_tool_latency_when_present():
    events = [_event("tool_call", timedelta(seconds=0.0), {"name": "read_file", "duration_ms": 42})]
    output = format_timeline(events)
    assert "read_file (42ms)" in output


def test_format_timeline_omits_tool_latency_when_missing():
    output = format_timeline(_sample_events())
    tool_call_line = next(ln for ln in output.splitlines() if re.search(r"#2", ln))
    assert "read_file" in tool_call_line
    assert "ms)" not in tool_call_line


# ---------------------------------------------------------------------------
# Issue #271: timeline shows cumulative token count on model_response lines.
# ---------------------------------------------------------------------------


def test_format_timeline_shows_cumulative_tokens_on_model_response():
    events = [
        _event("model_response", timedelta(seconds=0.0), {"tokens_used": 120}),
        _event(
            "model_response",
            timedelta(seconds=0.5),
            {"tokens_used": 345},
        ),
    ]
    output = format_timeline(events)

    lines = output.splitlines()
    first = next(ln for ln in lines if "#1" in ln)
    second = next(ln for ln in lines if "#2" in ln)
    assert "tokens:120" in first
    assert "tokens:345" in second


def test_format_timeline_omits_tokens_when_usage_missing():
    events = [_event("model_response", timedelta(seconds=0.0), {"finish_reason": "stop"})]
    output = format_timeline(events)
    assert "tokens:" not in output


# ---------------------------------------------------------------------------
# Issue #270: structured JSON timeline output.
# ---------------------------------------------------------------------------


def test_build_timeline_records_has_one_record_per_event():
    records = build_timeline_records(_sample_events())
    assert [r.step for r in records] == [1, 2, 3, 4, 5]
    assert [r.kind for r in records] == [
        "user_prompt",
        "tool_call",
        "tool_result",
        "error",
        "outcome",
    ]


def test_build_timeline_records_offsets_are_relative_deltas():
    records = build_timeline_records(_sample_events())
    assert records[0].offset_seconds == 0.0
    assert records[1].offset_seconds == 0.3
    assert records[2].offset_seconds == 0.8
    assert records[3].offset_seconds == 1.2


def test_build_timeline_records_marks_errors_with_is_error_flag():
    records = build_timeline_records(_sample_events())
    # The 4th event is the "error" kind; only it carries is_error=True.
    assert records[3].is_error is True
    assert all(not r.is_error for i, r in enumerate(records) if i != 3)


def test_build_timeline_records_empty_events_returns_empty_list():
    assert build_timeline_records([]) == []


def test_render_timeline_json_emits_parseable_array_with_required_keys():
    payload = json.loads(render_timeline_json(_sample_events()))
    assert isinstance(payload, list)
    assert len(payload) == len(_sample_events())
    for entry in payload:
        assert set(entry) == {"step", "offset_seconds", "kind", "summary", "is_error"}


def test_render_timeline_json_empty_events_emits_empty_array():
    assert json.loads(render_timeline_json([])) == []


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


# --- fx-trace timeline CLI tests ---


def _populate_session(db_path) -> str:
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Fix the bug in auth.py"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "tool_result", {"name": "read_file", "status": "ok"})
    return sid


def test_cli_timeline_prints_formatted_session(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(["timeline", "--db", str(db), "--session-id", sid])

    assert rc == 0
    captured = capsys.readouterr()
    assert "user_prompt" in captured.out
    assert "tool_call" in captured.out
    assert "read_file" in captured.out
    assert "tool_result" in captured.out
    # Five timeline lines are produced: each event renders a step line.
    step_lines = [ln for ln in captured.out.splitlines() if re.search(r"#\d+", ln)]
    assert len(step_lines) == 3
    # The known/loaded session should not be reported as missing.
    assert captured.err == ""


def test_cli_timeline_missing_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = cli_main(["timeline", "--db", str(db), "--session-id", "ghost-session"])

    assert rc != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ghost-session" in captured.err


def test_cli_timeline_jsonl_backend(tmp_path, capsys):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "user_prompt", {"prompt": "hi"})

    rc = cli_main(["timeline", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "user_prompt" in out
    assert "hi" in out


# ---------------------------------------------------------------------------
# Issue #270: fx-trace timeline --format json / --out .json
# ---------------------------------------------------------------------------


def test_cli_timeline_format_json_emits_parseable_array(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(["timeline", "--db", str(db), "--session-id", sid, "--format", "json"])

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, list)
    # _populate_session records three events.
    assert len(payload) == 3
    for entry in payload:
        assert set(entry) == {"step", "offset_seconds", "kind", "summary", "is_error"}
    assert [entry["kind"] for entry in payload] == [
        "user_prompt",
        "tool_call",
        "tool_result",
    ]


def test_cli_timeline_out_json_autoselects_json_format(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)
    out_path = tmp_path / "timeline.json"

    rc = cli_main(
        [
            "timeline",
            "--db",
            str(db),
            "--session-id",
            sid,
            "--out",
            str(out_path),
        ]
    )

    assert rc == 0
    # Nothing on stdout when --out is set.
    assert capsys.readouterr().out == ""
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert isinstance(payload, list)
    assert len(payload) == 3


def test_cli_timeline_default_format_is_markdown(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(["timeline", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    # Markdown output carries the step markers, not a JSON array.
    assert out.lstrip().startswith("#")
    with pytest.raises(json.JSONDecodeError):
        json.loads(out)


# ---------------------------------------------------------------------------
# Issue #710: fx-trace timeline --kind / -k filter by event kind.
# ---------------------------------------------------------------------------


def test_cli_timeline_kind_filter_single_kind(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(["timeline", "--db", str(db), "--session-id", sid, "--kind", "tool_call"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "tool_call" in out
    assert "user_prompt" not in out
    # Only the tool_call event should appear.
    step_lines = [ln for ln in out.splitlines() if re.search(r"#\d+", ln)]
    assert len(step_lines) == 1


def test_cli_timeline_kind_filter_short_form(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(["timeline", "--db", str(db), "--session-id", sid, "-k", "tool_result"])

    assert rc == 0
    out = capsys.readouterr().out
    assert "tool_result" in out
    assert "tool_call" not in out
    step_lines = [ln for ln in out.splitlines() if re.search(r"#\d+", ln)]
    assert len(step_lines) == 1


def test_cli_timeline_kind_filter_multiple_kinds(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(
        ["timeline", "--db", str(db), "--session-id", sid, "--kind", "tool_call,tool_result"]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "tool_call" in out
    assert "tool_result" in out
    assert "user_prompt" not in out
    # Two step lines: tool_call and tool_result.
    step_lines = [ln for ln in out.splitlines() if re.search(r"#\d+", ln)]
    assert len(step_lines) == 2


def test_cli_timeline_kind_filter_json_format(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(
        [
            "timeline",
            "--db",
            str(db),
            "--session-id",
            sid,
            "--kind",
            "tool_call",
            "--format",
            "json",
        ]
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert len(payload) == 1
    assert payload[0]["kind"] == "tool_call"


def test_cli_timeline_kind_filter_no_match_returns_empty(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(["timeline", "--db", str(db), "--session-id", sid, "--kind", "model_response"])

    # When the kind filter yields no events, the CLI reports "not found or empty".
    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "not found or empty" in captured.err


# ---------------------------------------------------------------------------
# Issue #268: fx-trace failure-report subcommand.
# ---------------------------------------------------------------------------


def _populate_failing_session(db_path) -> str:
    """Record a session whose ``error`` event trips the Digester."""
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Fix the bug in auth.py"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "error", {"message": "permission denied"})
    return sid


def test_cli_failure_report_prints_rendered_report(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_failing_session(db)

    rc = cli_main(["failure-report", "--db", str(db), "--session-id", sid])

    assert rc == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    # render_failure_report emits an h1 heading (see render.py:10).
    assert "# Failure Report" in captured.out
    assert sid in captured.out
    assert "## Summary" in captured.out
    assert "## Suspected Causes" in captured.out
    assert "## Failed Steps" in captured.out
    assert "## Classification:" in captured.out


def test_cli_failure_report_missing_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = cli_main(["failure-report", "--db", str(db), "--session-id", "ghost-session"])

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ghost-session" in captured.err
