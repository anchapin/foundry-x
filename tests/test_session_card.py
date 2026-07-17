"""Per-session triage card renderer (issue #804).

Acceptance pinned by this module:
  - ``fx-trace session-card --db <path> --session-id <sid>`` renders
    a card with the fields: session_id, harness_version, model_id,
    started_at, ended_at, duration, outcome, event_count, tool_calls,
    errors_by_kind, first_error.
  - Golden test pins the exact output format (issue #804 acceptance).
  - Non-zero exit code (2) when session not found.
  - All fields from ``format_session_card`` docstring present in output.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from foundry_x.observability.cli import main as cli_main
from foundry_x.observability.session_card import format_session_card
from foundry_x.trace.logger import TraceEvent, TraceLogger

_LABEL_WIDTH = 16
_VALUE_INDENT = "  "


def _format_line(label: str, value: str) -> str:
    return f"{label:<{_LABEL_WIDTH}}{_VALUE_INDENT}{value}".rstrip()


def _event(kind: str, offset: timedelta, payload: dict) -> TraceEvent:
    base = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)
    return TraceEvent(
        event_id=f"evt-{kind}",
        session_id="sess-test",
        timestamp=(base + offset).isoformat(),
        kind=kind,
        payload=payload,
    )


def _sample_session():
    from foundry_x.trace.logger import TraceSession

    return TraceSession(
        session_id="sess-test",
        started_at="2026-07-10T12:00:00+00:00",
        ended_at="2026-07-10T12:00:05+00:00",
        harness_version="0.1.0",
        model_id="gpt-4o",
    )


def _sample_events() -> list[TraceEvent]:
    return [
        _event("user_prompt", timedelta(seconds=0.0), {"prompt": "Fix the bug in auth.py"}),
        _event("tool_call", timedelta(seconds=0.3), {"name": "read_file"}),
        _event("tool_result", timedelta(seconds=0.8), {"name": "read_file", "status": "ok"}),
        _event("error", timedelta(seconds=1.2), {"message": "permission denied"}),
        _event("outcome", timedelta(seconds=2.0), {"status": "failed", "reason": "task_failed", "steps": 3}),
    ]


# --- format_session_card unit tests ---


def test_format_session_card_contains_all_required_fields():
    session = _sample_session()
    events = _sample_events()
    output = format_session_card(session, events)

    required_fields = [
        "session_id",
        "harness_version",
        "model_id",
        "started_at",
        "ended_at",
        "duration",
        "outcome",
        "event_count",
        "tool_calls",
        "errors_by_kind",
        "first_error",
    ]
    for field in required_fields:
        assert field in output, f"Missing field: {field}"


def test_format_session_card_golden_output():
    session = _sample_session()
    events = _sample_events()
    output = format_session_card(session, events)

    lines = output.splitlines()
    assert len(lines) == 11

    assert lines[0] == _format_line("session_id", "sess-test")
    assert lines[1] == _format_line("harness_version", "0.1.0")
    assert lines[2] == _format_line("model_id", "gpt-4o")
    assert lines[3] == _format_line("started_at", "2026-07-10T12:00:00+00:00")
    assert lines[4] == _format_line("ended_at", "2026-07-10T12:00:05+00:00")
    assert lines[5] == _format_line("duration", "0:00:05")
    assert lines[6] == _format_line(
        "outcome", "status=failed reason=task_failed steps=3"
    )
    assert lines[7] == _format_line("event_count", "5")
    assert lines[8] == _format_line("tool_calls", "read_file=1")
    assert lines[9] == _format_line("errors_by_kind", "error=1")
    assert "evt-erro" in lines[10]
    assert "permission denied" in lines[10]


def test_format_session_card_no_outcome_shows_placeholder():
    from foundry_x.trace.logger import TraceSession

    session = TraceSession(
        session_id="sess-no-outcome",
        started_at="2026-07-10T12:00:00+00:00",
        ended_at="2026-07-10T12:00:05+00:00",
        harness_version="0.1.0",
    )
    events = [
        _event("user_prompt", timedelta(seconds=0.0), {"prompt": "Hello"}),
    ]
    output = format_session_card(session, events)

    assert "_no outcome_" in output
    assert "event_count" in output


def test_format_session_card_no_errors_shows_none():
    session = _sample_session()
    events = [
        _event("user_prompt", timedelta(seconds=0.0), {"prompt": "Hello"}),
        _event("tool_call", timedelta(seconds=0.1), {"name": "read_file"}),
        _event("outcome", timedelta(seconds=0.2), {"status": "success", "reason": "done", "steps": 1}),
    ]
    output = format_session_card(session, events)

    assert "_none_" in output
    assert "read_file=1" in output


def test_format_session_card_no_tool_calls_shows_none():
    session = _sample_session()
    events = [
        _event("user_prompt", timedelta(seconds=0.0), {"prompt": "Hello"}),
        _event("error", timedelta(seconds=0.1), {"message": "oops"}),
        _event("outcome", timedelta(seconds=0.2), {"status": "failed", "reason": "error", "steps": 1}),
    ]
    output = format_session_card(session, events)

    assert "_none_" in output


# --- CLI integration tests ---


def _populate_session(db_path, harness_version="0.1.0", model_id="gpt-4o"):
    logger = TraceLogger(db_path)
    with logger.session(harness_version=harness_version, model_id=model_id) as sid:
        logger.record(sid, "user_prompt", {"prompt": "Fix the bug in auth.py"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "tool_result", {"name": "read_file", "status": "ok"})
        logger.record(sid, "error", {"message": "permission denied"})
        logger.record(sid, "outcome", {"status": "failed", "reason": "task_failed", "steps": 3})
    return sid


def test_cli_session_card_prints_card(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate_session(db)

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "session_id" in out
    assert "harness_version" in out
    assert "model_id" in out
    assert "started_at" in out
    assert "ended_at" in out
    assert "duration" in out
    assert "outcome" in out
    assert "event_count" in out
    assert "tool_calls" in out
    assert "errors_by_kind" in out
    assert "first_error" in out
    assert sid in out


def test_cli_session_card_missing_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = cli_main(["session-card", "--db", str(db), "--session-id", "ghost-session"])

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ghost-session" in captured.err


def test_cli_session_card_jsonl_backend(tmp_path, capsys):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="0.1.0", model_id="gpt-4o") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Hello"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "outcome", {"status": "success", "reason": "done", "steps": 1})

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "session_id" in out
    assert "read_file" in out


def test_cli_session_card_shows_tool_call_counts(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(str(db))
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Hello"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "tool_call", {"name": "write_file"})
        logger.record(sid, "outcome", {"status": "success", "reason": "done", "steps": 1})

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "read_file=2" in out
    assert "write_file=1" in out


def test_cli_session_card_shows_error_kinds(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(str(db))
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Hello"})
        logger.record(sid, "error", {"message": "oops"})
        logger.record(sid, "tool_call", {"name": "bash"})
        logger.record(sid, "error", {"message": "fail"})

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "error=2" in out
