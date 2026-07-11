"""Tests for the per-session triage card renderer (issue #180).

The acceptance criteria call for two layers of test:

* A *golden* string match against ``format_session_card`` directly, so a
  synthetic session with deterministic event IDs pins the exact output
  line-for-line.
* ``fx-trace session-card ...`` CLI integration, so the wiring and
  exit-code contract (0 when the session exists, non-zero otherwise) are
  pinned against the real :class:`TraceLogger`.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from foundry_x.observability.cli import main as cli_main
from foundry_x.observability.session_card import format_session_card
from foundry_x.trace.logger import TraceEvent, TraceLogger, TraceSession


_BASE = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)

# Sentinel for ``_session`` so callers can pass ``ended_at=None`` to
# exercise the missing-end path without being overridden by the helper
# default.
_USE_DEFAULT = object()


def _event(kind: str, offset: timedelta, payload: dict, event_id: str | None = None) -> TraceEvent:
    """Build a :class:`TraceEvent` with a deterministic identifier."""
    return TraceEvent(
        event_id=event_id or f"evt-{kind}",
        session_id="sess-test",
        timestamp=(_BASE + offset).isoformat(),
        kind=kind,
        payload=payload,
    )


def _session(
    session_id: str = "sess-test",
    harness_version: str = "0.1.0",
    model_id: str | None = "test-model",
    ended_at: str | None | object = _USE_DEFAULT,
) -> TraceSession:
    if ended_at is _USE_DEFAULT:
        ended_at = (_BASE + timedelta(seconds=2)).isoformat()
    return TraceSession(
        session_id=session_id,
        started_at=_BASE.isoformat(),
        harness_version=harness_version,
        model_id=model_id,
        metadata={},
        ended_at=ended_at,
    )


def _golden_session_card() -> tuple[TraceSession, list[TraceEvent]]:
    session = _session()
    events = [
        _event("user_prompt", timedelta(seconds=0.0), {"prompt": "Fix the bug"}),
        _event(
            "tool_call", timedelta(seconds=0.3), {"name": "read_file"}, event_id="evt-read-call"
        ),
        _event(
            "tool_call", timedelta(seconds=0.6), {"name": "edit_file"}, event_id="evt-edit-call"
        ),
        _event("tool_result", timedelta(seconds=0.9), {"name": "edit_file", "status": "ok"}),
        _event(
            "tool_error",
            timedelta(seconds=1.2),
            {"name": "bash", "error": "boom"},
            event_id="evt-bash-err01",
        ),
        _event(
            "outcome",
            timedelta(seconds=1.4),
            {"status": "failed", "reason": "tool_error", "steps": 3},
            event_id="evt-outcome",
        ),
    ]
    return session, events


# --- golden-string ------------------------------------------------------------


def test_format_session_card_golden_string():
    """A synthetic session renders the exact expected card (issue #180 §AC)."""
    session, events = _golden_session_card()
    output = format_session_card(session, events)
    expected = (
        "session_id        sess-test\n"
        "harness_version   0.1.0\n"
        "model_id          test-model\n"
        "started_at        2026-07-10T12:00:00+00:00\n"
        "ended_at          2026-07-10T12:00:02+00:00\n"
        "duration          0:00:02\n"
        "outcome           status=failed reason=tool_error steps=3\n"
        "event_count       6\n"
        "tool_calls        edit_file=1, read_file=1\n"
        "errors_by_kind    tool_error=1\n"
        "first_error       evt-bash tool_error: boom"
    )
    assert output == expected


def test_format_session_card_emits_all_required_labels():
    """Pin the full set of label keys called out in the issue's AC bullets."""
    session, events = _golden_session_card()
    output = format_session_card(session, events)
    labels = {line.split(None, 1)[0] for line in output.splitlines() if line.strip()}
    assert {
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
    } <= labels


# --- graceful degradation ----------------------------------------------------


def test_format_session_card_no_outcome_event_reads_no_outcome():
    """A session without an ``outcome`` event degrades to ``_no outcome_``."""
    session, events = _golden_session_card()
    without_outcome = [e for e in events if e.kind != "outcome"]
    output = format_session_card(session, without_outcome)
    assert "outcome           _no outcome_" in output


def test_format_session_card_no_events_shows_zeros_and_none():
    """An in-flight session with zero events still renders a usable card."""
    session = _session(ended_at=None)
    output = format_session_card(session, [])
    assert "event_count       0" in output
    assert "tool_calls        _none_" in output
    assert "errors_by_kind    _none_" in output
    assert "first_error       _none_" in output
    assert "outcome           _no outcome_" in output
    # duration falls back to ``_unknown_`` when ``ended_at`` is missing.
    assert "duration          _unknown_" in output


def test_format_session_card_missing_model_id_shows_dash():
    session = _session(model_id=None)
    output = format_session_card(session, [])
    assert "model_id          -" in output


def test_format_session_card_missing_ended_at_shows_dash():
    session = _session(ended_at=None)
    output = format_session_card(session, [])
    assert "ended_at          -" in output
    assert "duration          _unknown_" in output


# --- tool-call & error-count shape ------------------------------------------


def test_format_session_card_aggregates_tool_calls_by_name():
    session = _session()
    events = [
        _event("tool_call", timedelta(seconds=0.0), {"name": "read_file"}),
        _event("tool_call", timedelta(seconds=0.1), {"name": "read_file"}),
        _event("tool_call", timedelta(seconds=0.2), {"name": "edit_file"}),
    ]
    output = format_session_card(session, events)
    # Counts are sorted by name for stable golden output.
    assert "tool_calls        edit_file=1, read_file=2" in output


def test_format_session_card_buckets_untitled_tool_call():
    session = _session()
    events = [
        # Missing ``name`` payload key — bucket under the placeholder.
        _event("tool_call", timedelta(seconds=0.0), {}),
        _event("tool_call", timedelta(seconds=0.1), {"name": None}),
        _event("tool_call", timedelta(seconds=0.2), {"name": "read_file"}),
    ]
    output = format_session_card(session, events)
    assert "tool_calls        <unnamed>=2, read_file=1" in output


def test_format_session_card_aggregates_errors_by_kind():
    session = _session()
    events = [
        _event("tool_error", timedelta(seconds=0.0), {"name": "bash", "error": "x"}),
        _event("tool_error", timedelta(seconds=0.1), {"name": "bash", "error": "y"}),
        _event("task_failed", timedelta(seconds=0.2), {"message": "stopped"}),
        _event("abort_run", timedelta(seconds=0.3), {"reason": "timeout"}),
    ]
    output = format_session_card(session, events)
    assert "errors_by_kind    abort_run=1, task_failed=1, tool_error=2" in output


def test_format_session_card_first_error_is_earliest_in_timeline():
    session = _session()
    events = [
        _event(
            "tool_call", timedelta(seconds=0.0), {"name": "read_file"}, event_id="evt-read-call"
        ),
        _event(
            "tool_error",
            timedelta(seconds=0.2),
            {"name": "bash", "error": "boom"},
            event_id="evt-bash-err01",
        ),
        _event(
            "tool_error",
            timedelta(seconds=0.4),
            {"name": "edit", "error": "later"},
            event_id="evt-edit-err02",
        ),
    ]
    output = format_session_card(session, events)
    # Earliest tool_error wins; later occurrences are listed only under
    # ``errors_by_kind`` (their count == 2).
    assert "first_error       evt-bash tool_error: boom" in output
    assert "errors_by_kind    tool_error=2" in output


def test_format_session_card_truncates_long_first_error_message():
    long_message = "x" * 200
    events = [
        _event("tool_error", timedelta(seconds=0.0), {"error": long_message}, event_id="evt-long")
    ]
    output = format_session_card(_session(), events)
    # 120-char cap with a trailing ellipsis marker.
    snippet = "x" * 119 + "\u2026"
    assert f"first_error       evt-long tool_error: {snippet}" in output


def test_format_session_card_outcome_does_not_count_as_first_error():
    """The ``outcome`` kind is a terminal marker, not a failure event."""
    events = [_event("outcome", timedelta(seconds=0.0), {"status": "failed"}, event_id="evt-out")]
    output = format_session_card(_session(), events)
    assert "first_error       _none_" in output
    assert "errors_by_kind    _none_" in output


# --- CLI integration ----------------------------------------------------------


def _populate_session(db_path) -> tuple[str, TraceLogger]:
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Fix the bug"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "tool_error", {"name": "bash", "error": "boom"})
        logger.record(sid, "outcome", {"status": "failed", "reason": "tool_error", "steps": 2})
    return sid, logger


def test_cli_session_card_prints_expected_labels(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid, _ = _populate_session(db)

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    for label in (
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
    ):
        assert label in out
    assert "test-model" in out
    assert "0.1.0" in out
    # Errors recorded above surface in the count and inline snippet.
    assert "tool_error=1" in out
    assert "boom" in out


def test_cli_session_card_missing_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)  # empty store

    rc = cli_main(["session-card", "--db", str(db), "--session-id", "ghost-session"])

    assert rc != 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "ghost-session" in captured.err


def test_cli_session_card_jsonl_backend(tmp_path, capsys):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="0.2.0", model_id="json-model") as sid:
        logger.record(sid, "tool_call", {"name": "edit_file"})
        logger.record(sid, "outcome", {"status": "ok", "reason": "final_answer", "steps": 1})

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "harness_version   0.2.0" in out
    assert "model_id          json-model" in out
    assert "tool_calls        edit_file=1" in out
    assert "outcome           status=ok reason=final_answer steps=1" in out


def test_cli_session_card_no_outcome_event(tmp_path, capsys):
    """A session whose only events are tool calls degrades gracefully."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "tool_call", {"name": "read_file"})

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "outcome           _no outcome_" in out
    assert "tool_calls        read_file=1" in out
