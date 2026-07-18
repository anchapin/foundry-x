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
        _event(
            "outcome",
            timedelta(seconds=2.0),
            {"status": "failed", "reason": "task_failed", "steps": 3},
        ),
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
    assert lines[6] == _format_line("outcome", "status=failed reason=task_failed steps=3")
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
        _event(
            "outcome", timedelta(seconds=0.2), {"status": "success", "reason": "done", "steps": 1}
        ),
    ]
    output = format_session_card(session, events)

    assert "_none_" in output
    assert "read_file=1" in output


def test_format_session_card_no_tool_calls_shows_none():
    session = _sample_session()
    events = [
        _event("user_prompt", timedelta(seconds=0.0), {"prompt": "Hello"}),
        _event("error", timedelta(seconds=0.1), {"message": "oops"}),
        _event(
            "outcome", timedelta(seconds=0.2), {"status": "failed", "reason": "error", "steps": 1}
        ),
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


# --- Issue #872: tool-call argument parse-error events are surfaced in the
# session card via the existing ``errors_by_kind`` bucket. The runner emits a
# ``tool_argument_parse_error`` event whenever the model emits a tool call
# whose ``arguments`` JSON cannot be parsed (see
# ``src/foundry_x/execution/runner.py:1684``); the kind's "error" substring
# matches the session-card failure-regex, so it flows into the per-kind
# count automatically. These tests pin that contract. ---


def test_format_session_card_counts_tool_argument_parse_error_events():
    session = _sample_session()
    events = _sample_events() + [
        _event(
            "tool_argument_parse_error",
            timedelta(seconds=1.5),
            {
                "call_id": "call-abc",
                "name": "read_file",
                "raw": "not-json",
                "error": "JSONDecodeError",
            },
        ),
        _event(
            "tool_argument_parse_error",
            timedelta(seconds=1.7),
            {
                "call_id": "call-def",
                "name": "read_file",
                "raw": '{"oops"',
                "error": "expected JSON object, got str",
            },
        ),
    ]

    output = format_session_card(session, events)

    assert "tool_argument_parse_error=2" in output


def test_format_session_card_omits_parse_error_count_when_absent():
    session = _sample_session()
    events = _sample_events()  # no tool_argument_parse_error events

    output = format_session_card(session, events)

    assert "tool_argument_parse_error" not in output


def test_cli_session_card_surfaces_tool_argument_parse_error_count(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(str(db))
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Hello"})
        logger.record(
            sid,
            "tool_argument_parse_error",
            {
                "call_id": "call-abc",
                "name": "read_file",
                "raw": "not-json",
                "error": "JSONDecodeError",
            },
        )
        logger.record(
            sid,
            "tool_argument_parse_error",
            {
                "call_id": "call-def",
                "name": "read_file",
                "raw": '{"oops"',
                "error": "expected JSON object, got str",
            },
        )
        logger.record(
            sid,
            "outcome",
            {"status": "failed", "reason": "parse_error", "steps": 1},
        )

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid])

    assert rc == 0
    out = capsys.readouterr().out
    assert "tool_argument_parse_error=2" in out


# ---------------------------------------------------------------------------
# Issue #902: --latest flag on fx-trace session-card.
# ---------------------------------------------------------------------------


def _populate_two_sessions(db_path) -> tuple[str, str]:
    """Plant two sessions (different harness versions) so the ``--latest``
    path has a non-trivial choice. Returns ``(sid_old, sid_new)``.
    """
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0") as sid_old:
        logger.record(sid_old, "user_prompt", {"prompt": "older session"})
        logger.record(
            sid_old,
            "outcome",
            {"status": "success", "reason": "final_answer", "steps": 1},
        )
    with logger.session(harness_version="0.2.0") as sid_new:
        logger.record(sid_new, "user_prompt", {"prompt": "newer session"})
        logger.record(
            sid_new,
            "outcome",
            {"status": "failed", "reason": "task_failed", "steps": 2},
        )
    return sid_old, sid_new


def test_cli_session_card_latest_renders_most_recent_session(tmp_path, capsys):
    """Issue #902: ``fx-trace session-card --latest`` renders the most
    recent session card (no ``--session-id`` required).
    """
    db = tmp_path / "traces.db"
    sid_old, sid_new = _populate_two_sessions(db)

    rc = cli_main(["session-card", "--db", str(db), "--latest"])

    assert rc == 0
    out = capsys.readouterr().out
    # The newer session is rendered; the older one is not. The card shows
    # the outcome line derived from the most recent ``outcome`` event, so
    # we look there (not at the user_prompt text, which the card never
    # surfaces) to disambiguate the two sessions.
    assert sid_new in out
    assert sid_old not in out
    assert "status=failed reason=task_failed steps=2" in out
    assert "status=success reason=final_answer steps=1" not in out


def test_cli_session_card_latest_with_harness_version(tmp_path, capsys):
    """Issue #902: ``--latest --harness-version X`` picks the most recent
    session matching X. The fixture stores one 0.1.0 session and one
    0.2.0 session.
    """
    db = tmp_path / "traces.db"
    sid_old, sid_new = _populate_two_sessions(db)

    rc = cli_main(
        [
            "session-card",
            "--db",
            str(db),
            "--latest",
            "--harness-version",
            "0.1.0",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    # Only the 0.1.0 session (the older one) qualifies; it must be
    # rendered and the 0.2.0 session must be filtered out.
    assert sid_old in out
    assert sid_new not in out


def test_cli_session_card_latest_with_session_id_errors(tmp_path, capsys):
    """Issue #902: ``--latest`` and ``--session-id`` are mutually exclusive.
    Passing both is a usage error: CLI exits 2 with a friendly stderr
    message and no stdout.
    """
    db = tmp_path / "traces.db"
    sid_old, _ = _populate_two_sessions(db)

    rc = cli_main(
        [
            "session-card",
            "--db",
            str(db),
            "--latest",
            "--session-id",
            sid_old,
        ]
    )

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--latest and --session-id are mutually exclusive" in captured.err


def test_cli_session_card_neither_session_id_nor_latest_errors(tmp_path, capsys):
    """Issue #902: missing both ``--session-id`` and ``--latest`` is a usage
    error now that ``--session-id`` is no longer required.
    """
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = cli_main(["session-card", "--db", str(db)])

    assert rc == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "either --session-id or --latest is required" in captured.err


def test_cli_session_card_latest_empty_store_says_no_sessions(tmp_path, capsys):
    """Issue #902: empty store + ``--latest`` exits 0 with ``no sessions``.
    This is the friendly-operator contract; an empty store is not an
    error, just an informative message.
    """
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = cli_main(["session-card", "--db", str(db), "--latest"])

    assert rc == 0
    assert capsys.readouterr().out == "no sessions\n"


def test_cli_session_card_latest_with_harness_version_no_match(tmp_path, capsys):
    """Issue #902: ``--latest --harness-version X`` where no session matches
    X exits 0 with ``no sessions`` — mirrors the empty-store contract.
    """
    db = tmp_path / "traces.db"
    _populate_two_sessions(db)

    rc = cli_main(
        [
            "session-card",
            "--db",
            str(db),
            "--latest",
            "--harness-version",
            "9.9.9",
        ]
    )

    assert rc == 0
    assert capsys.readouterr().out == "no sessions\n"


def test_cli_session_card_session_id_remains_backward_compatible(tmp_path, capsys):
    """Issue #902 acceptance criterion 5: ``--session-id X`` (no ``--latest``)
    keeps its existing behavior unchanged. This guards against an
    accidental default flip and pins the contract for downstream callers.
    """
    db = tmp_path / "traces.db"
    sid_old, _ = _populate_two_sessions(db)

    rc = cli_main(["session-card", "--db", str(db), "--session-id", sid_old])

    assert rc == 0
    out = capsys.readouterr().out
    assert sid_old in out
    # ``--harness-version`` is documented to have no effect when
    # ``--session-id`` is supplied; verify the lookup still finds the
    # 0.1.0 session even when a different harness version is requested.
    rc = cli_main(
        [
            "session-card",
            "--db",
            str(db),
            "--session-id",
            sid_old,
            "--harness-version",
            "9.9.9",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert sid_old in out
