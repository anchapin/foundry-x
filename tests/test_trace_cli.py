from __future__ import annotations

import inspect
import json
from collections.abc import Iterator

from foundry_x.trace.cli import main
from foundry_x.trace.logger import TraceEvent, TraceLogger


def _populate(db_path) -> str:
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Fix the bug in auth.py"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "outcome", {"status": "ok"})
    return sid


def test_sessions_lists_session(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    rc = main(["sessions", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert sid in out
    assert "0.1.0" in out
    assert "test-model" in out


def test_sessions_empty_db(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["sessions", "--db", str(db)])

    assert rc == 0
    assert "No sessions found" in capsys.readouterr().out


def test_show_prints_ordered_events(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    rc = main(["show", sid, "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert sid in out
    assert "user_prompt" in out
    assert "tool_call" in out
    assert "read_file" in out


def test_show_unknown_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["show", "does-not-exist", "--db", str(db)])

    assert rc == 1


def test_export_produces_jsonl(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    rc = main(["export", sid, "--db", str(db)])

    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert record["session_id"] == sid
        assert "event_id" in record
        assert "kind" in record
        assert "payload" in record


def test_export_to_file(tmp_path):
    db = tmp_path / "traces.db"
    out_file = tmp_path / "export.jsonl"
    sid = _populate(db)

    rc = main(["export", sid, "--db", str(db), "--out", str(out_file)])

    assert rc == 0
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    json.loads(lines[0])


def test_list_sessions_method(tmp_path):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    sessions = TraceLogger(db).list_sessions()

    assert len(sessions) == 1
    session = sessions[0]
    assert session.session_id == sid
    assert session.harness_version == "0.1.0"
    assert session.model_id == "test-model"


def test_list_sessions_jsonl_backend(tmp_path):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="0.2.0", model_id="json-model") as sid:
        logger.record(sid, "user_prompt", {"prompt": "hi"})

    sessions = TraceLogger(db, backend="jsonl").list_sessions()

    assert len(sessions) == 1
    assert sessions[0].session_id == sid
    assert sessions[0].harness_version == "0.2.0"


# ---------------------------------------------------------------------------
# Issue #82: TraceLogger.list_sessions(harness_version=...) and
# TraceLogger.iter_events(session_id, kind=...) are the centralized query
# surface that kpis.py and regression_report.py now go through (ADR-0003).
# The tests below pin both methods on sqlite and jsonl backends.
# ---------------------------------------------------------------------------


def test_list_sessions_filters_by_harness_version(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1") as sid_v1:
        logger.record(sid_v1, "tool_call", {"name": "read"})
    with logger.session(harness_version="v2") as sid_v2:
        logger.record(sid_v2, "tool_call", {"name": "write"})

    only_v1 = TraceLogger(db).list_sessions(harness_version="v1")
    assert {s.session_id for s in only_v1} == {sid_v1}

    only_v2 = TraceLogger(db).list_sessions(harness_version="v2")
    assert {s.session_id for s in only_v2} == {sid_v2}

    all_versions = TraceLogger(db).list_sessions()
    assert {s.session_id for s in all_versions} == {sid_v1, sid_v2}


def test_list_sessions_harness_version_filter_jsonl(tmp_path):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="alpha") as sid_a:
        logger.record(sid_a, "tool_call", {"name": "read"})
    with logger.session(harness_version="beta") as sid_b:
        logger.record(sid_b, "tool_call", {"name": "write"})

    only_alpha = TraceLogger(db, backend="jsonl").list_sessions(harness_version="alpha")
    assert {s.session_id for s in only_alpha} == {sid_a}


def test_iter_events_yields_trace_events_in_order(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1") as sid:
        first = logger.record(sid, "task_received", {"prompt": "do work"})
        second = logger.record(sid, "tool_call", {"name": "read_file"})
        third = logger.record(sid, "critic_verdict", {"verdict": "approved"})

    events = list(logger.iter_events(sid))

    assert [e.event_id for e in events] == [first.event_id, second.event_id, third.event_id]
    assert [e.kind for e in events] == ["task_received", "tool_call", "critic_verdict"]
    assert all(isinstance(e, TraceEvent) for e in events)


def test_iter_events_filters_by_kind(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, "task_received", {"prompt": "do work"})
        verdict = logger.record(sid, "critic_verdict", {"verdict": "approved", "regression": False})
        logger.record(sid, "tool_call", {"name": "read_file"})

    only_verdicts = list(logger.iter_events(sid, kind="critic_verdict"))
    assert len(only_verdicts) == 1
    assert only_verdicts[0].event_id == verdict.event_id


def test_iter_events_is_lazy_and_streams_rows(tmp_path):
    """Plant 1000 events and consume iter_events lazily.

    Issue #82 acceptance: ``iter_events`` yields one row at a time so a
    long session does not blow memory. We pin this by:
      1. Asserting the return type is an ``Iterator`` (not a ``Sequence``).
      2. Asserting the runtime type is a generator (the streaming shape).
      3. Pulling only the first 3 events from a 1000-event session via
         ``next()`` calls — if iter_events eagerly materialized, the
         break-out would still succeed; the type assertion is what proves
         the streaming contract.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1") as sid:
        for i in range(1000):
            logger.record(sid, "tool_call", {"i": i})

    iterator = logger.iter_events(sid)

    # Return type is Iterator (issue #82 acceptance criterion). The
    # annotation is read from the source as a string because logger.py
    # uses ``from __future__ import annotations``, so compare against
    # ``typing.get_type_hints`` (which evaluates the string) instead of
    # ``signature.return_annotation`` directly.
    import typing

    resolved_hints = typing.get_type_hints(TraceLogger.iter_events)
    return_annotation = resolved_hints["return"]
    # ``typing.Iterator`` and ``collections.abc.Iterator`` are not the same
    # object but both expose ``__origin__`` semantics; either is acceptable
    # as the public return-type contract for ``iter_events``.
    assert return_annotation in (Iterator[TraceEvent], typing.Iterator[TraceEvent])
    # Runtime shape is a generator (the ``yield``/``yield from`` body).
    assert inspect.isgenerator(iterator)

    # Lazy consumption: pull only the first 3 rows and break out.
    first_three = [next(iterator), next(iterator), next(iterator)]
    assert len(first_three) == 3
    assert [e.kind for e in first_three] == ["tool_call", "tool_call", "tool_call"]
    assert [e.payload["i"] for e in first_three] == [0, 1, 2]

    # The full session is still iterable (we just stopped early).
    iterator.close()
    full_count = sum(1 for _ in logger.iter_events(sid))
    assert full_count == 1000


def test_iter_events_empty_session(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1") as sid:
        pass  # no events

    events = list(logger.iter_events(sid))
    assert events == []


def test_iter_events_unknown_session_yields_nothing(tmp_path):
    db = tmp_path / "traces.db"
    TraceLogger(db)  # empty store
    logger = TraceLogger(db)

    events = list(logger.iter_events("does-not-exist"))
    assert events == []


def test_iter_events_jsonl_backend(tmp_path):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="v1") as sid:
        first = logger.record(sid, "task_received", {"prompt": "hi"})
        second = logger.record(sid, "critic_verdict", {"verdict": "approved"})

    events = list(TraceLogger(db, backend="jsonl").iter_events(sid))
    assert [e.event_id for e in events] == [first.event_id, second.event_id]
    assert all(isinstance(e, TraceEvent) for e in events)


def test_iter_events_jsonl_filters_by_kind(tmp_path):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, "task_received", {"prompt": "hi"})
        verdict = logger.record(sid, "critic_verdict", {"verdict": "approved"})

    only_verdicts = list(TraceLogger(db, backend="jsonl").iter_events(sid, kind="critic_verdict"))
    assert len(only_verdicts) == 1
    assert only_verdicts[0].event_id == verdict.event_id


# --- Issue #83: session-list / session-show / events-grep ---------------------


def _populate_two_versions(db_path):
    """Plant two sessions (different harness versions) with distinguishable events.

    Returns ``(sid_a, sid_b)`` in start-order. ``sid_a`` records a
    ``user_prompt`` whose payload contains the needle ``"BUG-1234"`` that
    the ``events-grep`` tests look for.
    """
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0", model_id="model-a") as sid_a:
        logger.record(sid_a, "user_prompt", {"prompt": "Fix BUG-1234 in auth.py"})
        logger.record(sid_a, "tool_call", {"name": "read_file"})
    with logger.session(harness_version="0.2.0", model_id="model-b") as sid_b:
        logger.record(sid_b, "user_prompt", {"prompt": "Refactor renderer"})
    return sid_a, sid_b


def test_session_list_prints_identifiers(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid_a, sid_b = _populate_two_versions(db)

    rc = main(["session-list", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    # Header and both session_ids land in the rendered table.
    assert "session_id  started_at  ended_at  harness_version" in out
    assert sid_a in out
    assert sid_b in out
    assert "0.1.0" in out
    assert "0.2.0" in out


def test_session_list_filters_by_harness_version(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid_a, sid_b = _populate_two_versions(db)

    rc = main(
        [
            "session-list",
            "--db",
            str(db),
            "--harness-version",
            "0.2.0",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert sid_b in out
    assert sid_a not in out
    assert "0.2.0" in out
    assert "0.1.0" not in out


def test_session_list_respects_limit(tmp_path, capsys):
    db = tmp_path / "traces.db"
    _populate_two_versions(db)

    rc = main(["session-list", "--db", str(db), "--limit", "1"])

    assert rc == 0
    out_lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    # 1 header + 1 data row.
    assert len(out_lines) == 2


def test_session_list_empty_db_exits_zero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["session-list", "--db", str(db)])

    assert rc == 0
    # Header is always printed so downstream tools can rely on it.
    assert "session_id  started_at  ended_at  harness_version" in capsys.readouterr().out


def test_session_show_prints_event_timeline(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid_a, _ = _populate_two_versions(db)

    rc = main(["session-show", sid_a, "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert sid_a in out
    assert "user_prompt" in out
    assert "tool_call" in out
    assert "read_file" in out


def test_session_show_unknown_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["session-show", "does-not-exist", "--db", str(db)])

    assert rc == 1


def test_events_grep_finds_matching_event(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid_a, _ = _populate_two_versions(db)

    rc = main(
        [
            "events-grep",
            sid_a,
            "--db",
            str(db),
            "--pattern",
            r"BUG-1234",
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "user_prompt" in out
    assert "BUG-1234" in out


def test_events_grep_no_match_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid_a, _ = _populate_two_versions(db)

    rc = main(
        [
            "events-grep",
            sid_a,
            "--db",
            str(db),
            "--pattern",
            r"this-string-will-not-appear",
        ]
    )

    # Conventional grep semantics: 1 when nothing matched.
    assert rc == 1


def test_events_grep_unknown_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(
        [
            "events-grep",
            "missing-session",
            "--db",
            str(db),
            "--pattern",
            r"anything",
        ]
    )

    assert rc == 1
