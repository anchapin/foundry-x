from __future__ import annotations

import inspect
import json
from collections.abc import Iterator
from pathlib import Path

import pytest

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


# --- Issue #192: redact-session / redact-key ---------------------------------

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def _suffix(backend: str) -> str:
    return ".db" if backend == "sqlite" else ".jsonl"


def _populate_leak(tmp_path, backend: str) -> str:
    """Seed a session with a secret-bearing event for redaction tests."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            "tool_result",
            {"output": "config_api_key=leaked-value", "name": "read_file"},
        )
        logger.record(sid, "user_prompt", {"text": "deploy now"})
    return str(path), sid


@_BACKENDS
def test_redact_session_deletes_and_prints_count(tmp_path, backend, capsys):
    db, sid = _populate_leak(tmp_path, backend)

    rc = main(["redact-session", sid, "--db", db])

    assert rc == 0
    out = capsys.readouterr().out
    assert sid in out
    assert "2 event(s) removed" in out
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert logger.load_session(sid) == []
    assert sid not in [s.session_id for s in logger.list_sessions()]


@_BACKENDS
def test_redact_session_writes_audit_log(tmp_path, backend):
    db, sid = _populate_leak(tmp_path, backend)
    audit = tmp_path / "audit.jsonl"

    rc = main(["redact-session", sid, "--db", db, "--out", str(audit)])

    assert rc == 0
    lines = audit.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["action"] == "redact-session"
    assert record["session_id"] == sid
    assert record["events_deleted"] == 2


def test_redact_session_unknown_session_exits_zero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["redact-session", "never-existed", "--db", str(db)])

    assert rc == 0
    assert "0 event(s) removed" in capsys.readouterr().out


@_BACKENDS
def test_redact_key_rewrites_field(tmp_path, backend, capsys):
    db, sid = _populate_leak(tmp_path, backend)

    rc = main(["redact-key", sid, "0", "output", "--db", db])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Redacted key 'output'" in out
    path = tmp_path / f"traces{_suffix(backend)}"
    events = TraceLogger(path, backend=backend).load_session(sid)
    assert events[0].payload["output"] == "[REDACTED]"
    assert events[0].payload["name"] == "read_file"
    assert events[1].payload == {"text": "deploy now"}


@_BACKENDS
def test_redact_key_out_of_range_exits_nonzero(tmp_path, backend, capsys):
    db, sid = _populate_leak(tmp_path, backend)

    rc = main(["redact-key", sid, "99", "output", "--db", db])

    assert rc == 1
    err = capsys.readouterr().err
    assert "out of range" in err
    path = tmp_path / f"traces{_suffix(backend)}"
    events = TraceLogger(path, backend=backend).load_session(sid)
    assert events[0].payload["output"] == "config_api_key=leaked-value"


def test_redact_key_unknown_session_exits_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["redact-key", "missing", "0", "output", "--db", str(db)])

    assert rc == 1


@_BACKENDS
def test_redact_key_writes_audit_log(tmp_path, backend):
    db, sid = _populate_leak(tmp_path, backend)
    audit = tmp_path / "audit.jsonl"

    rc = main(["redact-key", sid, "0", "output", "--db", db, "--out", str(audit)])

    assert rc == 0
    lines = audit.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["action"] == "redact-key"
    assert record["session_id"] == sid
    assert record["event_index"] == 0
    assert record["key"] == "output"


def test_redact_audit_log_appends_across_invocations(tmp_path, capsys):
    db, sid = _populate_leak(tmp_path, "sqlite")
    audit = tmp_path / "audit.jsonl"

    main(["redact-key", sid, "0", "output", "--db", db, "--out", str(audit)])
    main(["redact-session", sid, "--db", db, "--out", str(audit)])

    lines = audit.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["action"] == "redact-key"
    assert json.loads(lines[1])["action"] == "redact-session"


# --- Issue #275: delete-session / prune --------------------------------------
# Retention-management subcommands. ``delete-session`` is the thin CLI
# wrapper over ``TraceLogger.delete_session``; ``prune`` adds two
# retention modes (``--keep-last N`` and ``--older-than DAYS``) plus a
# ``--dry-run`` flag. Both must work on sqlite and jsonl backends.


def _populate_multi(tmp_path, backend: str, count: int) -> tuple[str, list[str]]:
    """Plant ``count`` single-event sessions and return (db_path, session_ids).

    Sessions are created in insertion order; ``list_sessions`` returns them
    ascending by ``started_at``, so session_ids[0] is the oldest.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    sids: list[str] = []
    for _ in range(count):
        with logger.session(harness_version="0.1.0") as sid:
            logger.record(sid, "tool_call", {"name": "read_file"})
            sids.append(sid)
    return str(path), sids


def _backdate(path, backend: str, session_id: str, started_at: str) -> None:
    """Rewrite a session's ``started_at`` so ``--older-than`` is testable.

    The normal ``logger.session()`` API stamps ``started_at`` with "now",
    which makes age-based retention untestable without freezing the clock.
    Going directly to the store keeps the helper backend-aware and avoids
    monkey-patching ``datetime.now`` for every test.
    """
    if backend == "sqlite":
        import sqlite3

        with sqlite3.connect(path) as conn:
            conn.execute(
                "UPDATE sessions SET started_at = ? WHERE session_id = ?",
                (started_at, session_id),
            )
    else:
        lines = Path(path).read_text(encoding="utf-8").splitlines(keepends=True)
        rewritten: list[str] = []
        for line in lines:
            stripped = line.strip()
            if not stripped:
                rewritten.append(line)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                rewritten.append(line)
                continue
            if record.get("session_id") == session_id and record.get("kind") == "session_start":
                record["started_at"] = started_at
                rewritten.append(json.dumps(record) + "\n")
            else:
                rewritten.append(line)
        Path(path).write_text("".join(rewritten), encoding="utf-8")


_OLDEST_TS = "2020-01-01T00:00:00+00:00"


@_BACKENDS
def test_delete_session_removes_one_and_exits_zero(tmp_path, backend, capsys):
    db, sids = _populate_multi(tmp_path, backend, count=2)

    rc = main(["delete-session", sids[0], "--db", db])

    assert rc == 0
    out = capsys.readouterr().out
    assert sids[0] in out
    assert "1 event(s) removed" in out
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert sids[0] not in [s.session_id for s in logger.list_sessions()]
    assert sids[1] in [s.session_id for s in logger.list_sessions()]


@_BACKENDS
def test_delete_session_unknown_is_idempotent(tmp_path, backend, capsys):
    db, _ = _populate_multi(tmp_path, backend, count=1)

    rc = main(["delete-session", "never-existed", "--db", db])

    assert rc == 0
    assert "0 event(s) removed" in capsys.readouterr().out


def test_delete_session_empty_store_exits_zero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["delete-session", "anything", "--db", str(db)])

    assert rc == 0
    assert "0 event(s) removed" in capsys.readouterr().out


@_BACKENDS
def test_prune_keep_last_keeps_n_most_recent(tmp_path, backend, capsys):
    db, sids = _populate_multi(tmp_path, backend, count=4)

    rc = main(["prune", "--keep-last", "2", "--db", db])

    assert rc == 0
    remaining = [s.session_id for s in TraceLogger(db, backend=backend).list_sessions()]
    assert remaining == sids[2:]  # oldest two removed, two most recent kept


@_BACKENDS
def test_prune_keep_last_zero_removes_all(tmp_path, backend, capsys):
    db, _ = _populate_multi(tmp_path, backend, count=3)

    rc = main(["prune", "--keep-last", "0", "--db", db])

    assert rc == 0
    assert TraceLogger(db, backend=backend).list_sessions() == []


@_BACKENDS
def test_prune_older_than_removes_old_sessions(tmp_path, backend, capsys):
    db, sids = _populate_multi(tmp_path, backend, count=3)
    # Backdate the first (oldest) session to 2020 so it is unambiguously
    # older than 1 day; the other two stay "now".
    _backdate(tmp_path / f"traces{_suffix(backend)}", backend, sids[0], _OLDEST_TS)

    rc = main(["prune", "--older-than", "1", "--db", db])

    assert rc == 0
    remaining = [s.session_id for s in TraceLogger(db, backend=backend).list_sessions()]
    assert sids[0] not in remaining
    assert sids[1] in remaining
    assert sids[2] in remaining


@_BACKENDS
def test_prune_dry_run_does_not_mutate(tmp_path, backend, capsys):
    db, sids = _populate_multi(tmp_path, backend, count=4)

    rc = main(["prune", "--keep-last", "2", "--dry-run", "--db", db])

    assert rc == 0
    out = capsys.readouterr().out
    assert "would delete" in out
    assert "Dry run" in out
    # Store untouched after a dry run.
    remaining = [s.session_id for s in TraceLogger(db, backend=backend).list_sessions()]
    assert remaining == sids


@_BACKENDS
def test_prune_empty_store_reports_nothing(tmp_path, backend, capsys):
    path = tmp_path / f"traces{_suffix(backend)}"
    TraceLogger(path, backend=backend)

    rc = main(["prune", "--keep-last", "3", "--db", str(path)])

    assert rc == 0
    assert "Nothing to prune" in capsys.readouterr().out


def test_prune_neither_flag_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["prune", "--db", str(db)])

    assert rc == 1
    assert "specify --keep-last or --older-than" in capsys.readouterr().err


def test_prune_both_flags_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["prune", "--keep-last", "1", "--older-than", "5", "--db", str(db)])

    assert rc == 1
    assert "not both" in capsys.readouterr().err


def test_prune_negative_keep_last_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["prune", "--keep-last", "-1", "--db", str(db)])

    assert rc == 1
    assert "must be >= 0" in capsys.readouterr().err


def test_prune_nonpositive_older_than_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["prune", "--older-than", "0", "--db", str(db)])

    assert rc == 1
    assert "positive integer" in capsys.readouterr().err


# --- Issue #896: prune --vacuum reclaims WAL space ---------------------------
# SQLite's WAL accumulates deleted pages across pruning cycles; ``--vacuum``
# runs ``VACUUM`` + ``PRAGMA wal_checkpoint(TRUNCATE)`` so the ``-wal``
# sidecar stays bounded relative to the live data.


def _populate_many_sessions(
    path: Path, count: int, events_per: int = 8, blob_size: int = 256
) -> list[str]:
    """Seed *count* sessions each with several events carrying a payload.

    The payload gives each DELETE real pages to free so the WAL-side
    reclaim is observable on disk rather than rounding away to nothing
    on tiny databases.
    """
    logger = TraceLogger(path, backend="sqlite")
    blob = "x" * blob_size
    sids: list[str] = []
    for _ in range(count):
        with logger.session(harness_version="0.1.0") as sid:
            for _ in range(events_per):
                logger.record(sid, "tool_call", {"name": "read_file", "blob": blob})
            sids.append(sid)
    return sids


def _wal_size(path: Path) -> int:
    wal = path.with_suffix(path.suffix + "-wal")
    return wal.stat().st_size if wal.exists() else 0


def test_prune_vacuum_keeps_wal_under_2x_db(tmp_path, capsys):
    """Acceptance criterion #3 for issue #896 (1000→900 path)."""
    db = tmp_path / "traces.db"
    _populate_many_sessions(db, count=1000, events_per=4, blob_size=128)

    rc = main(["prune", "--keep-last", "100", "--vacuum", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Deleted 900 session(s)" in out
    assert "WAL space reclaimed" in out
    # Vacuum truncates the WAL sidecar, so it must be smaller than 2x the
    # live database file — the regression that motivated issue #896.
    assert _wal_size(db) < 2 * db.stat().st_size


def test_prune_vacuum_shrinks_wal_relative_to_no_vacuum(tmp_path):
    """Running ``--vacuum`` must not leave the WAL larger than without it."""
    db = tmp_path / "traces.db"
    sids = _populate_many_sessions(db, count=200, events_per=8, blob_size=256)

    # Prune 180 without vacuum and snapshot the WAL size.
    logger = TraceLogger(db, backend="sqlite")
    logger.prune_sessions(sids[:180])
    wal_without_vacuum = _wal_size(db)
    logger.close()

    # Prune the remaining 20 with vacuum and confirm the WAL is no larger.
    logger = TraceLogger(db, backend="sqlite")
    logger.prune_sessions(sids[180:], vacuum=True)
    wal_with_vacuum = _wal_size(db)
    logger.close()

    assert wal_with_vacuum <= wal_without_vacuum


def test_prune_vacuum_without_deletes_is_backward_compatible(tmp_path, capsys):
    """``--vacuum`` on an empty prune result must not run VACUUM (criterion #2)."""
    db = tmp_path / "traces.db"
    TraceLogger(db, backend="sqlite")

    rc = main(["prune", "--keep-last", "99", "--vacuum", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Nothing to prune" in out
    # No sessions were deleted, so the vacuum branch must not have printed.
    assert "WAL space reclaimed" not in out


@_BACKENDS
def test_prune_vacuum_jsonl_is_noop_message(tmp_path, backend, capsys):
    """``--vacuum`` must not break the JSONL backend (and must not claim to vacuum it)."""
    if backend != "jsonl":
        pytest.skip("jsonl-only assertion")
    db, _ = _populate_multi(tmp_path, backend, count=3)

    rc = main(["prune", "--keep-last", "1", "--vacuum", "--db", db])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Deleted 2 session(s)" in out
    # SQLite-only message; must not appear for the JSONL backend.
    assert "WAL space reclaimed" not in out


def test_prune_vacuum_flag_defaults_off(tmp_path):
    """Without ``--vacuum`` the ``TraceLogger.prune_sessions`` API is unchanged."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db, backend="sqlite")
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "tool_call", {"name": "read_file"})

    # No vacuum kwarg → backward-compatible signature.
    deleted = logger.prune_sessions([sid])

    assert deleted == 1
    assert logger.list_sessions() == []
    logger.close()


# --- Issue #632: compact ----------------------------------------------------
# Rewrites JSONL file removing orphaned session_end markers (session_end
# without a corresponding session_start).


def _write_jsonl(path: Path, lines: list[dict]) -> None:
    """Write a list of dicts as JSONL, one JSON object per line."""
    with path.open("w", encoding="utf-8") as fh:
        for obj in lines:
            fh.write(json.dumps(obj) + "\n")


def test_compact_jsonl_no_orphans_leaves_file_unchanged(tmp_path, capsys):
    """When no orphaned markers exist, compact is a no-op."""
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, "tool_call", {"name": "read_file"})

    original_content = db.read_text("utf-8")
    rc = main(["compact", "--db", str(db)])

    assert rc == 0
    assert "No orphaned markers found" in capsys.readouterr().out
    assert db.read_text("utf-8") == original_content


def test_compact_jsonl_removes_orphaned_session_end(tmp_path, capsys):
    """Orphaned session_end markers (no matching session_start) are removed."""
    db = tmp_path / "traces.jsonl"
    _write_jsonl(
        db,
        [
            {"kind": "session_end", "session_id": "orphan-sid", "ended_at": "2025-01-01T00:00:00Z"},
            {
                "kind": "session_start",
                "session_id": "valid-sid",
                "started_at": "2025-01-01T00:00:00Z",
                "harness_version": "0.1.0",
            },
            {
                "event_id": "e1",
                "session_id": "valid-sid",
                "timestamp": "2025-01-01T00:00:01Z",
                "kind": "tool_call",
                "payload": {},
            },
        ],
    )

    rc = main(["compact", "--db", str(db)])

    assert rc == 0
    assert "Removed 1 orphaned marker" in capsys.readouterr().out
    remaining = db.read_text("utf-8")
    assert "orphan-sid" not in remaining
    assert "valid-sid" in remaining


def test_compact_jsonl_dry_run_does_not_modify_file(tmp_path, capsys):
    """--dry-run prints what would be removed without touching the file."""
    db = tmp_path / "traces.jsonl"
    _write_jsonl(
        db,
        [
            {"kind": "session_end", "session_id": "orphan-sid", "ended_at": "2025-01-01T00:00:00Z"},
            {
                "kind": "session_start",
                "session_id": "valid-sid",
                "started_at": "2025-01-01T00:00:00Z",
                "harness_version": "0.1.0",
            },
        ],
    )

    original_content = db.read_text("utf-8")
    rc = main(["compact", "--dry-run", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "would remove" in out
    assert "orphan-sid" in out
    assert "Dry run" in out
    assert db.read_text("utf-8") == original_content


def test_compact_sqlite_is_noop(tmp_path, capsys):
    """compact on sqlite backend exits 0 with a message (VACUUM is automatic)."""
    db = tmp_path / "traces.db"
    TraceLogger(db, backend="sqlite")

    rc = main(["compact", "--db", str(db)])

    assert rc == 0
    assert "jsonl backend required" in capsys.readouterr().out


def test_compact_nonexistent_jsonl_file(tmp_path, capsys):
    """compact on a non-existent JSONL file exits 0 gracefully."""
    db = tmp_path / "nonexistent.jsonl"
    assert not db.exists()

    rc = main(["compact", "--db", str(db)])

    assert rc == 0
    assert "No orphaned markers found" in capsys.readouterr().out


# --- Issue #195: seed-sample-trace -------------------------------------------
# Offline smoke subcommand — plants a deterministic session so the Digester,
# KPI, and regression-report CLIs have something to chew on without standing
# up llama-server. Mirrors the event vocabulary the runner emits.


_REQUIRED_SEED_KINDS: tuple[str, ...] = (
    "task_received",
    "user_prompt",
    "model_request",
    "model_response",
    "tool_call",
    "tool_result",
    "outcome",
)


def _parse_seeded_session_id(stdout: str) -> str:
    """Extract the ``session_id`` printed by ``seed-sample-trace``.

    The CLI emits ``seeded session_id=<uuid>`` on success; this helper
    trims that down to just the UUID so assertions stay readable.
    """
    for line in stdout.splitlines():
        line = line.strip()
        if line.startswith("seeded session_id="):
            return line.split("=", 1)[1]
    raise AssertionError(f"no seeded session_id line in stdout:\n{stdout}")


def test_seed_sample_trace_plants_required_event_kinds(tmp_path, capsys):
    """Acceptance criterion: planted session has all 7 required event kinds.

    Issue #195 calls out ``task_received``, ``user_prompt``,
    ``model_request``, ``model_response``, ``tool_call``, ``tool_result``,
    and ``outcome`` as the kinds a real run emits and the offline seed must
    reproduce. Every one of them must be present in the planted session.
    """
    db = tmp_path / "traces.db"
    assert not db.exists()

    rc = main(["seed-sample-trace", "--db", str(db)])

    assert rc == 0
    captured = capsys.readouterr()
    session_id = _parse_seeded_session_id(captured.out)

    events = TraceLogger(db).load_session(session_id)
    kinds = [event.kind for event in events]

    # Acceptance criterion: ">= 6 events".
    assert len(events) >= 6
    for required_kind in _REQUIRED_SEED_KINDS:
        assert required_kind in kinds, f"missing required kind {required_kind!r}: {kinds}"

    # Kind order in the planted session follows the runner's emission order
    # so the timeline renderer shows a faithful shape.
    assert kinds.index("task_received") < kinds.index("user_prompt")
    assert kinds.index("user_prompt") < kinds.index("model_request")
    assert kinds.index("model_request") < kinds.index("model_response")
    assert kinds.index("model_response") < kinds.index("tool_call")
    assert kinds.index("tool_call") < kinds.index("tool_result")
    assert kinds.index("tool_result") < kinds.index("outcome")


def test_seed_sample_trace_creates_persistent_session_row(tmp_path, capsys):
    """Seeded session lands in the sessions table with the expected metadata.

    The KPI CLI filters by ``harness_version`` and the regression-report
    CLI iterates sessions via ``list_sessions``. Both must see the planted
    session, so the ``session()`` context manager must have committed the
    start-of-life row before ``record()`` calls landed.
    """
    db = tmp_path / "traces.db"

    rc = main(["seed-sample-trace", "--db", str(db)])

    assert rc == 0
    captured = capsys.readouterr()
    session_id = _parse_seeded_session_id(captured.out)

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    planted = sessions[0]
    assert planted.session_id == session_id
    # Default harness version is "seed-sample" so the planted row is
    # trivially filterable by the KPI / regression-report CLIs.
    assert planted.harness_version == "seed-sample"
    assert planted.model_id == "seeded-llama-sample"
    # ``session()`` sets ended_at on exit; the seeded session is closed
    # before ``record()`` is called, so ended_at is populated.
    assert planted.ended_at is not None


def test_seed_sample_trace_honors_harness_version_override(tmp_path, capsys):
    """``--harness-version`` lets the seeded session simulate a candidate.

    Issue #195 acceptance criterion: ``--harness-version`` flag lets the
    seeded session simulate a candidate version. The compare-kpis CLI
    filters sessions by harness version, so the planted row must surface
    under the override value.
    """
    db = tmp_path / "traces.db"

    rc = main(
        [
            "seed-sample-trace",
            "--db",
            str(db),
            "--harness-version",
            "1.0.0-candidate",
        ]
    )

    assert rc == 0

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    assert sessions[0].harness_version == "1.0.0-candidate"

    only_candidate = TraceLogger(db).list_sessions(harness_version="1.0.0-candidate")
    assert len(only_candidate) == 1
    assert only_candidate[0].session_id == sessions[0].session_id


def test_seed_sample_trace_payloads_are_secret_free(tmp_path):
    """Acceptance criterion: planted payloads contain no real secrets.

    SECURITY.md §Secrets forbids persisting raw tokens, keys, or PEM
    blocks. The seed command plants synthetic placeholder content; this
    test walks the planted session and asserts the redaction scanners
    matched nothing. A regression that pulled real credentials into the
    seed would silently leak them into ``logs/traces.db`` on every
    developer machine, so we pin the property here.
    """
    from foundry_x.trace.logger import _redact, _redact_value

    db = tmp_path / "traces.db"
    rc = main(["seed-sample-trace", "--db", str(db)])
    assert rc == 0

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1
    sid = sessions[0].session_id

    # Sanity check the redaction helper: a sentinel token MUST be redacted
    # so the test below is actually proving something (i.e. we'd catch a
    # regression in the redaction layer too).
    assert "[REDACTED" in _redact_value("sk-abcdefghijklmnopqrstuvwxyz")
    assert _redact({"api_key": "anything"}) == {"api_key": "[REDACTED:secret]"}

    for event in TraceLogger(db).load_session(sid):
        redacted_payload = _redact(event.payload)
        # ``_redact`` is idempotent: re-applying it to a payload that
        # already contains no secret-shaped substrings must return the
        # same shape. A regression that introduced a token would surface
        # as ``[REDACTED:...]`` markers on the second pass.
        assert redacted_payload == _redact(redacted_payload)
        for value in _iter_payload_strings(redacted_payload):
            assert "[REDACTED" not in value, (
                f"seed planted a secret-shaped substring in {event.kind!r}: {value!r}"
            )


def _iter_payload_strings(payload):
    """Yield every string value nested inside ``payload``.

    Walks dicts and lists recursively so a deeply-nested tool-call
    argument cannot hide a secret-shaped substring from the scanner.
    """
    if isinstance(payload, dict):
        for value in payload.values():
            yield from _iter_payload_strings(value)
    elif isinstance(payload, list):
        for item in payload:
            yield from _iter_payload_strings(item)
    elif isinstance(payload, str):
        yield payload


def test_seed_sample_trace_creates_parent_directory(tmp_path):
    """``--db`` path is created if its parent does not yet exist.

    Mirrors the ``TraceLogger`` constructor's behavior so a developer can
    point the seed at a fresh ``logs/traces.db`` without mkdir-ing first.
    """
    nested_db = tmp_path / "deep" / "nested" / "traces.db"
    assert not nested_db.parent.exists()

    rc = main(["seed-sample-trace", "--db", str(nested_db)])

    assert rc == 0
    assert nested_db.exists()
    sessions = TraceLogger(nested_db).list_sessions()
    assert len(sessions) == 1


def test_seed_sample_trace_is_visible_to_existing_subcommands(tmp_path, capsys):
    """``session-list`` and ``session-show`` see the planted session.

    The point of the seed is to give downstream CLIs something to chew on
    — this is the end-to-end check that the planted row is consumable.
    """
    db = tmp_path / "traces.db"

    rc = main(["seed-sample-trace", "--db", str(db)])
    assert rc == 0
    session_id = _parse_seeded_session_id(capsys.readouterr().out)

    rc = main(["session-list", "--db", str(db)])
    assert rc == 0
    listing = capsys.readouterr().out
    assert session_id in listing
    assert "seed-sample" in listing

    rc = main(["session-show", session_id, "--db", str(db)])
    assert rc == 0
    timeline = capsys.readouterr().out
    for required_kind in _REQUIRED_SEED_KINDS:
        assert required_kind in timeline
