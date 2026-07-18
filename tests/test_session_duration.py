"""Session ``ended_at`` and wall-clock duration (issue #8).

Acceptance per issue #8:
- Entering and exiting ``session()`` stamps a non-null ``ended_at`` on both
  sqlite and jsonl backends, and ``session_duration(sid) > 0``.
- A session whose body raises still receives an ``ended_at``.
- A pre-#8 ``logs/*.db`` (no ``ended_at`` column) migrates without error
  when re-opened.
- ``session_duration()`` returns ``None`` for unknown / un-ended sessions.
"""

from __future__ import annotations

import sqlite3
import time

import pytest

from foundry_x.trace.logger import TraceLogger

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def _suffix(backend: str) -> str:
    return ".db" if backend == "sqlite" else ".jsonl"


@_BACKENDS
def test_session_records_ended_at(tmp_path, backend):
    logger = TraceLogger(tmp_path / f"traces{_suffix(backend)}", backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        pass

    sessions = logger.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == sid
    assert sessions[0].ended_at is not None


@_BACKENDS
def test_session_duration_is_positive(tmp_path, backend):
    logger = TraceLogger(tmp_path / f"traces{_suffix(backend)}", backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        time.sleep(0.01)

    duration = logger.session_duration(sid)
    assert duration is not None
    assert duration.total_seconds() > 0


def test_sqlite_session_duration_uses_targeted_query(tmp_path, monkeypatch):
    logger = TraceLogger(tmp_path / "traces.db")
    with logger.session(harness_version="test-0.0") as sid:
        time.sleep(0.01)

    statements = []
    assert logger._conn is not None
    logger._conn.set_trace_callback(statements.append)
    monkeypatch.setattr(
        logger,
        "list_sessions",
        lambda: pytest.fail("session_duration must not list sessions"),
    )

    duration = logger.session_duration(sid)

    assert duration is not None
    assert duration.total_seconds() > 0
    selects = [statement for statement in statements if statement.startswith("SELECT ")]
    assert len(selects) == 1
    assert "FROM sessions WHERE session_id =" in selects[0]
    assert sid in selects[0]


@_BACKENDS
def test_failed_session_body_still_records_ended_at(tmp_path, backend):
    logger = TraceLogger(tmp_path / f"traces{_suffix(backend)}", backend=backend)

    with pytest.raises(RuntimeError, match="boom"):
        with logger.session(harness_version="test-0.0") as sid:
            raise RuntimeError("boom")

    sessions = logger.list_sessions()
    matching = [s for s in sessions if s.session_id == sid]
    assert len(matching) == 1
    assert matching[0].ended_at is not None


def test_pre_issue8_db_migrates_without_error(tmp_path):
    db = tmp_path / "legacy.db"
    # Construct a pre-#8 database: old schema, no ``ended_at`` column.
    with sqlite3.connect(db) as conn:
        conn.executescript(
            """
            CREATE TABLE sessions (
                session_id TEXT PRIMARY KEY,
                started_at TEXT NOT NULL,
                harness_version TEXT NOT NULL,
                model_id TEXT,
                metadata TEXT
            );
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                kind TEXT NOT NULL,
                payload TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(session_id)
            );
            CREATE INDEX idx_events_session ON events(session_id);
            INSERT INTO sessions VALUES
                ('legacy-1', '2026-07-01T00:00:00+00:00', '0.0.1', NULL, '{}');
            """
        )

    # Re-opening must add the column non-destructively.
    logger = TraceLogger(db)

    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        row = conn.execute(
            "SELECT session_id, ended_at FROM sessions WHERE session_id = ?",
            ("legacy-1",),
        ).fetchone()
    assert "ended_at" in cols
    assert row == ("legacy-1", None)

    # The legacy session's pre-existing data is intact and queryable.
    sessions = logger.list_sessions()
    assert any(s.session_id == "legacy-1" for s in sessions)
    assert logger.session_duration("legacy-1") is None


def test_session_duration_unknown_session_returns_none(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    assert logger.session_duration("does-not-exist") is None


def test_open_session_has_no_duration(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    # Simulate an open session by writing only the start row directly.
    with logger.session(harness_version="test-0.0") as sid:
        assert logger.session_duration(sid) is None
