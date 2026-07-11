"""Connection-cache and WAL-mode tests for ``TraceLogger`` (issue #84).

Phase 2 of the roadmap runs the Runner (writer) and the Critic (reader)
concurrently against the same trace file. Without WAL journal mode the
default sqlite3 rollback journal serialises writers and surfaces
``SQLITE_BUSY`` to the Critic, which silently regresses the
Digester -> Evolver -> Critic loop. These tests pin the contract:

- every sqlite connection opened by ``TraceLogger`` has ``journal_mode=WAL``
  and ``synchronous=NORMAL`` set on first connect (issue acceptance criteria);
- multiple ``TraceLogger`` instances on the same path reuse a single shared
  ``sqlite3.Connection`` (the cache key is ``(str(path), backend)``);
- re-instantiation does not lock: a writer and a reader on the same file
  interleave without ``SQLITE_BUSY`` and the reader observes committed
  writes within 100 ms (the issue acceptance criterion);
- the last releasing TraceLogger closes the underlying connection.

Each test uses ``tmp_path`` so cache keys are unique across tests and we
do not need to clear the module-level cache between tests.
"""

from __future__ import annotations

import sqlite3
import threading
import time

import pytest

from foundry_x.trace import logger as trace_logger
from foundry_x.trace.logger import TraceLogger


def _journal_mode(db_path) -> str:
    """Read the journal_mode pragma directly from the file on disk.

    Uses a fresh connection (not the cached one) so the assertion holds even
    if the TraceLogger under test has been garbage-collected and the cache
    entry already closed. SQLite persists ``journal_mode=WAL`` in the file
    header, so the value is durable across reconnects.
    """
    with sqlite3.connect(db_path) as conn:
        return conn.execute("PRAGMA journal_mode").fetchone()[0]


def test_sqlite_connection_runs_in_wal_mode(tmp_path):
    """The acceptance criterion: WAL is set on first connect (issue #84)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db, backend="sqlite")

    try:
        cached = logger._sqlite
        assert cached is not None, "sqlite backend must populate _sqlite"
        mode = cached.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode.lower() == "wal", f"expected WAL, got {mode!r}"

        # And the pragma is durable in the file header, so reopening with
        # the stdlib sqlite3 module still reports WAL.
        assert _journal_mode(db) == "wal"
    finally:
        # Hold the reference until we've asserted; the test fixture's tmp_path
        # cleanup will still work because __del__ closes the connection.
        del logger


def test_sqlite_connection_sets_synchronous_normal(tmp_path):
    """The issue also calls for ``synchronous=NORMAL`` alongside WAL."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db, backend="sqlite")

    try:
        cached = logger._sqlite
        assert cached is not None
        sync = cached.conn.execute("PRAGMA synchronous").fetchone()[0]
        assert sync == 1, (
            f"expected synchronous=1 (NORMAL), got {sync}; "
            f"0=FULL, 1=NORMAL, 2=OFF per https://www.sqlite.org/pragma.html#pragma_synchronous"
        )
    finally:
        del logger


def test_multiple_instances_share_one_sqlite_connection(tmp_path):
    """Two TraceLoggers on the same path share the cached Connection."""
    db = tmp_path / "traces.db"
    writer = TraceLogger(db, backend="sqlite")
    reader = TraceLogger(db, backend="sqlite")

    try:
        assert writer._sqlite is reader._sqlite, (
            "issue #84: TraceLogger instances on the same (path, backend) "
            "must share a single cached sqlite3.Connection"
        )
        assert writer._sqlite.conn is reader._sqlite.conn
        # refcount tracks the live holders so the cache only closes when both
        # have been released.
        assert writer._sqlite.refcount == 2
    finally:
        del reader
        del writer


def test_distinct_paths_get_distinct_connections(tmp_path):
    """Different paths must NOT share a connection (different files)."""
    db_a = tmp_path / "a.db"
    db_b = tmp_path / "b.db"
    a = TraceLogger(db_a, backend="sqlite")
    b = TraceLogger(db_b, backend="sqlite")

    try:
        assert a._sqlite is not b._sqlite
        assert a._sqlite.lock is not b._sqlite.lock
    finally:
        del b
        del a


def test_sqlite_and_jsonl_backends_get_distinct_cache_entries(tmp_path):
    """Same path but different backends -> separate cache entries.

    The jsonl backend never opens sqlite, so this is really only meaningful
    if we ever added a non-sqlite backend that did. For now we just confirm
    the jsonl path leaves ``_sqlite`` unset on the instance.
    """
    db = tmp_path / "traces"
    sqlite_logger = TraceLogger(db, backend="sqlite")
    jsonl_logger = TraceLogger(db, backend="jsonl")

    try:
        assert sqlite_logger._sqlite is not None
        assert jsonl_logger._sqlite is None
    finally:
        del jsonl_logger
        del sqlite_logger


def test_releasing_last_instance_closes_cached_connection(tmp_path):
    """The connection is closed when the last holder releases it."""
    db = tmp_path / "traces.db"
    a = TraceLogger(db, backend="sqlite")
    cached = a._sqlite
    assert cached is not None

    b = TraceLogger(db, backend="sqlite")
    assert cached.refcount == 2

    del a
    # b still holds a reference -> connection stays open.
    assert cached.refcount == 1
    assert _CONNECTION_KEY_PRESENT(db, "sqlite")

    del b
    # After the last holder releases, the cache entry is evicted and closed.
    assert not _CONNECTION_KEY_PRESENT(
        db, "sqlite"
    ), "issue #84: cache entry must be evicted once refcount hits 0"


def _CONNECTION_KEY_PRESENT(path, backend: str) -> bool:
    """Test-only helper: is (str(path), backend) currently in the cache?"""
    return (str(path), backend) in trace_logger._CONNECTION_CACHE


def test_reinstantiation_does_not_lock_writer_and_reader(tmp_path):
    """Issue #84 acceptance: writer + reader on the same file cooperate.

    Re-instantiation (a second TraceLogger on the same path) must not lock:
    the reader must observe committed writes within 100 ms without ever
    raising ``SQLITE_BUSY``. In WAL mode SQLite allows one writer and many
    readers concurrently, so this passes as long as our write lock is held
    only for the actual write transaction — never across the reader's
    SELECT.
    """
    db = tmp_path / "traces.db"
    writer = TraceLogger(db, backend="sqlite")
    reader = TraceLogger(db, backend="sqlite")

    try:
        with writer.session(harness_version="0.1.0", model_id="m") as sid:
            writer.record(sid, kind="user_prompt", payload={"prompt": "hi"})

        observed: list[str] = []
        busy_errors: list[sqlite3.OperationalError] = []
        stop = threading.Event()

        def reader_loop(rdr: TraceLogger) -> None:
            while not stop.is_set():
                try:
                    sessions = rdr.list_sessions()
                    observed.extend(s.session_id for s in sessions)
                except sqlite3.OperationalError as exc:  # pragma: no cover
                    if "database is locked" in str(exc) or "SQLITE_BUSY" in str(exc):
                        busy_errors.append(exc)
                time.sleep(0.001)

        thread = threading.Thread(target=reader_loop, args=(reader,), daemon=True)
        thread.start()

        try:
            # Write a few sessions + events while the reader thread spins.
            for i in range(20):
                with writer.session(harness_version=f"0.1.{i}") as sid:
                    writer.record(sid, kind="user_prompt", payload={"i": i})
                    writer.record(sid, kind="tool_call", payload={"name": "read_file"})
                # Reader must see the just-committed session before the budget.
                wait_deadline = time.monotonic() + 0.1
                while time.monotonic() < wait_deadline:
                    if any(s.harness_version == f"0.1.{i}" for s in reader.list_sessions()):
                        break
                    time.sleep(0.001)
                else:  # pragma: no cover
                    pytest.fail(
                        f"reader did not observe session 0.1.{i} within 100 ms; "
                        f"WAL or write-lock semantics appear to be blocking reads."
                    )
        finally:
            stop.set()
            thread.join(timeout=1.0)

        assert not busy_errors, f"reader saw SQLITE_BUSY {len(busy_errors)} time(s); WAL is not in effect: {busy_errors[:1]}"
        # Sanity: the reader actually saw sessions written by the writer.
        assert observed, "reader thread never observed any sessions"
    finally:
        del reader
        del writer


def test_writer_commit_failure_rolls_back_transaction(tmp_path):
    """An exception inside ``_write_conn`` must roll back, not silently commit
    half the work. AGENTS.md forbids silently swallowing exceptions, and
    dropping a partial write would silently corrupt the trace store."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db, backend="sqlite")

    try:
        cached = logger._sqlite
        assert cached is not None

        with pytest.raises(sqlite3.IntegrityError):
            with logger._write_conn() as conn:
                # Insert a session row, then violate the PRIMARY KEY on the
                # second insert. The first must roll back along with the second.
                conn.execute(
                    "INSERT INTO sessions (session_id, started_at, harness_version) "
                    "VALUES (?, ?, ?)",
                    ("dup-id", "2026-07-10T00:00:00+00:00", "0.0.1"),
                )
                conn.execute(
                    "INSERT INTO sessions (session_id, started_at, harness_version) "
                    "VALUES (?, ?, ?)",
                    ("dup-id", "2026-07-10T00:00:01+00:00", "0.0.1"),
                )

        rows = cached.conn.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?", ("dup-id",)
        ).fetchall()
        assert (
            rows == []
        ), f"the failed transaction must roll back; found {rows!r} in sessions table"
    finally:
        del logger


def test_jsonl_logger_does_not_open_sqlite(tmp_path):
    """Sanity check: the jsonl backend never touches the sqlite cache.

    Guards against a future refactor accidentally wiring jsonl through the
    sqlite connection.
    """
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    try:
        assert logger._sqlite is None
        with logger.session(harness_version="0.1.0") as sid:
            logger.record(sid, "user_prompt", {"prompt": "hi"})
        sessions = logger.list_sessions()
        assert len(sessions) == 1
        assert sessions[0].session_id == sid
    finally:
        del logger
