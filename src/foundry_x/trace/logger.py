"""Trace logger for FoundryX.

This module is the only writer to the trace store. Both backends (sqlite
and jsonl) MUST scrub secret-like substrings from every value they
persist, including the ``metadata`` dict that an Operator passes to
:class:`TraceLogger.session`. See ``docs/SECURITY.md`` §Secrets
(lines 61-69) for the policy, ADR-0003 for the trace-store rationale,
and the ``_redact`` / ``_redact_value`` helpers below for the
implementation. Issue #121 extended the original layer (issue #3) to
cover modern token formats and the previously-untouched metadata path.
"""

from __future__ import annotations

import json
import re
import sqlite3
import sys
import threading
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence

from pydantic import BaseModel, Field


class TraceEvent(BaseModel):
    event_id: str
    session_id: str
    timestamp: str
    kind: str = Field(min_length=1)
    # ``Any`` is justified here per ADR-0006: the payload is the
    # serialization-boundary free-form dict whose schema is owned by the
    # event producer (the closed ``kind`` vocabulary arrives in Phase 2).
    payload: dict[str, Any]


class TraceSession(BaseModel):
    session_id: str
    started_at: str
    harness_version: str
    model_id: str | None = None
    # ``Any`` per ADR-0006 serialization-boundary carve-out (same rationale
    # as TraceEvent.payload).
    metadata: dict[str, Any] = Field(default_factory=dict)
    ended_at: str | None = None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    harness_version TEXT NOT NULL,
    model_id TEXT,
    metadata TEXT,
    ended_at TEXT
);
CREATE TABLE IF NOT EXISTS events (
    event_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id);
"""


# --- SQLite connection cache (issue #84) -------------------------------------
# Phase 2 runs the Runner (writer) and the Critic (reader) concurrently against
# the same trace file. The default rollback journal serialises writers and
# surfaces SQLITE_BUSY to readers, which is unacceptable for the
# Digester -> Evolver -> Critic loop. WAL mode lets one writer and many readers
# proceed without blocking each other, and ``synchronous=NORMAL`` keeps writes
# durable across process crashes without the fsync cost of FULL.
#
# Multiple ``TraceLogger`` instances on the same path (e.g. the CLI's ``main()``
# alongside an in-process runner) historically opened a fresh ``sqlite3.connect``
# per call. With WAL that would still work but waste file handles and force a
# PRAGMA re-application on every connect. The cache below lets every instance
# on the same ``(path, backend)`` share a single long-lived ``Connection``
# (with ``check_same_thread=False`` so threads are safe) plus a per-entry
# ``RLock`` that serialises writes. Readers do not take the lock, so concurrent
# reads proceed in parallel as WAL permits. The ``_CACHE_LOCK`` only guards the
# dict itself, never the cached connections.
#
# See ADR-0003 §Consequences (concurrent writers were the named revisit
# trigger) and the issue for the original trace evidence.

_CACHE_LOCK = threading.Lock()
_CONNECTION_CACHE: dict[tuple[str, str], "_CachedConnection"] = {}


@dataclass
class _CachedConnection:
    conn: sqlite3.Connection
    lock: threading.RLock
    refcount: int = 0


def _acquire_sqlite_connection(path: Path, backend: str) -> _CachedConnection:
    """Return a refcounted, thread-safe sqlite3 connection for ``(path, backend)``.

    The first caller for a given key opens the file with ``journal_mode=WAL``
    and ``synchronous=NORMAL`` and runs the schema migration. Subsequent
    callers reuse the same ``Connection`` and just bump the refcount. Each
    TraceLogger instance must pair this call with a ``_release_sqlite_connection``
    in ``__del__`` (or an explicit close path) so the connection is closed
    when the last referencing instance goes away.
    """
    key = (str(path), backend)
    with _CACHE_LOCK:
        cached = _CONNECTION_CACHE.get(key)
        if cached is not None:
            cached.refcount += 1
            return cached
        # ``check_same_thread=False`` because the connection is shared across
        # TraceLogger instances that may live on different threads; the
        # ``RLock`` on the cached entry provides the actual serialisation.
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        # Verify the WAL pragma actually took effect. If it did not (e.g. the
        # file lives on a read-only mount that rejected the journal-mode
        # switch), fail loudly rather than silently degrading to the default
        # rollback journal — that is the silent regression issue #84 is here
        # to prevent.
        actual_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        if actual_mode.lower() != "wal":
            conn.close()
            raise sqlite3.OperationalError(
                f"PRAGMA journal_mode=WAL did not take effect on {path!s} "
                f"(got {actual_mode!r}); concurrent-readers/writer safety "
                f"guaranteed by issue #84 cannot hold."
            )
        conn.executescript(_SCHEMA)
        # Non-destructive migration for pre-issue-#8 databases that predate
        # the ``ended_at`` column (issue #8). Guarded by a pragma check so
        # freshly-created databases are untouched and existing ``logs/*.db``
        # files do not break. Runs once per (path, backend) at first acquire.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "ended_at" not in columns:
            conn.execute("ALTER TABLE sessions ADD COLUMN ended_at TEXT")
        cached = _CachedConnection(conn=conn, lock=threading.RLock(), refcount=1)
        _CONNECTION_CACHE[key] = cached
        return cached


def _release_sqlite_connection(path: Path, backend: str) -> None:
    """Decrement the refcount; close and evict when the last holder releases."""
    key = (str(path), backend)
    with _CACHE_LOCK:
        cached = _CONNECTION_CACHE.get(key)
        if cached is None:
            return
        cached.refcount -= 1
        if cached.refcount <= 0:
            try:
                cached.conn.close()
            except sqlite3.Error:
                # Connection already closed externally (e.g. interpreter
                # shutdown ordering); nothing useful to do here.
                pass
            _CONNECTION_CACHE.pop(key, None)


# --- Secret redaction ---------------------------------------------------------
# Required by docs/SECURITY.md (lines 44-46, 68-69) and ADR-0003 (line 34).
# Patterns are matched against every string value in a payload before it is
# persisted to either backend, so that the trace store never holds raw
# credentials. The Digester still sees a `[REDACTED:<kind>]` sentinel, which
# preserves the signal that a secret *was* present without leaking its value.

_API_KEY_RE = re.compile(r"sk-[A-Za-z0-9_\-]{8,}")
_BEARER_RE = re.compile(r"(?i)bearer\s+[A-Za-z0-9\-._~+/]+=*")
_PEM_RE = re.compile(
    r"-----BEGIN (?:[A-Z ]*)PRIVATE KEY-----.*?-----END (?:[A-Z ]*)PRIVATE KEY-----",
    re.DOTALL,
)
# GitHub classic PATs (ghp_, gho_, ghs_, ghu_) and fine-grained PATs
# (github_pat_ plus the 11-char-suffix variant github_pat_11XXXX...). Issue
# #121 expands coverage beyond the original three regexes shipped with #3.
_GITHUB_CLASSIC_PAT_RE = re.compile(r"(?:ghp|gho|ghs|ghu)_[A-Za-z0-9]{20,}")
_GITHUB_FINE_GRAINED_PAT_RE = re.compile(r"github_pat_[A-Za-z0-9_]{20,}")
# JSON Web Tokens: three base64url segments separated by dots.
_JWT_RE = re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b")
# AWS access key IDs — fixed 20-char body prefixed with AKIA / ASIA.
_AWS_ACCESS_KEY_RE = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
# Stripe live keys (sk_live_, pk_live_, also restricted/sidecar variants).
_STRIPE_LIVE_KEY_RE = re.compile(r"\b(?:sk|pk|rk)_(?:live|restricted)_[A-Za-z0-9]{16,}")
# Slack tokens: xox[baprs]- followed by the segment body.
_SLACK_TOKEN_RE = re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}")

_DEFAULT_SECRET_KEY_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "aws_access_key_id",
        "aws_secret_access_key",
        "github_token",
        "anthropic_api_key",
        "openai_api_key",
        "access_token",
        "id_token",
        "jwt",
        "password",
        "passwd",
        "refresh_token",
        "secret",
        "secret_key",
        "slack_token",
        "stripe_key",
        "token",
    }
)


def _redact_value(value: str) -> str:
    """Mask secret-like substrings within a single string.

    Order matters only for readability: PEM blocks (which can contain
    ``-----BEGIN`` and ``sk-``-like substrings) are scrubbed first, then
    the remaining content-patterns. Token order is fixed across calls.
    """
    value = _PEM_RE.sub("[REDACTED:pem]", value)
    value = _JWT_RE.sub("[REDACTED:jwt]", value)
    value = _API_KEY_RE.sub("[REDACTED:api-key]", value)
    value = _GITHUB_CLASSIC_PAT_RE.sub("[REDACTED:github-pat]", value)
    value = _GITHUB_FINE_GRAINED_PAT_RE.sub("[REDACTED:github-pat]", value)
    value = _AWS_ACCESS_KEY_RE.sub("[REDACTED:aws-access-key]", value)
    value = _STRIPE_LIVE_KEY_RE.sub("[REDACTED:stripe-key]", value)
    value = _SLACK_TOKEN_RE.sub("[REDACTED:slack-token]", value)
    value = _BEARER_RE.sub("[REDACTED:bearer]", value)
    return value


def _redact(
    payload: Any,
    secret_key_names: frozenset[str] = _DEFAULT_SECRET_KEY_NAMES,
) -> Any:
    """Recursively scrub secret-like values from a payload.

    Returns a new structure; the input is not mutated. Dict keys whose
    lower-cased name is in ``secret_key_names`` have their entire value
    replaced with ``[REDACTED:secret]`` regardless of content; all other
    string values are scanned for ``sk-...``, ``Bearer ...``, PEM blocks,
    GitHub classic/fine-grained PATs, JWTs, AWS access key IDs, Stripe
    live keys, and Slack tokens. Issue #121 added the modern-token set
    and the metadata-path coverage.
    """
    if isinstance(payload, dict):
        redacted: dict[str, Any] = {}
        for key, val in payload.items():
            if isinstance(key, str) and key.lower() in secret_key_names:
                redacted[key] = "[REDACTED:secret]"
            else:
                redacted[key] = _redact(val, secret_key_names)
        return redacted
    if isinstance(payload, list):
        return [_redact(item, secret_key_names) for item in payload]
    if isinstance(payload, str):
        return _redact_value(payload)
    return payload


class TraceLogger:
    def __init__(self, path: str | Path, backend: str = "sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        # ``_sqlite`` is None for the jsonl backend, otherwise it holds the
        # shared cached connection + write lock for this ``(path, backend)``.
        # Multiple TraceLogger instances on the same path share the entry via
        # the module-level cache (issue #84), which is what lets the writer
        # (Runner) and the reader (CLI / Critic) coexist on one file without
        # reopening it on every call.
        self._sqlite: _CachedConnection | None = None
        if backend == "sqlite":
            self._sqlite = _acquire_sqlite_connection(self.path, backend)

    def __del__(self) -> None:
        # Release the cached connection when this TraceLogger is garbage-collected.
        # CPython guarantees ``__del__`` runs when the instance refcount hits 0,
        # which is the common case for short-lived ``TraceLogger(db_path)`` call
        # sites in CLI handlers and tests. Cyclic-reference cases fall back to
        # interpreter-shutdown cleanup, which is acceptable: the underlying file
        # handle is still released when the OS reaps the process.
        if self._sqlite is not None:
            _release_sqlite_connection(self.path, self.backend)
            self._sqlite = None

    @contextmanager
    def _write_conn(self) -> Iterator[sqlite3.Connection]:
        """Yield the shared sqlite connection under the per-(path, backend)
        write lock, committing on success and rolling back on exception.

        Equivalent to the previous ``with sqlite3.connect(self.path) as conn:``
        pattern: every write is its own short transaction so a single failing
        statement does not poison the next. Readers never enter this helper.
        """
        if self._sqlite is None:
            raise RuntimeError(
                "TraceLogger has no sqlite connection (backend="
                f"{self.backend!r}); did you mean to pass backend='sqlite'?"
            )
        with self._sqlite.lock:
            try:
                yield self._sqlite.conn
                self._sqlite.conn.commit()
            except Exception:
                self._sqlite.conn.rollback()
                raise

    @contextmanager
    def session(
        self,
        harness_version: str,
        model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[str]:
        # Issue #121: scrub the metadata dict before either backend writes it.
        # The original ``record()`` path already redacts its payload; the
        # ``session()`` start-of-life marker did not, so an Operator passing
        # ``metadata={'github_token': 'ghp_...'}`` would have persisted the raw
        # token. SECURITY.md §Secrets.
        redacted_metadata: dict[str, Any] = _redact(metadata) if metadata else {}
        session_id = str(uuid.uuid4())
        if self.backend == "jsonl":
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "session_id": session_id,
                            "started_at": _now(),
                            "harness_version": harness_version,
                            "model_id": model_id,
                            "metadata": redacted_metadata,
                            "kind": "session_start",
                        }
                    )
                    + "\n"
                )
        else:
            with self._write_conn() as conn:
                conn.execute(
                    "INSERT INTO sessions "
                    "(session_id, started_at, harness_version, model_id, metadata) "
                    "VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        _now(),
                        harness_version,
                        model_id,
                        json.dumps(redacted_metadata),
                    ),
                )
        try:
            yield session_id
        finally:
            # Record the wall-clock end timestamp even when the body raised,
            # so a failed session still gets an ``ended_at`` (issue #8). This
            # is the primitive the PRD cycle-time KPI and SECURITY.md runaway
            # detection build on.
            self._end_session(session_id)

    def _end_session(self, session_id: str) -> None:
        """Stamp ``ended_at`` on session exit (issue #8).

        Writes to the ``sessions`` table for sqlite, or appends a
        ``session_end`` marker line for jsonl. Per AGENTS.md we never
        silently swallow exceptions, so when the ``session()`` body
        completed cleanly any write error is re-raised (it indicates a
        real problem with the trace store). When the body is already
        unwinding for an exception, re-raising here would mask the
        caller's original error, so in that one case the write error is
        suppressed — leaving ``ended_at`` null, which degrades gracefully
        downstream rather than corrupting the caller's stack trace.
        """
        ended_at = _now()
        masking_active_exception = sys.exc_info()[1] is not None
        try:
            if self.backend == "jsonl":
                with self.path.open("a", encoding="utf-8") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "session_id": session_id,
                                "ended_at": ended_at,
                                "kind": "session_end",
                            }
                        )
                        + "\n"
                    )
            else:
                with self._write_conn() as conn:
                    conn.execute(
                        "UPDATE sessions SET ended_at = ? WHERE session_id = ?",
                        (ended_at, session_id),
                    )
        except Exception:
            if masking_active_exception:
                return
            raise

    def session_duration(self, session_id: str) -> timedelta | None:
        """Wall-clock duration of a session, or ``None`` if not yet ended.

        Returns ``ended_at - started_at`` parsed from ISO-8601 timestamps.
        ``None`` means the session is still open, has no recorded
        ``ended_at`` (e.g. a pre-#8 row never re-opened), or the stored
        timestamps could not be parsed. Intended for the Digester and the
        trace CLI.
        """
        for session in self.list_sessions():
            if session.session_id != session_id:
                continue
            if session.ended_at is None:
                return None
            try:
                start = datetime.fromisoformat(session.started_at)
                end = datetime.fromisoformat(session.ended_at)
            except ValueError:
                return None
            return end - start
        return None

    def record(
        self,
        session_id: str,
        kind: str,
        payload: dict[str, Any],
    ) -> TraceEvent:
        event = TraceEvent(
            event_id=str(uuid.uuid4()),
            session_id=session_id,
            timestamp=_now(),
            kind=kind,
            payload=_redact(payload),
        )
        if self.backend == "sqlite":
            data = event.model_dump()
            with self._write_conn() as conn:
                conn.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                    (
                        data["event_id"],
                        data["session_id"],
                        data["timestamp"],
                        data["kind"],
                        json.dumps(data["payload"]),
                    ),
                )
        elif self.backend == "jsonl":
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(event.model_dump_json() + "\n")
        return event

    def load_session(self, session_id: str) -> Sequence[TraceEvent]:
        if self.backend == "jsonl":
            return self._load_session_jsonl(session_id)
        return self._load_session_sqlite(session_id)

    def list_sessions(self) -> list[TraceSession]:
        if self.backend == "jsonl":
            return self._list_sessions_jsonl()
        return self._list_sessions_sqlite()

    def _list_sessions_sqlite(self) -> list[TraceSession]:
        assert self._sqlite is not None
        conn = self._sqlite.conn
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)").fetchall()}
        selected = [
            "session_id",
            "started_at",
            "harness_version",
            "model_id",
            "metadata",
        ]
        if "ended_at" in columns:
            selected.append("ended_at")
        query = "SELECT " + ", ".join(selected) + " FROM sessions ORDER BY started_at"
        rows = conn.execute(query).fetchall()
        sessions: list[TraceSession] = []
        for row in rows:
            values = dict(zip(selected, row))
            sessions.append(
                TraceSession(
                    session_id=values["session_id"],
                    started_at=values["started_at"],
                    harness_version=values["harness_version"],
                    model_id=values.get("model_id"),
                    metadata=json.loads(values.get("metadata") or "{}"),
                    ended_at=values.get("ended_at"),
                )
            )
        return sessions

    def _list_sessions_jsonl(self) -> list[TraceSession]:
        # Two marker kinds share the JSONL file: ``session_start`` (written
        # on session entry) and ``session_end`` (written on exit, issue #8).
        # The first start line per session_id wins; a later end line, if
        # present, fills in ``ended_at``.
        sessions: dict[str, dict[str, Any]] = {}
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record: dict[str, Any] = json.loads(line)
                kind = record.get("kind")
                session_id = record.get("session_id")
                if kind == "session_start":
                    if session_id in sessions:
                        continue
                    sessions[session_id] = {
                        "session_id": session_id,
                        "started_at": record["started_at"],
                        "harness_version": record["harness_version"],
                        "model_id": record.get("model_id"),
                        "metadata": record.get("metadata") or {},
                        "ended_at": None,
                    }
                elif kind == "session_end":
                    if session_id in sessions:
                        sessions[session_id]["ended_at"] = record.get("ended_at")
        result = [TraceSession(**data) for data in sessions.values()]
        result.sort(key=lambda s: s.started_at)
        return result

    def _load_session_sqlite(self, session_id: str) -> list[TraceEvent]:
        assert self._sqlite is not None
        conn = self._sqlite.conn
        rows = conn.execute(
            "SELECT event_id, session_id, timestamp, kind, payload "
            "FROM events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
        events: list[TraceEvent] = []
        for event_id, sid, ts, kind, payload in rows:
            events.append(
                TraceEvent.model_validate(
                    {
                        "event_id": event_id,
                        "session_id": sid,
                        "timestamp": ts,
                        "kind": kind,
                        "payload": json.loads(payload),
                    }
                )
            )
        return events

    def _load_session_jsonl(self, session_id: str) -> list[TraceEvent]:
        """Replay events for a session from a JSONL trace file.

        The JSONL backend interleaves ``session_start`` marker lines with
        event lines in a single append-only file. Marker lines carry no
        ``event_id`` and are skipped here; event lines matching
        ``session_id`` are reconstructed into :class:`TraceEvent` objects
        ordered by timestamp.
        """
        events: list[TraceEvent] = []
        if not self.path.exists():
            return events
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record: dict[str, Any] = json.loads(line)
                if record.get("session_id") != session_id:
                    continue
                if "event_id" not in record:
                    continue
                events.append(TraceEvent.model_validate(record))
        events.sort(key=lambda e: e.timestamp)
        return events


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
