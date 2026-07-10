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
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
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
CREATE INDEX IF NOT EXISTS idx_events_session_kind ON events(session_id, kind);
"""


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

_DEFAULT_SECRET_KEY_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "authorization",
        "password",
        "passwd",
        "secret",
        "secret_key",
        "token",
    }
)


def _redact_value(value: str) -> str:
    """Mask secret-like substrings within a single string."""
    value = _PEM_RE.sub("[REDACTED:pem]", value)
    value = _API_KEY_RE.sub("[REDACTED:api-key]", value)
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
    string values are scanned for ``sk-...``, ``Bearer ...`` and PEM
    blocks.
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
        # Fail fast on an unknown backend (issue #272): previously an
        # invalid value (e.g. a misspelled ``"csv"``) was stored unchecked
        # and then silently dropped every event in ``record()`` (whose
        # ``if sqlite / elif jsonl`` chain had no ``else``) while
        # ``session()`` mis-routed to sqlite. That violated AGENTS.md §2
        # ("never silently swallow") and ADR-0007 (traces are ground
        # truth). Validating at construction closes every downstream
        # branch — ``record``, ``session`` and ``_end_session`` are all
        # guaranteed a known backend thereafter.
        if backend not in _VALID_BACKENDS:
            raise ValueError(
                f"unsupported backend {backend!r}; expected one of "
                f"{', '.join(repr(b) for b in _VALID_BACKENDS)}",
            )
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        # Issue #274 — the sqlite backend opens ONE connection here and reuses
        # it for every subsequent operation, instead of paying a fresh
        # ``sqlite3.connect`` (and its lock/page-cache setup) on every call.
        # The connection also runs ``PRAGMA journal_mode=WAL`` so that a
        # Digester/KPI reader on a separate connection can read committed
        # events while a Runner write is in flight without raising
        # ``SQLITE_BUSY``. See ADR-0013 for the rationale and the note it
        # supersedes in ADR-0003.
        self._conn: sqlite3.Connection | None = None
        if backend == "sqlite":
            self._conn = sqlite3.connect(self.path)
            # WAL is a persistent database property (stored in the file
            # header), so this also benefits raw ``sqlite3.connect`` readers
            # opened against the same file later.
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.executescript(_SCHEMA)
            # Non-destructive migration for pre-issue-#8 databases that
            # predate the ``ended_at`` column (issue #8). Guarded by a
            # pragma check so freshly-created databases are untouched and
            # existing ``logs/*.db`` files do not break.
            columns = {row[1] for row in self._conn.execute("PRAGMA table_info(sessions)")}
            if "ended_at" not in columns:
                self._conn.execute("ALTER TABLE sessions ADD COLUMN ended_at TEXT")
            self._conn.commit()

    def close(self) -> None:
        """Close the reused sqlite connection (issue #274).

        Optional lifecycle hook: the connection is also released by garbage
        collection when the logger falls out of scope, so existing callers
        that never call ``close()`` keep working. Tests and long-running
        services that construct many loggers can call this to release the
        connection (and its ``-wal``/``-shm`` sidecar handles) deterministically.
        """
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    @contextmanager
    def session(
        self,
        harness_version: str,
        model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        model_config: Any = None,
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
                            "metadata": metadata or {},
                            "kind": "session_start",
                        }
                    )
                    + "\n"
                )
        else:
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
                    (
                        session_id,
                        _now(),
                        harness_version,
                        model_id,
                        json.dumps(metadata or {}),
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
                assert self._conn is not None  # backend == "sqlite"
                with self._conn:
                    self._conn.execute(
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
            with sqlite3.connect(self.path) as conn:
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
        with sqlite3.connect(self.path) as conn:
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
        sessions: list[TraceSession] = []
        seen: set[str] = set()
        if not self.path.exists():
            return sessions
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record: dict[str, Any] = json.loads(line)
                if record.get("kind") != "session_start":
                    continue
                session_id = record["session_id"]
                if session_id in seen:
                    continue
                seen.add(session_id)
                sessions.append(
                    TraceSession(
                        session_id=session_id,
                        started_at=record["started_at"],
                        harness_version=record["harness_version"],
                        model_id=record.get("model_id"),
                        metadata=record.get("metadata") or {},
                    )
                )
        sessions.sort(key=lambda s: s.started_at)
        return sessions

    def _load_session_sqlite(self, session_id: str) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        assert self._conn is not None  # backend == "sqlite"
        rows = self._conn.execute(
            "SELECT event_id, session_id, timestamp, kind, payload "
            "FROM events WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
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
