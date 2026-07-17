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


class ModelConfig(BaseModel):
    """Model configuration for trace attribution (issue #361).

    Records the model identity and hardware configuration so the improvement
    rate KPI can attribute benchmark outcomes to specific quantizations.
    """

    model_id: str | None = Field(default=None, description="Model identifier or quantization name")
    quantization: str | None = Field(default=None, description="Quantization scheme (e.g. Q5_K_M)")
    context_window: int | None = Field(default=None, description="Context window size in tokens")
    hardware: str | None = Field(default=None, description="Hardware accelerator used")


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


_VALID_BACKENDS = ("sqlite", "jsonl")


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
                            "metadata": redacted_metadata,
                            "kind": "session_start",
                        }
                    )
                    + "\n"
                )
        else:
            assert self._conn is not None  # backend == "sqlite"
            with self._conn:
                self._conn.execute(
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
            assert self._conn is not None  # backend == "sqlite"
            with self._conn:
                self._conn.execute(
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

    def list_sessions(self, harness_version: str | None = None) -> Sequence[TraceSession]:
        """Return every recorded session, optionally filtered by harness version.

        ``harness_version`` mirrors the parameter accepted by the KPI and
        regression-report callers so they can stop reaching past the
        logger with raw ``sqlite3.connect``. Issue #82 — adding the filter
        is the centralization step the issue's acceptance criteria call
        for; the no-argument form preserves backward compatibility with
        the existing ``tests/`` and ``cli.py`` callers.
        """
        if self.backend == "jsonl":
            return self._list_sessions_jsonl(harness_version=harness_version)
        return self._list_sessions_sqlite(harness_version=harness_version)

    def iter_events(
        self,
        session_id: str,
        kind: str | None = None,
    ) -> Iterator[TraceEvent]:
        """Yield :class:`TraceEvent` rows for *session_id* one at a time.

        ``kind`` optionally narrows the stream to a single event kind
        (``"critic_verdict"``, ``"injection_blocked"``, ...). The method
        is a generator: rows are pulled from the underlying store as the
        caller iterates, so long sessions do not need to fit in memory.
        Issue #82 — replaces the raw ``sqlite3.connect + SELECT`` calls
        that previously leaked schema knowledge into ``kpis.py`` and
        ``regression_report.py``.
        """
        if self.backend == "jsonl":
            yield from self._iter_events_jsonl(session_id, kind=kind)
            return
        yield from self._iter_events_sqlite(session_id, kind=kind)

    def query_events(
        self,
        kind: str | None = None,
        harness_version: str | None = None,
    ) -> Iterator[TraceEvent]:
        """Yield :class:`TraceEvent` rows across **all** matching sessions.

        Issue #273 — every cross-session consumer (``compute_kpis``,
        ``_verdict_rates``, ``_injection_blocks`` and
        ``regression_report._load_verdict_events``) previously called
        ``list_sessions()`` and then looped ``iter_events(sid)`` once per
        session per kind. Each ``iter_events`` call opened a fresh
        ``sqlite3.connect`` (11 connect sites across the codebase, prior
        to issue #274 collapsing them onto one reused connection), so
        for Phase-3 scale (many sessions/day) the Digester→Critic
        feedback path paid S*K round-trips for what is logically a single
        ordered scan.

        ``query_events`` collapses that to one streaming cursor. Rows are
        yielded in timestamp order — the same order ``iter_events``
        promises within a session — so callers that need
        first-event-per-session semantics can use ``setdefault`` on a
        ``session_id -> event`` map as they stream.

        Parameters
        ----------
        kind:
            When provided, only events whose ``kind`` column equals this
            value are yielded. Pushed down to the underlying store as a
            ``WHERE kind = ?`` clause (sqlite) or an inline filter
            (jsonl) so the kind-bounded cursor never materializes the
            other rows.
        harness_version:
            When provided, only events belonging to sessions whose
            ``harness_version`` matches are yielded. Implemented as a
            JOIN against the ``sessions`` table (sqlite) or by tracking
            ``session_start`` marker lines inline (jsonl). ``None`` means
            no filter — events from every session qualify.
        """
        if self.backend == "jsonl":
            yield from self._query_events_jsonl(kind=kind, harness_version=harness_version)
            return
        yield from self._query_events_sqlite(kind=kind, harness_version=harness_version)

    def _list_sessions_sqlite(self, harness_version: str | None = None) -> Sequence[TraceSession]:
        assert self._conn is not None  # backend == "sqlite"
        columns = {row[1] for row in self._conn.execute("PRAGMA table_info(sessions)").fetchall()}
        selected = [
            "session_id",
            "started_at",
            "harness_version",
            "model_id",
            "metadata",
        ]
        if "ended_at" in columns:
            selected.append("ended_at")
        query = "SELECT " + ", ".join(selected) + " FROM sessions"
        params: tuple[Any, ...] = ()
        if harness_version is not None:
            query += " WHERE harness_version = ?"
            params = (harness_version,)
        query += " ORDER BY started_at"
        rows = self._conn.execute(query, params).fetchall()
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

    def _list_sessions_jsonl(self, harness_version: str | None = None) -> Sequence[TraceSession]:
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
        filtered = [
            data
            for data in sessions.values()
            if harness_version is None or data["harness_version"] == harness_version
        ]
        result = [TraceSession(**data) for data in filtered]
        result.sort(key=lambda s: s.started_at)
        return result

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

    def _iter_events_sqlite(
        self,
        session_id: str,
        kind: str | None = None,
    ) -> Iterator[TraceEvent]:
        # Stream rows from the driver one at a time (issue #82): the
        # ``cursor`` returned by ``conn.execute`` yields rows on demand
        # without buffering the full result set in Python memory, which is
        # what callers like a future Digester need for long sessions.
        query = (
            "SELECT event_id, session_id, timestamp, kind, payload FROM events WHERE session_id = ?"
        )
        params: list[Any] = [session_id]
        if kind is not None:
            query += " AND kind = ?"
            params.append(kind)
        query += " ORDER BY timestamp"
        assert self._conn is not None  # backend == "sqlite"
        cursor = self._conn.execute(query, params)
        for event_id, sid, ts, k, payload in cursor:
            yield TraceEvent.model_validate(
                {
                    "event_id": event_id,
                    "session_id": sid,
                    "timestamp": ts,
                    "kind": k,
                    "payload": json.loads(payload),
                }
            )

    def _query_events_sqlite(
        self,
        kind: str | None = None,
        harness_version: str | None = None,
    ) -> Iterator[TraceEvent]:
        # Issue #273 — one streaming cursor across all matching sessions.
        # The optional ``harness_version`` filter is implemented as a JOIN
        # against the sessions table rather than a Python-side filter so
        # the database prunes non-matching sessions before rows cross the
        # process boundary. ``ORDER BY timestamp`` preserves the promise
        # ``iter_events`` makes within a single session, extended to the
        # cross-session stream.
        query = "SELECT e.event_id, e.session_id, e.timestamp, e.kind, e.payload FROM events e"
        params: list[Any] = []
        conditions: list[str] = []
        if harness_version is not None:
            query += " JOIN sessions s ON e.session_id = s.session_id"
            conditions.append("s.harness_version = ?")
            params.append(harness_version)
        if kind is not None:
            conditions.append("e.kind = ?")
            params.append(kind)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY e.timestamp"
        assert self._conn is not None  # backend == "sqlite"
        cursor = self._conn.execute(query, params)
        for event_id, sid, ts, k, payload in cursor:
            yield TraceEvent.model_validate(
                {
                    "event_id": event_id,
                    "session_id": sid,
                    "timestamp": ts,
                    "kind": k,
                    "payload": json.loads(payload),
                }
            )

    def _query_events_sqlite(
        self,
        kind: str | None = None,
        harness_version: str | None = None,
    ) -> Iterator[TraceEvent]:
        # Issue #273 — one streaming cursor across all matching sessions.
        # The optional ``harness_version`` filter is implemented as a JOIN
        # against the sessions table rather than a Python-side filter so
        # the database prunes non-matching sessions before rows cross the
        # process boundary. ``ORDER BY timestamp`` preserves the promise
        # ``iter_events`` makes within a single session, extended to the
        # cross-session stream.
        query = "SELECT e.event_id, e.session_id, e.timestamp, e.kind, e.payload FROM events e"
        params: list[Any] = []
        conditions: list[str] = []
        if harness_version is not None:
            query += " JOIN sessions s ON e.session_id = s.session_id"
            conditions.append("s.harness_version = ?")
            params.append(harness_version)
        if kind is not None:
            conditions.append("e.kind = ?")
            params.append(kind)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY e.timestamp"
        assert self._conn is not None  # backend == "sqlite"
        cursor = self._conn.execute(query, params)
        for event_id, sid, ts, k, payload in cursor:
            yield TraceEvent.model_validate(
                {
                    "event_id": event_id,
                    "session_id": sid,
                    "timestamp": ts,
                    "kind": k,
                    "payload": json.loads(payload),
                }
            )

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

    def delete_session(self, session_id: str) -> bool:
        """Remove every event and the session row for ``session_id``.

        Idempotent: returns ``True`` whether or not the session existed,
        so an operator running this on a stale ``session_id`` after a
        previous delete does not see an error. The other session in the
        store is untouched. Works on both sqlite and jsonl backends.
        """
        if self.backend == "jsonl":
            self._delete_session_jsonl(session_id)
        else:
            self._delete_session_sqlite(session_id)
        return True

    def prune_sessions(self, to_delete: Sequence[str]) -> int:
        """Remove every event and the session row for each ``session_id`` in *to_delete*.

        This is a bulk operation: all sessions in *to_delete* are deleted in a single
        file rewrite (jsonl) or batch DELETE (sqlite), making it O(1) file I/O
        instead of O(n) for n sessions. Returns the number of sessions deleted.
        Issue #752.
        """
        if not to_delete:
            return 0
        if self.backend == "jsonl":
            return self._prune_jsonl(to_delete)
        return self._prune_sqlite(to_delete)

    def _prune_sqlite(self, to_delete: Sequence[str]) -> int:
        assert self._conn is not None  # backend == "sqlite"
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM events WHERE session_id IN ("
                + ",".join("?" * len(to_delete))
                + ")",
                list(to_delete),
            )
            cur = self._conn.execute(
                "DELETE FROM sessions WHERE session_id IN ("
                + ",".join("?" * len(to_delete))
                + ")",
                list(to_delete),
            )
        return cur.rowcount

    def _prune_jsonl(self, to_delete: Sequence[str]) -> int:
        if not self.path.exists():
            return 0
        delete_set = set(to_delete)
        kept: list[str] = []
        removed_sessions: set[str] = set()
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    kept.append(line)
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if record.get("session_id") in delete_set:
                    removed_sessions.add(record.get("session_id"))
                    continue
                kept.append(line)
        with self.path.open("w", encoding="utf-8") as fh:
            fh.writelines(kept)
        return len(removed_sessions)

    def _delete_session_sqlite(self, session_id: str) -> None:
        assert self._conn is not None  # backend == "sqlite"
        with self._conn:
            self._conn.execute("DELETE FROM events WHERE session_id = ?", (session_id,))
            self._conn.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))

    def _delete_session_jsonl(self, session_id: str) -> None:
        if not self.path.exists():
            return
        kept: list[str] = []
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    kept.append(line)
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue
                if record.get("session_id") == session_id:
                    continue
                kept.append(line)
        with self.path.open("w", encoding="utf-8") as fh:
            fh.writelines(kept)

    def compact(self) -> int:
        """Rewrite the JSONL file removing orphaned session markers.

        An orphaned ``session_end`` marker is one where the ``session_id`` has
        no corresponding ``session_start`` marker in the file. This can happen
        if a session was deleted and then a stale ``session_end`` marker was
        written afterward.

        Returns the number of orphaned ``session_end`` markers removed.
        Only works on the JSONL backend; SQLite VACUUM is handled automatically
        by the database engine (per issue #632 out-of-scope).
        """
        if self.backend != "jsonl":
            return 0
        if not self.path.exists():
            return 0

        kept: list[str] = []
        seen_starts: set[str] = set()
        orphaned: list[str] = []

        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if not stripped:
                    kept.append(line)
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    kept.append(line)
                    continue

                kind = record.get("kind")
                session_id = record.get("session_id")

                if kind == "session_start":
                    if session_id not in seen_starts:
                        seen_starts.add(session_id)
                    kept.append(line)
                elif kind == "session_end":
                    if session_id not in seen_starts:
                        orphaned.append(line)
                    else:
                        kept.append(line)
                else:
                    kept.append(line)

        orphaned_count = len(orphaned)
        with self.path.open("w", encoding="utf-8") as fh:
            fh.writelines(kept)

        return orphaned_count

    def redact_event(
        self,
        session_id: str,
        event_index: int,
        key: str,
    ) -> bool:
        """Replace ``payload[key]`` with ``"[REDACTED]"`` on the indexed event.

        ``event_index`` is the position of the target event in the
        timestamp-ordered stream for ``session_id`` (the same order
        :meth:`load_session` and :meth:`iter_events` return). Returns
        ``True`` when the event was found and rewritten; ``False`` when
        the index is out of range so a stale index surfaces immediately
        rather than silently rewriting the wrong row.
        """
        if self.backend == "jsonl":
            return self._redact_event_jsonl(session_id, event_index, key)
        return self._redact_event_sqlite(session_id, event_index, key)

    def _redact_event_sqlite(
        self,
        session_id: str,
        event_index: int,
        key: str,
    ) -> bool:
        assert self._conn is not None  # backend == "sqlite"
        with self._conn:
            rows = self._conn.execute(
                "SELECT event_id, payload FROM events WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
            if event_index < 0 or event_index >= len(rows):
                return False
            event_id, payload_text = rows[event_index]
            payload = json.loads(payload_text)
            payload[key] = "[REDACTED]"
            self._conn.execute(
                "UPDATE events SET payload = ? WHERE event_id = ?",
                (json.dumps(payload), event_id),
            )
        return True

    def _redact_event_jsonl(
        self,
        session_id: str,
        event_index: int,
        key: str,
    ) -> bool:
        if not self.path.exists():
            return False
        with self.path.open("r", encoding="utf-8") as fh:
            lines = fh.readlines()
        session_events: list[tuple[int, dict[str, Any]]] = []
        for idx, line in enumerate(lines):
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if record.get("session_id") == session_id and "event_id" in record:
                session_events.append((idx, record))
        if not session_events:
            return False
        session_events.sort(key=lambda pair: pair[1].get("timestamp", ""))
        if event_index < 0 or event_index >= len(session_events):
            return False
        target_idx, target_record = session_events[event_index]
        target_record["payload"][key] = "[REDACTED]"
        lines[target_idx] = json.dumps(target_record) + "\n"
        with self.path.open("w", encoding="utf-8") as fh:
            fh.writelines(lines)
        return True

    def _iter_events_jsonl(
        self,
        session_id: str,
        kind: str | None = None,
    ) -> Iterator[TraceEvent]:
        # Stream lines through the file object so we never hold the full
        # JSONL file in memory (issue #82). We do materialize one event at
        # a time per yield, which is the smallest unit the producer can
        # hand us — there is no row buffer to overflow on long sessions.
        if not self.path.exists():
            return
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
                if kind is not None and record.get("kind") != kind:
                    continue
                yield TraceEvent.model_validate(record)

    def _query_events_jsonl(
        self,
        kind: str | None = None,
        harness_version: str | None = None,
    ) -> Iterator[TraceEvent]:
        # Issue #273 — stream the JSONL file exactly once, yielding every
        # matching event in append order (which is timestamp order for a
        # well-formed append-only trace). The optional ``harness_version``
        # filter is resolved inline by tracking each ``session_start``
        # marker as we walk: the file format guarantees a session's start
        # line precedes its event lines, so by the time we reach an event
        # its session's harness version is already known. Sessions whose
        # start marker we have not yet seen (a corrupted / mid-write file)
        # are excluded when a filter is set, matching the sqlite JOIN
        # semantics.
        if not self.path.exists():
            return
        session_versions: dict[str, str] = {}
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record: dict[str, Any] = json.loads(line)
                record_kind = record.get("kind")
                if record_kind == "session_start":
                    sid = record.get("session_id")
                    if sid is not None and sid not in session_versions:
                        session_versions[sid] = record.get("harness_version", "")
                    continue
                # Only real events (lines carrying an ``event_id``) qualify.
                # ``session_end`` markers and any future non-event lines are
                # skipped here so they never reach the caller.
                if "event_id" not in record:
                    continue
                if kind is not None and record_kind != kind:
                    continue
                if harness_version is not None:
                    sid = record.get("session_id")
                    if session_versions.get(sid) != harness_version:
                        continue
                yield TraceEvent.model_validate(record)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
