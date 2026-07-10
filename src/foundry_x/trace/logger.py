from __future__ import annotations

import json
import re
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


@dataclass
class TraceEvent:
    event_id: str
    session_id: str
    timestamp: str
    kind: str
    payload: dict[str, Any]


@dataclass
class TraceSession:
    session_id: str
    started_at: str
    harness_version: str
    model_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    harness_version TEXT NOT NULL,
    model_id TEXT,
    metadata TEXT
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
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        if backend == "sqlite":
            with sqlite3.connect(self.path) as conn:
                conn.executescript(_SCHEMA)

    @contextmanager
    def session(
        self,
        harness_version: str,
        model_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> Iterator[str]:
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
            pass

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
            with sqlite3.connect(self.path) as conn:
                conn.execute(
                    "INSERT INTO events VALUES (?, ?, ?, ?, ?)",
                    (
                        event.event_id,
                        event.session_id,
                        event.timestamp,
                        event.kind,
                        json.dumps(event.payload),
                    ),
                )
        elif self.backend == "jsonl":
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(
                    json.dumps(
                        {
                            "event_id": event.event_id,
                            "session_id": event.session_id,
                            "timestamp": event.timestamp,
                            "kind": event.kind,
                            "payload": event.payload,
                        }
                    )
                    + "\n"
                )
        return event

    def load_session(self, session_id: str) -> Sequence[TraceEvent]:
        if self.backend == "jsonl":
            return self._load_session_jsonl(session_id)
        return self._load_session_sqlite(session_id)

    def _load_session_sqlite(self, session_id: str) -> list[TraceEvent]:
        events: list[TraceEvent] = []
        with sqlite3.connect(self.path) as conn:
            rows = conn.execute(
                "SELECT event_id, session_id, timestamp, kind, payload "
                "FROM events WHERE session_id = ? ORDER BY timestamp",
                (session_id,),
            ).fetchall()
        for event_id, sid, ts, kind, payload in rows:
            events.append(
                TraceEvent(
                    event_id=event_id,
                    session_id=sid,
                    timestamp=ts,
                    kind=kind,
                    payload=json.loads(payload),
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
                events.append(
                    TraceEvent(
                        event_id=record["event_id"],
                        session_id=record["session_id"],
                        timestamp=record["timestamp"],
                        kind=record["kind"],
                        payload=record["payload"],
                    )
                )
        events.sort(key=lambda e: e.timestamp)
        return events


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
