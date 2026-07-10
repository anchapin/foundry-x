from __future__ import annotations

import json
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
            payload=payload,
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


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()
