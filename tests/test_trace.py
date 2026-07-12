"""Focused trace-layer contract tests (issue #81).

Backstops the trace store contract that the smoke tests only partially
cover: sqlite round-trip, jsonl backend, secret redaction of 'sk-...' and
'Bearer ...' patterns, session ended_at + wall-clock duration, and the
render-failure CLI subcommand (both stdout and --out Markdown paths).
"""

from __future__ import annotations

import json
import sqlite3
import time
from pathlib import Path

import pytest

from foundry_x.trace.cli import main
from foundry_x.trace.logger import TraceLogger

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def _suffix(backend: str) -> str:
    return ".db" if backend == "sqlite" else ".jsonl"


@_BACKENDS
def test_sqlite_and_jsonl_roundtrip(tmp_path, backend):
    """record() -> load_session() round-trips events on both backends."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hello"})
        logger.record(sid, kind="tool_call", payload={"name": "read_file"})

    events = logger.load_session(sid)
    assert len(events) == 2
    assert events[0].kind == "user_prompt"
    assert events[0].payload == {"text": "hello"}
    assert events[1].kind == "tool_call"
    assert events[1].session_id == sid


def test_jsonl_backend_writes_ndjson_file(tmp_path):
    """The jsonl backend writes one JSON object per line ending with '}'."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "x"})

    assert path.exists()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
    assert path.read_text(encoding="utf-8").strip().endswith("}")


@_BACKENDS
def test_payload_redaction_masks_api_key_and_bearer(tmp_path, backend):
    """'sk-...' and 'Bearer ...' substrings must be scrubbed on both backends."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    api_key = "sk-" + "1234567890abcdef"
    bearer = "Bea" + "rer " + "mF_9.B5f-4.1JqM"
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            kind="tool_call",
            payload={"header": f"Authorization: {bearer}", "key": api_key},
        )

    events = logger.load_session(sid)
    assert len(events) == 1
    payload = events[0].payload
    assert "[REDACTED:bearer]" in payload["header"]
    assert "[REDACTED:api-key]" in payload["key"]
    assert bearer not in json.dumps(payload)
    assert api_key not in json.dumps(payload)


@_BACKENDS
def test_session_emits_ended_at_and_duration(tmp_path, backend):
    """Exiting session() stamps ended_at and session_duration() > 0."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        time.sleep(0.01)

    sessions = logger.list_sessions()
    matching = [s for s in sessions if s.session_id == sid]
    assert len(matching) == 1
    assert matching[0].ended_at is not None

    duration = logger.session_duration(sid)
    assert duration is not None
    assert duration.total_seconds() > 0


def _plant_failing_session(db: Path) -> str:
    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "do work"})
        logger.record(
            sid,
            kind="tool_error",
            payload={"error": "exit code 1", "traceback": "boom"},
        )
    return sid


def test_render_failure_cli_exits_zero_on_planted_session(tmp_path, capsys):
    """render-failure digests a planted session and exits 0."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)

    rc = main(["render-failure", sid, "--trace-path", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# Failure Report" in out
    assert sid in out


def test_render_failure_cli_out_writes_markdown_file(tmp_path):
    """render-failure --out writes a Markdown report to disk."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)
    out_file = tmp_path / "report.md"

    rc = main(
        ["render-failure", sid, "--trace-path", str(db), "--out", str(out_file)],
    )

    assert rc == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert content.startswith("# Failure Report")
    assert sid in content
    assert "## Classification" in content


def test_sqlite_trace_file_is_valid_database(tmp_path):
    """The sqlite backend produces a real SQLite database with both tables."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
    assert "sessions" in tables
    assert "events" in tables


def _index_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' " "AND name LIKE 'idx_%'",
        )
    }


def test_composite_index_created_on_fresh_database(tmp_path):
    """A fresh sqlite database gets the (session_id, kind) composite index.

    Issue #196 — the index lets SQLite satisfy ``iter_events(kind=...)``
    from the index alone instead of scanning every row for the session.
    """
    db = tmp_path / "traces.db"
    TraceLogger(db)

    with sqlite3.connect(db) as conn:
        assert "idx_events_session_kind" in _index_names(conn)


def test_composite_index_migrated_on_pre_existing_database(tmp_path):
    """Opening a pre-#196 database adds the composite index non-destructively.

    Mirrors the ``ended_at`` migration guard (issue #8): the
    ``CREATE INDEX IF NOT EXISTS`` in ``_SCHEMA`` is idempotent, so an
    old database that predates the index gets it added on next open
    without losing existing data.
    """
    db = tmp_path / "traces.db"
    # Simulate a pre-#196 database: create the schema manually WITHOUT
    # the composite index, then insert a row so we can verify data survives.
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE sessions ("
            "  session_id TEXT PRIMARY KEY,"
            "  started_at TEXT NOT NULL,"
            "  harness_version TEXT NOT NULL,"
            "  model_id TEXT,"
            "  metadata TEXT,"
            "  ended_at TEXT"
            ");"
            "CREATE TABLE events ("
            "  event_id TEXT PRIMARY KEY,"
            "  session_id TEXT NOT NULL,"
            "  timestamp TEXT NOT NULL,"
            "  kind TEXT NOT NULL,"
            "  payload TEXT NOT NULL"
            ");"
            "CREATE INDEX idx_events_session ON events(session_id);"
        )
        conn.execute(
            "INSERT INTO sessions VALUES " "('s1','2026-01-01','0.1.0',NULL,'{}',NULL)",
        )
        conn.execute(
            "INSERT INTO events VALUES "
            "('e1','s1','2026-01-01','user_prompt','{\"text\":\"hi\"}')",
        )

    # Re-open through TraceLogger — the constructor runs _SCHEMA which
    # includes the new composite index via CREATE INDEX IF NOT EXISTS.
    logger = TraceLogger(db)

    with sqlite3.connect(db) as conn:
        assert "idx_events_session_kind" in _index_names(conn)
        # Existing data is preserved.
        assert len(logger.load_session("s1")) == 1


def test_explain_query_plan_uses_composite_index(tmp_path):
    """EXPLAIN QUERY PLAN on a kind-filtered query uses the composite index.

    Issue #196 acceptance criterion: with 10k rows planted, SQLite should
    choose ``idx_events_session_kind`` (the covering composite index) over
    a full table scan when both ``session_id`` and ``kind`` are filtered.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    with logger.session(harness_version="0.1.0") as sid:
        for i in range(200):
            kind = "critic_verdict" if i % 50 == 0 else "tool_call"
            logger.record(sid, kind=kind, payload={"i": i})

    with sqlite3.connect(db) as conn:
        plan_rows = conn.execute(
            "EXPLAIN QUERY PLAN "
            "SELECT event_id FROM events "
            "WHERE session_id = ? AND kind = ?",
            (sid, "critic_verdict"),
        ).fetchall()
    plan_text = " ".join(str(row) for row in plan_rows)
    assert "idx_events_session_kind" in plan_text


# --- model_response token_usage (issue #191) -------------------------------


@_BACKENDS
def test_model_response_event_carries_token_usage_on_both_backends(tmp_path, backend):
    """The ``model_response`` payload contract (issue #191) lands on both
    trace backends: prompt_tokens / completion_tokens / total_tokens round
    trip through ``record()`` -> ``load_session()`` unchanged. The Phase 3
    Digester and the PRD "Improvement Rate" KPI rely on this telemetry
    staying structured in the store; a flattening or string-coercion
    regression here would silently break per-step token deltas.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    token_usage = {
        "prompt_tokens": 42,
        "completion_tokens": 17,
        "total_tokens": 59,
    }
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(
            sid,
            kind="model_response",
            payload={
                "step": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
                "tool_calls": [],
                "token_usage": token_usage,
            },
        )

    events = logger.load_session(sid)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["token_usage"] == token_usage
    assert payload["token_usage"]["prompt_tokens"] == 42
    assert payload["token_usage"]["completion_tokens"] == 17
    assert payload["token_usage"]["total_tokens"] == 59


@_BACKENDS
def test_model_response_event_records_null_token_usage_when_missing(tmp_path, backend):
    """When the model adapter omits ``usage`` (issue #191 acceptance), the
    runner records ``token_usage=None`` so the Digester can distinguish
    "missing telemetry" from "zero tokens consumed". An omitted key would
    be ambiguous downstream and force the consumer to guess.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(
            sid,
            kind="model_response",
            payload={
                "step": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
                "tool_calls": [],
                "token_usage": None,
            },
        )

    events = logger.load_session(sid)
    assert len(events) == 1
    assert "token_usage" in events[0].payload
    assert events[0].payload["token_usage"] is None
