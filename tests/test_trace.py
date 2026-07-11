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
