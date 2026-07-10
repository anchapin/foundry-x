from __future__ import annotations

import json
from pathlib import Path

from foundry_x.trace.logger import TraceLogger

_REPO_ROOT = Path(__file__).resolve().parents[1]
_MANIFEST_PATH = _REPO_ROOT / "harness" / "manifest.json"


def test_trace_logger_roundtrip(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})
        logger.record(sid, kind="tool_call", payload={"name": "read_file"})
        events = logger.load_session(sid)
    assert len(events) == 2
    assert events[0].kind == "user_prompt"
    assert events[1].kind == "tool_call"


def test_trace_logger_jsonl_backend(tmp_path):
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "x"})
    assert path.exists()
    assert path.read_text().strip().endswith("}")


def test_trace_logger_jsonl_session_record_load_roundtrip(tmp_path):
    """session() + record() + load_session() must round-trip on jsonl.

    Acceptance criterion from issue #2: the jsonl backend's session +
    record + load round-trip must work, mirroring the sqlite path.
    """
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="test-0.0", model_id="m1") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})
        logger.record(sid, kind="tool_call", payload={"name": "read_file"})
        events = logger.load_session(sid)
    assert len(events) == 2
    assert events[0].kind == "user_prompt"
    assert events[1].kind == "tool_call"
    assert events[0].session_id == sid
    assert events[0].payload == {"text": "hi"}


def test_trace_logger_jsonl_does_not_touch_sqlite(tmp_path):
    """session() under backend='jsonl' must never create an SQLite file.

    Acceptance criterion from issue #2: the jsonl backend must not touch
    any SQLite database. Only the jsonl file should appear on disk.
    """
    jsonl_path = tmp_path / "traces.jsonl"
    logger = TraceLogger(jsonl_path, backend="jsonl")
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "x"})
    assert jsonl_path.exists()
    sqlite_artifacts = [p for p in tmp_path.iterdir() if p.suffix in {".db", ".sqlite", ".sqlite3"}]
    assert sqlite_artifacts == []
