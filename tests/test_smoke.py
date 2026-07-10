from __future__ import annotations

from foundry_x.trace.logger import TraceLogger


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
