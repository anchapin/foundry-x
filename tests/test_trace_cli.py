from __future__ import annotations

import json

from foundry_x.trace.cli import main
from foundry_x.trace.logger import TraceLogger


def _populate(db_path) -> str:
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
        logger.record(sid, "user_prompt", {"prompt": "Fix the bug in auth.py"})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "outcome", {"status": "ok"})
    return sid


def test_sessions_lists_session(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    rc = main(["sessions", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert sid in out
    assert "0.1.0" in out
    assert "test-model" in out


def test_sessions_empty_db(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["sessions", "--db", str(db)])

    assert rc == 0
    assert "No sessions found" in capsys.readouterr().out


def test_show_prints_ordered_events(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    rc = main(["show", sid, "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert sid in out
    assert "user_prompt" in out
    assert "tool_call" in out
    assert "read_file" in out


def test_show_unknown_session_returns_nonzero(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = main(["show", "does-not-exist", "--db", str(db)])

    assert rc == 1


def test_export_produces_jsonl(tmp_path, capsys):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    rc = main(["export", sid, "--db", str(db)])

    assert rc == 0
    lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    assert len(lines) == 3
    for line in lines:
        record = json.loads(line)
        assert record["session_id"] == sid
        assert "event_id" in record
        assert "kind" in record
        assert "payload" in record


def test_export_to_file(tmp_path):
    db = tmp_path / "traces.db"
    out_file = tmp_path / "export.jsonl"
    sid = _populate(db)

    rc = main(["export", sid, "--db", str(db), "--out", str(out_file)])

    assert rc == 0
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    json.loads(lines[0])


def test_list_sessions_method(tmp_path):
    db = tmp_path / "traces.db"
    sid = _populate(db)

    sessions = TraceLogger(db).list_sessions()

    assert len(sessions) == 1
    session = sessions[0]
    assert session.session_id == sid
    assert session.harness_version == "0.1.0"
    assert session.model_id == "test-model"


def test_list_sessions_jsonl_backend(tmp_path):
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="0.2.0", model_id="json-model") as sid:
        logger.record(sid, "user_prompt", {"prompt": "hi"})

    sessions = TraceLogger(db, backend="jsonl").list_sessions()

    assert len(sessions) == 1
    assert sessions[0].session_id == sid
    assert sessions[0].harness_version == "0.2.0"
