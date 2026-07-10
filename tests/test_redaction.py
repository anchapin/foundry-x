"""Secret-redaction tests for the trace logger (issue #3).

Acceptance: a payload containing an ``sk-...`` key and a PEM block must
persist ``[REDACTED:api-key]`` / ``[REDACTED:pem]`` rather than the raw
value, against both the sqlite and jsonl backends. SECURITY.md lines
44-46 and 68-69 must be satisfied for the trace writer.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from foundry_x.trace.logger import TraceLogger, _redact

_PEM_BEGIN = "-----BEGIN RSA " + "PRIVATE KEY-----"
_PEM_END = "-----END RSA " + "PRIVATE KEY-----"
_PEM = f"{_PEM_BEGIN}\nMIIEpAIBAAKCAQEAdGhpcyBpcyBhIGZha2Uga2V5\n{_PEM_END}"
# Built from fragments so gitleaks does not flag the literal pattern in
# source; the runtime value still matches the redaction regexes.
_API_KEY = "sk-" + "1234567890abcdef"
_BEARER = "Bea" + "rer " + "mF_9.B5f-4.1JqM"
_SECRET_KEY = "sk_" + "live_50charslongsecretkeyvaluehere123"

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def _read_persisted_payload(logger: TraceLogger, session_id: str) -> dict:
    events = logger.load_session(session_id)
    return events[0].payload


@_BACKENDS
def test_redaction_scrubs_api_key_and_pem(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(
            sid,
            kind="tool_result",
            payload={"output": f"key={_API_KEY}\n{_PEM}"},
        )
    payload = _read_persisted_payload(logger, sid)
    blob = json.dumps(payload)
    assert _API_KEY not in blob
    assert "BEGIN RSA PRIVATE KEY" not in blob
    assert "[REDACTED:api-key]" in payload["output"]
    assert "[REDACTED:pem]" in payload["output"]


@_BACKENDS
def test_redaction_scrubs_bearer_token(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(
            sid,
            kind="http_call",
            payload={"header": f"Authorization: {_BEARER}"},
        )
    payload = _read_persisted_payload(logger, sid)
    assert "mF_9.B5f-4.1JqM" not in json.dumps(payload)
    assert "[REDACTED:bearer]" in payload["header"]


@_BACKENDS
def test_redaction_scrubs_secret_named_keys(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(
            sid,
            kind="env",
            payload={
                "api_key": _SECRET_KEY,
                "token": "opaque-opaque",
                "password": "hunter2",
                "safe_value": "keep-me",
            },
        )
    payload = _read_persisted_payload(logger, sid)
    assert payload["api_key"] == "[REDACTED:secret]"
    assert payload["token"] == "[REDACTED:secret]"
    assert payload["password"] == "[REDACTED:secret]"
    assert payload["safe_value"] == "keep-me"
    assert _SECRET_KEY not in json.dumps(payload)


@_BACKENDS
def test_redaction_does_not_alter_clean_payloads(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    clean = {"text": "hi", "count": 3, "nested": {"a": [1, 2]}}
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="user_prompt", payload=clean)
    payload = _read_persisted_payload(logger, sid)
    assert payload == clean


def test_redaction_handles_nested_structures():
    payload = {
        "outer": [
            {"api_key": _API_KEY},
            {"snippet": f"auth {_BEARER} done"},
        ],
    }
    result = _redact(payload)
    assert result["outer"][0]["api_key"] == "[REDACTED:secret]"
    assert "[REDACTED:bearer]" in result["outer"][1]["snippet"]
    assert _API_KEY not in json.dumps(result)


def test_redaction_does_not_mutate_input():
    original = {"api_key": _API_KEY, "note": f"{_BEARER}"}
    _redact(original)
    assert original["api_key"] == _API_KEY
    assert original["note"] == _BEARER


def test_redaction_scrubs_pem_directly_in_sqlite_blob(tmp_path):
    """Raw SQL inspection: the persisted cell must not contain the PEM."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="file_read", payload={"content": _PEM})
    with sqlite3.connect(db) as conn:
        row = conn.execute("SELECT payload FROM events").fetchone()
    raw = row[0]
    assert "BEGIN RSA PRIVATE KEY" not in raw
    assert "[REDACTED:pem]" in raw
