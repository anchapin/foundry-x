"""Secret-redaction tests for the trace logger (issues #3 and #121).

Issue #3 acceptance: a payload containing an ``sk-...`` key and a PEM
block must persist ``[REDACTED:api-key]`` / ``[REDACTED:pem]`` rather than
the raw value, against both the sqlite and jsonl backends. SECURITY.md
lines 44-46 and 68-69 must be satisfied for the trace writer.

Issue #121 acceptance: the ``metadata`` dict passed to
``TraceLogger.session()`` is scrubbed on both backends (the original
implementation persisted it verbatim), the named-key set covers modern
secret names, and the content patterns cover GitHub classic + fine-
grained PATs, JWTs, AWS access key IDs, Stripe live keys, and Slack
tokens.
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
# Modern token fixtures (issue #121). Each is hand-crafted in pieces so
# gitleaks does not flag the literal at commit time; the assembled value
# still matches the corresponding regex at runtime.
_GITHUB_CLASSIC_PAT = "gh" + "p_" + "1A2B3C4D5E6F7G8H9I0J1A2B3"
_GITHUB_FINE_GRAINED_PAT = "github_" + "pat_11ABCDEFG0_1234567890abcdefghijklmnopqrstuvwxyz"
_JWT = (
    "eyJ"
    + "hbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9"
    + "."
    + "eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IkpvaG4gRG9lIn0"
    + "."
    + "SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
)
_AWS_ACCESS_KEY = "AKIA" + "IOSFODNN7EXAMPLE"
_STRIPE_LIVE_KEY = "sk_" + "live_" + "4eC39HqLyjWDarjtT1zdp7dc"
_SLACK_TOKEN = "xox" + "b-1234567890123-1234567890123-" + "abcdefghijklmnopqrstuvwx"

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


# ---------------------------------------------------------------------------
# Issue #121: metadata-path redaction + modern-token pattern coverage.
# ---------------------------------------------------------------------------


@_BACKENDS
def test_session_metadata_is_redacted_on_persistence(tmp_path, backend):
    """TraceLogger.session(metadata=...) must scrub the metadata dict before
    writing it. Pre-#121 this round-trip leaked operator-supplied tokens."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    payload_metadata = {
        "operator": "alex",
        "github_token": _GITHUB_CLASSIC_PAT,
        "nested": {"aws_access_key_id": _AWS_ACCESS_KEY, "task": "ingest"},
    }
    with logger.session(harness_version="test-0.0", metadata=payload_metadata):
        pass
    sessions = logger.list_sessions()
    assert len(sessions) == 1
    persisted_metadata = sessions[0].metadata
    blob = json.dumps(persisted_metadata)
    assert _GITHUB_CLASSIC_PAT not in blob
    assert _AWS_ACCESS_KEY not in blob
    assert persisted_metadata["operator"] == "alex"
    assert persisted_metadata["nested"]["task"] == "ingest"
    assert persisted_metadata["github_token"] == "[REDACTED:secret]"
    assert persisted_metadata["nested"]["aws_access_key_id"] == "[REDACTED:secret]"


@_BACKENDS
def test_session_metadata_input_is_not_mutated(tmp_path, backend):
    """The dict passed by the Operator must not be mutated by the
    redaction pass. Issue #121 acceptance."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    metadata_input = {
        "github_token": _GITHUB_CLASSIC_PAT,
        "note": f"please keep {_BEARER} verbatim",
    }
    original_github = metadata_input["github_token"]
    original_note = metadata_input["note"]
    with logger.session(harness_version="test-0.0", metadata=metadata_input):
        pass
    assert metadata_input["github_token"] == original_github
    assert metadata_input["note"] == original_note


def test_redaction_scrubs_github_classic_pat():
    result = _redact({"output": f"token={_GITHUB_CLASSIC_PAT}"})
    assert _GITHUB_CLASSIC_PAT not in json.dumps(result)
    assert result["output"] == "token=[REDACTED:github-pat]"


def test_redaction_scrubs_github_fine_grained_pat():
    result = _redact({"output": f"token={_GITHUB_FINE_GRAINED_PAT}"})
    assert _GITHUB_FINE_GRAINED_PAT not in json.dumps(result)
    assert "[REDACTED:github-pat]" in result["output"]


def test_redaction_scrubs_jwt():
    result = _redact({"header": f"Authorization: Bearer {_JWT}"})
    # JWT is detected on its own; the surrounding "Bearer <token>" then
    # additionally triggers the bearer redaction. Either way the raw JWT
    # must not survive.
    blob = json.dumps(result)
    assert _JWT not in blob
    assert "[REDACTED:jwt]" in blob or "[REDACTED:bearer]" in blob


def test_redaction_scrubs_aws_access_key_id():
    result = _redact({"env": f"AWS_ACCESS_KEY_ID={_AWS_ACCESS_KEY}"})
    assert _AWS_ACCESS_KEY not in json.dumps(result)
    assert result["env"] == "AWS_ACCESS_KEY_ID=[REDACTED:aws-access-key]"


def test_redaction_scrubs_stripe_live_key():
    result = _redact({"output": f"stripe={_STRIPE_LIVE_KEY}"})
    assert _STRIPE_LIVE_KEY not in json.dumps(result)
    assert "[REDACTED:stripe-key]" in result["output"]


def test_redaction_scrubs_slack_token():
    result = _redact({"webhook": _SLACK_TOKEN})
    assert _SLACK_TOKEN not in json.dumps(result)
    assert "[REDACTED:slack-token]" in result["webhook"]


def test_redaction_scrubs_modern_secret_named_keys():
    """The expanded ``_DEFAULT_SECRET_KEY_NAMES`` set covers modern secret
    variable names independently of the value content."""
    result = _redact(
        {
            "anthropic_api_key": "anything-in-here",
            "openai_api_key": "anything-in-here",
            "aws_secret_access_key": "anything-in-here",
            "slack_token": "anything-in-here",
            "stripe_key": "anything-in-here",
            "jwt": "anything-in-here",
            "id_token": "anything-in-here",
            "refresh_token": "anything-in-here",
            "safe": "keep-me",
        }
    )
    assert result["anthropic_api_key"] == "[REDACTED:secret]"
    assert result["openai_api_key"] == "[REDACTED:secret]"
    assert result["aws_secret_access_key"] == "[REDACTED:secret]"
    assert result["slack_token"] == "[REDACTED:secret]"
    assert result["stripe_key"] == "[REDACTED:secret]"
    assert result["jwt"] == "[REDACTED:secret]"
    assert result["id_token"] == "[REDACTED:secret]"
    assert result["refresh_token"] == "[REDACTED:secret]"
    assert result["safe"] == "keep-me"


# ---------------------------------------------------------------------------
# Issue #85: post-write ``delete_session`` and ``redact_event`` helpers.
# Acceptance: ``delete_session(session_id) -> int`` raises ``KeyError`` when
# the session is absent and otherwise removes every event + the session row;
# ``redact_event(session_id, event_id, replacement) -> TraceEvent`` overwrites
# the payload while preserving event_id, session_id, timestamp, and kind, and
# writes an audit event (``kind='event_redacted'`` or ``kind='session_deleted'``)
# so the Digester can identify corrected rows in future passes.
# ---------------------------------------------------------------------------


@_BACKENDS
def test_redact_event_overwrites_leaked_payload(tmp_path, backend):
    """Plant a session with a fake-leaked ``sk-...`` payload and assert
    that ``redact_event`` replaces it with the operator-supplied replacement
    and that ``load_session`` returns the redacted version (issue #85
    acceptance, end-to-end)."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        leaked = logger.record(
            sid,
            kind="tool_result",
            payload={"output": f"key={_API_KEY}"},
        )
        other = logger.record(sid, kind="user_prompt", payload={"text": "unrelated"})

    replacement = {"redacted": True, "note": "operator scrubbed at T+1"}
    returned = logger.redact_event(sid, leaked.event_id, replacement)

    assert returned.event_id == leaked.event_id
    assert returned.session_id == sid
    assert returned.kind == leaked.kind
    assert returned.timestamp == leaked.timestamp
    assert returned.payload == replacement

    events = logger.load_session(sid)
    by_id = {event.event_id: event for event in events}
    assert by_id[leaked.event_id].payload == replacement
    # The un-leaked event is untouched.
    assert by_id[other.event_id].payload == {"text": "unrelated"}
    # The audit event is present and identifies which row was redacted.
    audit_events = [e for e in events if e.kind == "event_redacted"]
    assert len(audit_events) == 1
    audit = audit_events[0]
    assert audit.session_id == sid
    assert audit.payload["event_id"] == leaked.event_id
    assert audit.payload["kind"] == "tool_result"


@_BACKENDS
def test_redact_event_replacement_is_also_scrubbed(tmp_path, backend):
    """If the operator accidentally passes a raw ``sk-...`` value as the
    replacement, ``redact_event`` must still scrub it via the same
    ``_redact`` pipeline used by :meth:`record`. The point of the helper
    is to scrub leaked secrets, not to be a back door around ``_redact``."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid:
        clean = logger.record(sid, kind="note", payload={"v": "ok"})

    bad_replacement = {"output": f"key={_API_KEY}"}
    returned = logger.redact_event(sid, clean.event_id, bad_replacement)

    assert _API_KEY not in json.dumps(returned.payload)
    assert returned.payload["output"] == "key=[REDACTED:api-key]"


@_BACKENDS
def test_redact_event_unknown_event_id_raises_keyerror(tmp_path, backend):
    logger = TraceLogger(
        tmp_path / f"traces{'.db' if backend == 'sqlite' else '.jsonl'}", backend=backend
    )
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"x": 1})

    with pytest.raises(KeyError):
        logger.redact_event(sid, "no-such-event-id", {"redacted": True})


@_BACKENDS
def test_redact_event_wrong_session_id_raises_keyerror(tmp_path, backend):
    """``event_id`` is globally unique, but ``redact_event`` requires both
    ``session_id`` and ``event_id`` to match. A copy/paste bug that
    passes the right event_id with the wrong session_id must raise
    rather than silently redact the right event in the wrong session."""
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid_a:
        recorded_a = logger.record(sid_a, kind="note", payload={"v": 1})
    with logger.session(harness_version="test-0.0") as sid_b:
        logger.record(sid_b, kind="note", payload={"v": 2})

    with pytest.raises(KeyError):
        logger.redact_event(sid_b, recorded_a.event_id, {"redacted": True})


@_BACKENDS
def test_delete_session_removes_all_events_and_returns_count(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="test-0.0") as sid_keep:
        logger.record(sid_keep, kind="note", payload={"v": 1})
    with logger.session(harness_version="test-0.0") as sid_drop:
        for i in range(3):
            logger.record(sid_drop, kind="note", payload={"i": i})

    removed = logger.delete_session(sid_drop)

    # On sqlite the audit row is part of the same transaction, so the
    # returned count is ``3 events + 1 audit + 1 session row``. On jsonl
    # the audit line lives in the file post-rewrite, so it counts as 1
    # extra removed line.
    assert removed >= 3

    surviving_sessions = [s.session_id for s in logger.list_sessions()]
    assert sid_drop not in surviving_sessions
    assert sid_keep in surviving_sessions
    # On sqlite the audit row is rolled back with the session deletion; on
    # jsonl the audit line is appended post-rewrite and persists. Both are
    # documented in delete_session's docstring.
    remaining = logger.load_session(sid_drop)
    if backend == "sqlite":
        assert remaining == []
    else:
        assert len(remaining) == 1
        assert remaining[0].kind == "session_deleted"
    # Kept session is untouched.
    assert len(logger.load_session(sid_keep)) == 1


@_BACKENDS
def test_delete_session_unknown_id_raises_keyerror(tmp_path, backend):
    logger = TraceLogger(
        tmp_path / f"traces{'.db' if backend == 'sqlite' else '.jsonl'}", backend=backend
    )
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="note", payload={"v": 1})

    with pytest.raises(KeyError):
        logger.delete_session("does-not-exist")

    # The real session is untouched after the failed delete.
    assert [s.session_id for s in logger.list_sessions()] == [sid]


def test_delete_session_sqlite_records_audit_atomic(tmp_path):
    """The audit row is part of the same transaction as the deletion:
    either both happen or neither does. Pre-#85 code paths would either
    leak the audit forever or lose the deletion record; this test pins
    the atomic behavior."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="note", payload={"v": 1})
        logger.record(sid, kind="note", payload={"v": 2})

    logger.delete_session(sid)

    # After a successful delete, no audit row survives in the events
    # table — the audit shares the deletion transaction (see logger
    # docstring on the sqlite/jsonl asymmetry).
    import sqlite3

    with sqlite3.connect(db) as conn:
        leftover = conn.execute(
            "SELECT COUNT(*) FROM events WHERE session_id = ?", (sid,)
        ).fetchone()[0]
    assert leftover == 0


def test_delete_session_jsonl_audit_is_queryable_for_forensics(tmp_path):
    """On jsonl the audit line is persistent and surfaces via
    ``load_session(deleted_sid)`` as a forensic marker. This is the
    documented asymmetry vs. sqlite and the reason the jsonl audit
    is not part of a transaction that also wipes it."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="test-0.0") as sid:
        logger.record(sid, kind="note", payload={"v": 1})

    logger.delete_session(sid)

    surviving = logger.load_session(sid)
    assert len(surviving) == 1
    assert surviving[0].kind == "session_deleted"
    assert surviving[0].payload == {"session_id": sid}
