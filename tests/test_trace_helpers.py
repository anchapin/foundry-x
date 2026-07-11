"""Post-write correction helpers for :class:`TraceLogger` (issue #157).

Acceptance from the issue body:

- ``TraceLogger.delete_session(session_id)`` removes the session and its
  events from the underlying store. Idempotent: a second call with the
  same (or never-existed) ``session_id`` returns ``True`` rather than
  raising, so an operator running it on a stale id does not see an
  error.
- ``TraceLogger.redact_event(session_id, event_index, key)`` rewrites
  ``payload[key]`` to ``"[REDACTED]"`` on the indexed event. The
  redaction is persisted; a fresh ``TraceLogger`` reads the redacted
  value back.

Both helpers operate on both the sqlite and jsonl backends.
"""

from __future__ import annotations

import json

import pytest

from foundry_x.trace.logger import TraceLogger

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def _suffix(backend: str) -> str:
    return ".db" if backend == "sqlite" else ".jsonl"


@_BACKENDS
def test_redact_event_replaces_payload_value_with_redacted(tmp_path, backend):
    """``redact_event`` overwrites ``payload[key]`` with ``"[REDACTED]"``."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    api_key = "sk-" + "1234567890abcdef"
    bearer = "Bearer abcdefghij1234567890"
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            kind="tool_result",
            payload={
                "output": f"key={api_key}",
                "header": f"Authorization: {bearer}",
                "metadata": "keep-me",
            },
        )
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.redact_event(sid, event_index=0, key="output") is True
    assert logger.redact_event(sid, event_index=0, key="header") is True

    events = logger.load_session(sid)
    payload = events[0].payload
    assert payload["output"] == "[REDACTED]"
    assert payload["header"] == "[REDACTED]"
    assert payload["metadata"] == "keep-me"
    assert "[REDACTED]" in json.dumps(payload)
    assert api_key not in json.dumps(payload)
    assert bearer not in json.dumps(payload)
    assert events[1].payload == {"text": "hi"}


@_BACKENDS
def test_redact_event_persists_across_new_logger_instance(tmp_path, backend):
    """The redaction is written to disk, so a fresh ``TraceLogger``
    instance on the same path reads the redacted value back."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    api_key = "sk-" + "abcdef0123456789"
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            kind="tool_call",
            payload={"secret": api_key, "name": "read_file"},
        )

    assert logger.redact_event(sid, event_index=0, key="secret") is True

    fresh = TraceLogger(path, backend=backend)
    events = fresh.load_session(sid)
    assert events[0].payload["secret"] == "[REDACTED]"
    assert events[0].payload["name"] == "read_file"
    assert api_key not in json.dumps(events[0].payload)


@_BACKENDS
def test_redact_event_out_of_range_returns_false(tmp_path, backend):
    """A stale ``event_index`` returns ``False`` without rewriting anything."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.redact_event(sid, event_index=99, key="text") is False
    assert logger.redact_event(sid, event_index=-1, key="text") is False

    events = logger.load_session(sid)
    assert events[0].payload == {"text": "hi"}


@_BACKENDS
def test_redact_event_unknown_session_returns_false(tmp_path, backend):
    """A non-existent ``session_id`` returns ``False`` without crashing."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert logger.redact_event("never-existed", event_index=0, key="text") is False


@_BACKENDS
def test_delete_session_removes_target_and_keeps_others(tmp_path, backend):
    """``delete_session`` removes the target session and its events, leaving
    every other session in the store untouched."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid_drop:
        logger.record(sid_drop, kind="user_prompt", payload={"text": "drop-me"})
        logger.record(sid_drop, kind="tool_call", payload={"name": "read"})
    with logger.session(harness_version="0.1.0") as sid_keep:
        logger.record(sid_keep, kind="user_prompt", payload={"text": "keep-me"})

    assert logger.delete_session(sid_drop) is True

    assert logger.load_session(sid_drop) == []
    surviving = [s.session_id for s in logger.list_sessions()]
    assert sid_drop not in surviving
    assert sid_keep in surviving
    assert len(logger.load_session(sid_keep)) == 1
    assert logger.load_session(sid_keep)[0].payload == {"text": "keep-me"}


@_BACKENDS
def test_delete_session_persists_across_new_logger_instance(tmp_path, backend):
    """The deletion is written to disk, so a fresh ``TraceLogger``
    instance on the same path sees the post-delete state."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "x"})

    assert logger.delete_session(sid) is True

    fresh = TraceLogger(path, backend=backend)
    assert fresh.load_session(sid) == []
    assert [s.session_id for s in fresh.list_sessions()] == []


@_BACKENDS
def test_delete_session_is_idempotent(tmp_path, backend):
    """A second ``delete_session`` on the same id, or one on a never-existed
    id, returns ``True`` rather than raising."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    assert logger.delete_session(sid) is True
    assert logger.delete_session(sid) is True
    assert logger.delete_session("never-existed") is True


@_BACKENDS
def test_delete_session_on_empty_store_returns_true(tmp_path, backend):
    """Calling ``delete_session`` on a never-touched store is a no-op."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert logger.delete_session("anything") is True
