"""Pydantic model validation and round-trip tests for trace events (issue #7).

Acceptance per ADR-0006 / issue #7:
- Constructing ``TraceEvent(kind="")`` must raise ``ValidationError``.
- A round-trip through ``record()`` -> ``load_session()`` is byte-stable
  for payloads containing nested dicts, on both sqlite and jsonl backends.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from foundry_x.trace.logger import TraceEvent, TraceLogger

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def test_trace_event_empty_kind_raises_validation_error():
    with pytest.raises(ValidationError):
        TraceEvent(
            event_id="evt-1",
            session_id="sess-1",
            timestamp="2026-07-10T00:00:00+00:00",
            kind="",
            payload={},
        )


def test_trace_event_round_trips_through_model_validate():
    original = TraceEvent(
        event_id="evt-rt",
        session_id="sess-rt",
        timestamp="2026-07-10T01:00:00+00:00",
        kind="tool_call",
        payload={"nested": {"deep": [1, 2, {"x": True}]}},
    )
    restored = TraceEvent.model_validate(original.model_dump())
    assert restored == original


@_BACKENDS
def test_record_load_round_trip_nested_dicts(tmp_path, backend):
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)
    payload = {
        "outer": {"inner": [1, 2, 3]},
        "flag": True,
        "nested": {"deep": {"value": 42, "text": "hello"}},
    }
    with logger.session(harness_version="test-0.0") as sid:
        recorded = logger.record(sid, kind="user_prompt", payload=payload)

    events = logger.load_session(sid)
    assert len(events) == 1
    assert events[0].payload == payload
    assert events[0].payload == recorded.payload


@_BACKENDS
def test_critic_verdict_event_joins_to_session_model_id(tmp_path, backend):
    """critic_verdict events can be joined to their session's model_id (issue #361).

    Phase 3's core KPI — Improvement Rate — requires attributing benchmark
    outcomes to specific quantizations. This test verifies that a critic_verdict
    event recorded in a session with a known model_id can be correctly attributed
    by joining the events table to the sessions table.
    """
    suffix = ".db" if backend == "sqlite" else ".jsonl"
    path = tmp_path / f"traces{suffix}"
    logger = TraceLogger(path, backend=backend)

    model_id = "codellama-7b.Q5_K_M.gguf"
    with logger.session(harness_version="0.1.0", model_id=model_id) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        logger.record(
            sid, kind="critic_verdict", payload={"approved": True, "reason": "all checks pass"}
        )

    sessions = logger.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].model_id == model_id

    events = logger.load_session(sid)
    verdict_events = [e for e in events if e.kind == "critic_verdict"]
    assert len(verdict_events) == 1
    assert verdict_events[0].payload["approved"] is True

    session_for_event = next(s for s in sessions if s.session_id == verdict_events[0].session_id)
    assert session_for_event.model_id == model_id
