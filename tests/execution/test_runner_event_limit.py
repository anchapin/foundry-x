"""Integration tests for ``max_events_per_session`` (issue #708)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry_x.execution.model_adapter import (
    ModelResponseChunk,
    ModelToolCallChunk,
    ModelUsage,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import RunLimits, run_task as real_run_task
from foundry_x.trace.logger import TraceLogger


def _stub_harness(harness_dir: Path) -> None:
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


class _ScriptedAdapter:
    """Adapter that yields two turns: first emits a tool call, second yields final answer."""

    def __init__(self) -> None:
        self._turn = 0

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self._turn += 1
        if self._turn == 1:
            yield ModelResponseChunk(
                tool_calls=[
                    ModelToolCallChunk(
                        index=0,
                        id="call_1",
                        type="function",
                        function=ToolCallFunctionChunk(
                            name="bash",
                            arguments='{"command": "true"}',
                        ),
                    )
                ]
            )
            yield ModelResponseChunk(finish_reason="tool_calls")
            return
        yield ModelResponseChunk(content="done")
        yield ModelResponseChunk(finish_reason="stop")

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream()")

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream()")


async def _executor(name: str, arguments: dict) -> dict:  # noqa: ANN001
    return {"status": "ok"}


@pytest.mark.asyncio
async def test_max_events_per_session_aborts_with_event_limit(tmp_path, monkeypatch):
    """Issue #708 acceptance: when max_events_per_session is exceeded,
    run_task emits task_aborted(reason="event_limit") and outcome status=failed."""
    import foundry_x.execution.runner as runner_mod

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    monkeypatch.setattr(runner_mod, "build_model_adapter", _ScriptedAdapter)

    limits = RunLimits(max_events_per_session=3)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "event-limit-test",
            harness_dir,
            logger,
            session_id,
            skill_executor=_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 1, f"expected 1 task_aborted, got {aborted}"
    assert aborted[0].payload["reason"] == "event_limit"
    assert aborted[0].payload["max_events_per_session"] == 3

    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "failed"
    assert outcome.payload["reason"] == "event_limit"


@pytest.mark.asyncio
async def test_max_events_per_session_none_does_not_abort(tmp_path, monkeypatch):
    """When max_events_per_session is None (default), the session runs to completion."""
    import foundry_x.execution.runner as runner_mod

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    monkeypatch.setattr(runner_mod, "build_model_adapter", _ScriptedAdapter)

    limits = RunLimits(max_events_per_session=None)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "no-limit-test",
            harness_dir,
            logger,
            session_id,
            skill_executor=_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 0, f"expected no task_aborted, got {aborted}"

    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "success"
    assert outcome.payload["reason"] == "final_answer"


@pytest.mark.asyncio
async def test_max_events_per_session_under_limit(tmp_path, monkeypatch):
    """When the event count stays under the limit, the session runs to completion."""
    import foundry_x.execution.runner as runner_mod

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    monkeypatch.setattr(runner_mod, "build_model_adapter", _ScriptedAdapter)

    limits = RunLimits(max_events_per_session=100)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "under-limit-test",
            harness_dir,
            logger,
            session_id,
            skill_executor=_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 0, f"expected no task_aborted, got {aborted}"

    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "success"


class _FiveChunksPerResponseAdapter:
    """Adapter that emits exactly 5 SSE chunks per response (issue #790).

    Turn 1: 5 chunks containing a single ``bash`` tool call.
    Turn 2: 5 chunks with final assistant content + ``stop``.

    Without the fix, ``event_count += chunk_count`` adds 5 per turn on
    top of the logical events, so a session configured with
    ``max_events_per_session=10`` aborts mid-tool-call after turn 1.
    With the fix, only logical events count toward the limit and the
    session completes normally.
    """

    def __init__(self) -> None:
        self._turn = 0

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self._turn += 1
        usage = ModelUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2)
        if self._turn == 1:
            yield ModelResponseChunk(
                tool_calls=[
                    ModelToolCallChunk(
                        index=0,
                        id="call_1",
                        type="function",
                        function=ToolCallFunctionChunk(
                            name="bash",
                            arguments='{"command": "true"}',
                        ),
                    )
                ],
            )
            for _ in range(3):
                yield ModelResponseChunk(content="", usage=usage)
            yield ModelResponseChunk(finish_reason="tool_calls", usage=usage)
            return
        for _ in range(4):
            yield ModelResponseChunk(content="done", usage=usage)
        yield ModelResponseChunk(finish_reason="stop", usage=usage)

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream()")

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream()")


@pytest.mark.asyncio
async def test_streaming_chunks_do_not_count_toward_event_limit(tmp_path, monkeypatch):
    """Issue #790 regression: per-chunk ``model_response_chunk`` events must not
    inflate the local ``event_count`` used by ``max_events_per_session``.

    Prior to the fix, ``run_task`` did ``event_count += chunk_count`` after
    ``_consume_model_stream`` returned, in addition to the chunks already
    being recorded via raw ``log.record()`` inside ``_consume_model_stream``.
    That double-count caused the limit to fire approximately 2x earlier
    than configured when the adapter streamed many deltas per turn.
    """
    import foundry_x.execution.runner as runner_mod

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    monkeypatch.setattr(runner_mod, "build_model_adapter", _FiveChunksPerResponseAdapter)

    limits = RunLimits(max_events_per_session=10)

    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "chunk-count-test",
            harness_dir,
            logger,
            session_id,
            skill_executor=_executor,
            limits=limits,
        )

    events = logger.load_session(session_id)
    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 0, (
        f"session aborted by event_limit despite staying under the configured "
        f"logical-event budget: {aborted}"
    )

    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "success"
    assert outcome.payload["reason"] == "final_answer"

    chunk_events = [e for e in events if e.kind == "model_response_chunk"]
    assert len(chunk_events) == 10, (
        f"expected 10 model_response_chunk events (5 chunks x 2 turns), got {len(chunk_events)}"
    )
