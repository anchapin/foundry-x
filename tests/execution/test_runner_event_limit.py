"""Integration tests for ``max_events_per_session`` (issue #708)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry_x.execution.model_adapter import (
    ModelResponseChunk,
    ModelToolCallChunk,
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
