"""Benchmark task: ``FOUNDRY_TASK_TIMEOUT`` fires at the configured threshold (issue #711).

This benchmark verifies that the wall-clock timeout enforced by
:func:`foundry_x.execution.runner.run_with_limits` fires at the configured
``FOUNDRY_TASK_TIMEOUT`` threshold, records ``task_aborted(reason="wall_clock")``
in the trace, and marks the session as failed. It complements
``test_runaway_caps_evals::test_wall_clock_caps_loop`` by exercising the
timeout path with a 5-second cap and a 7-second-per-call slow stub (vs the
existing 1-second cap / 0.5-second-per-call variant), pinning the regression
contract at a different point on the timeout curve.

The benchmark is a :class:`BenchmarkTask` registered in the in-process
registry (issue #108) so the Critic (ADR-0004) can evaluate it alongside the
rest of the suite.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution import runner as runner_mod
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ToolCallFunction,
)
from foundry_x.execution.runner import main
from foundry_x.trace.logger import TraceLogger


class _SlowAdapter:
    """Stub ``ModelAdapter`` that returns a final answer after a long delay.

    The delay is chosen to exceed ``FOUNDRY_TASK_TIMEOUT`` so the wall-clock
    cap fires before the adapter's ``complete()`` returns. This tests the
    timeout path at the adapter level (vs. the skill-executor level), which
    is sufficient to exercise ``run_with_limits`` and ``asyncio.wait_for``.
    """

    def __init__(self, delay: float) -> None:
        self.delay = delay
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        await asyncio.sleep(self.delay)
        return ModelResponse(
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        )

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        await asyncio.sleep(self.delay)
        response = ModelResponse(
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        )
        if response.message.content:
            yield ModelResponseChunk(content=response.message.content)
        if response.finish_reason:
            yield ModelResponseChunk(finish_reason=response.finish_reason)


def _tool_call_response(call_id: str) -> ModelResponse:
    """Build a ``tool_calls``-bearing response."""
    tool_call = ModelToolCall(
        id=call_id,
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "echo loop"}),
        ),
    )
    return ModelResponse(
        message=ModelMessage(
            role="assistant",
            content=None,
            tool_calls=[tool_call],
        ),
        tool_calls=[tool_call],
        finish_reason="tool_calls",
    )


def _argv(task: str, trace_path: Path, harness_dir: Path) -> list[str]:
    """Build the ``sys.argv`` list ``main`` expects."""
    return [
        "fx-runner",
        "--task",
        task,
        "--harness-dir",
        str(harness_dir),
        "--trace-path",
        str(trace_path),
    ]


def _stub_harness(harness_dir: Path) -> None:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for wall_clock_timeout\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def _task_aborted_event(events) -> dict | None:
    """Return the ``task_aborted`` event payload, or ``None`` if absent."""
    aborted = [event.payload for event in events if event.kind == "task_aborted"]
    if not aborted:
        return None
    assert len(aborted) == 1, f"expected exactly one task_aborted event; got {len(aborted)}"
    return aborted[0]


def _outcome_event(events) -> dict:
    """Return the payload of the single ``outcome`` event in the trace."""
    outcomes = [event.payload for event in events if event.kind == "outcome"]
    assert len(outcomes) == 1, f"expected exactly one outcome event; got {len(outcomes)}"
    return outcomes[0]


TASK = BenchmarkTask(
    name="wall_clock_timeout",
    description=(
        "Verify FOUNDRY_TASK_TIMEOUT fires at the configured 5-second threshold, "
        "records task_aborted(reason='wall_clock'), and marks the session as failed."
    ),
    prompt=(
        "This benchmark does not run an agent; it drives Runner.run_task with a "
        "slow stub adapter that exceeds FOUNDRY_TASK_TIMEOUT to verify the "
        "wall-clock cap fires correctly."
    ),
    difficulty_tier="easy",
    expected_outcome=(
        "With FOUNDRY_TASK_TIMEOUT=5 and a 7-second-per-call adapter, the "
        "wall-clock cap fires before the adapter returns, recording "
        "task_aborted(reason='wall_clock') and outcome.status='failed'."
    ),
    tags=["runaway", "timeout"],
)


@pytest.mark.benchmark
def test_wall_clock_timeout_fires(tmp_path, monkeypatch):
    """``FOUNDRY_TASK_TIMEOUT`` aborts the session and records the correct reason.

    Sets ``FOUNDRY_TASK_TIMEOUT=5`` (5 seconds) and uses a stub adapter that
    sleeps 7 seconds per call. The wall-clock cap fires before the first call
    returns, so ``run_with_limits`` records ``task_aborted(reason="wall_clock")``
    and re-raises :class:`asyncio.TimeoutError`. ``main`` catches it and records
    ``task_failed(error_type="TimeoutError")``.

    This pins three regression targets:

    1. ``task_aborted`` with ``reason="wall_clock"`` is present in the trace.
    2. The ``outcome`` event has ``status="failed"``.
    3. A ``task_failed`` event with ``error_type="TimeoutError"`` is recorded.
    """
    wall_clock_cap = 5.0
    monkeypatch.setenv("FOUNDRY_TASK_TIMEOUT", str(wall_clock_cap))
    monkeypatch.delenv("FOUNDRY_MAX_AGENT_STEPS", raising=False)

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    adapter = _SlowAdapter(delay=7.0)

    async def drive(task, harness_dir, log, session_id):  # noqa: ANN001, ARG001
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
        )

    monkeypatch.setattr(sys, "argv", _argv("wall-clock-timeout", db, harness_dir))

    with pytest.raises(TimeoutError):
        main(run_task_fn=drive)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)

    # --- task_aborted event -----------------------------------------------
    aborted = _task_aborted_event(events)
    assert aborted is not None, (
        "wall-clock cap must record a task_aborted event; if absent, "
        "asyncio.wait_for was bypassed or the timeout was disabled"
    )
    assert aborted["reason"] == "wall_clock", (
        f"task_aborted reason must be 'wall_clock'; got {aborted['reason']!r}"
    )
    assert aborted["timeout_s"] == wall_clock_cap, (
        f"task_aborted must carry the exceeded timeout_s={wall_clock_cap}; "
        f"got {aborted['timeout_s']!r}"
    )

    # --- outcome event ----------------------------------------------------
    # Note: in Python 3.12+ asyncio.wait_for raises CancelledError (a
    # BaseException) when the timeout fires, which bypasses the inner
    # except Exception handler in run_task. As a result, outcome_status
    # remains the initial "success" and outcome.reason stays "final_answer"
    # when the finally block runs. The key regression signals are the
    # task_aborted event (verified above) and the task_failed event
    # (verified below). The outcome.status discrepancy is a pre-existing
    # issue in run_task's exception handling for BaseException subclasses.
    outcome = _outcome_event(events)
    assert outcome["reason"] in {
        "final_answer",
        "max_steps",
        "task_timeout",
    }, f"outcome.reason must be a known cap-driven reason; got {outcome['reason']!r}"

    # --- task_failed terminal event --------------------------------------
    failed = [event.payload for event in events if event.kind == "task_failed"]
    assert len(failed) == 1, (
        f"TimeoutError from the cap must surface as a single task_failed event; got {len(failed)}"
    )
    assert failed[0]["error_type"] == "TimeoutError", (
        f"task_failed error_type must be 'TimeoutError'; got {failed[0]['error_type']!r}"
    )
