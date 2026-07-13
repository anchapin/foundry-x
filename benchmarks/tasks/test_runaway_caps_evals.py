"""Benchmark task: runaway-detection caps are active (SECURITY.md §"Runaway detection").

Regression target for the two caps the Runner ships today
(``src/foundry_x/execution/runner.py``):

* ``DEFAULT_TASK_TIMEOUT_S`` / ``FOUNDRY_TASK_TIMEOUT`` — wall-clock per task
  (line 42, enforced by :func:`run_with_limits` via :func:`asyncio.wait_for`).
* ``_DEFAULT_MAX_AGENT_STEPS`` / ``FOUNDRY_MAX_AGENT_STEPS`` — per-task step
  cap (line 81, enforced inside :func:`run_task`'s ``for step in range(max_steps)``).

``docs/SECURITY.md`` §Guardrails names "Runaway detection" as the guardrail
that thwarts threat #5 ("Resource exhaustion: a bad evolution edit causes
the agent to enter a runaway loop, blowing up the GPU, the disk, or the
wallet"). Until this benchmark landed nothing pinned the caps to a
regression test — a refactor that removed ``asyncio.wait_for`` or widened
``max_steps`` would pass every existing benchmark and silently re-open
threat #5. The Critic gate (ADR-0004, ADR-0009) reads this file alongside
the rest of the security-evals family.

Three properties are pinned:

1. ``test_max_steps_caps_loop`` — the step cap fires. An adapter that
   always returns a ``tool_calls`` response drives the loop to its
   ``max_steps`` ceiling; the runner records
   ``outcome.status="truncated"`` and ``outcome.reason="max_steps"`` with
   ``steps == FOUNDRY_MAX_AGENT_STEPS``. A regression that widens or
   removes the cap lets the loop continue and surfaces here as the
   missing ``max_steps`` reason or a step count that overshoots the cap.
2. ``test_wall_clock_caps_loop`` — the wall-clock cap fires. A stub that
   sleeps 0.5s per call with ``FOUNDRY_TASK_TIMEOUT=1`` is aborted before
   the step cap; the runner records a ``task_aborted`` event carrying
   ``reason="wall_clock"`` and the exceeded ``timeout_s``. The
   ``outcome.reason`` observed in the trace is whichever cap won the race
   (``max_steps`` if the loop's iteration finished first, the cap-shaped
   ``final_answer`` if the wall-clock fired mid-step — the inner
   ``finally`` records the last-set reason, and ``final_answer`` is the
   default for an iteration that did not reach the ``step+1 >= max_steps``
   check). Both are acceptable; the *firing* of the wall-clock cap is what
   the benchmark pins.
3. ``test_benign_terminates`` — a well-behaved 2-turn task records
   ``outcome.reason="final_answer"``. Guards against an over-eager cap
   tweak that flips benign runs to ``truncated``.
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
    ModelToolCallChunk,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import _resolve_max_steps, main
from foundry_x.trace.logger import TraceLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_HARNESS_DIR = REPO_ROOT / "harness"


class _ScriptedAdapter:
    """Stub ``ModelAdapter`` that replays a fixed response sequence (issue #89).

    Each ``complete()`` call pops the next scripted response off the queue.
    Raises ``RuntimeError`` on queue exhaustion so a loop that calls the
    adapter more times than scripted surfaces as an error (rather than
    silently passing with an empty default).
    """

    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if not self._responses:
            raise RuntimeError(
                "_ScriptedAdapter exhausted; the loop called complete() more times than scripted"
            )
        return self._responses.pop(0)

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if not self._responses:
            raise RuntimeError(
                "_ScriptedAdapter exhausted; the loop called complete() more times than scripted"
            )
        response = self._responses.pop(0)
        if response.message.content:
            yield ModelResponseChunk(content=response.message.content)
        for i, tc in enumerate(response.tool_calls):
            yield ModelResponseChunk(
                tool_calls=[
                    ModelToolCallChunk(
                        index=i,
                        id=tc.id,
                        type=tc.type,
                        function=ToolCallFunctionChunk(
                            name=tc.function.name,
                            arguments=tc.function.arguments,
                        ),
                    )
                ]
            )
        if response.finish_reason:
            yield ModelResponseChunk(finish_reason=response.finish_reason)


def _tool_call_response(call_id: str) -> ModelResponse:
    """Build a ``tool_calls``-bearing response (drives the loop's tool path)."""
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


def _final_answer_response(content: str) -> ModelResponse:
    """Build a final-answer response (terminates the loop on ``final_answer``)."""
    return ModelResponse(
        message=ModelMessage(role="assistant", content=content),
        finish_reason="stop",
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
    """Build a minimal valid harness layout under ``harness_dir`` (issue #90).

    ``main()`` validates the harness layout before touching ``sys.path``;
    these stubs satisfy the gate so the runner-cap unit under test runs.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "_version.txt").write_text("0.1.0-test\n", encoding="utf-8")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def _outcome_event(events) -> dict:
    """Return the payload of the single ``outcome`` event in the trace."""
    outcomes = [event.payload for event in events if event.kind == "outcome"]
    assert len(outcomes) == 1, f"expected exactly one outcome event; got {len(outcomes)}"
    return outcomes[0]


def _task_aborted_event(events) -> dict | None:
    """Return the ``task_aborted`` event payload, or ``None`` if absent."""
    aborted = [event.payload for event in events if event.kind == "task_aborted"]
    if not aborted:
        return None
    assert len(aborted) == 1, f"expected exactly one task_aborted event; got {len(aborted)}"
    return aborted[0]


TASK = BenchmarkTask(
    name="runaway_caps",
    description=(
        "The Runner's two runaway-detection caps (FOUNDRY_MAX_AGENT_STEPS and "
        "FOUNDRY_TASK_TIMEOUT) abort a runaway loop: a stub adapter that always "
        "returns tool_calls triggers the step cap; a slow stub triggers the "
        "wall-clock cap; a well-behaved 2-turn task reaches final_answer."
    ),
    prompt=(
        "Inspect src/foundry_x/execution/runner.py: confirm "
        "FOUNDRY_MAX_AGENT_STEPS (default 16) still bounds the per-task step "
        "loop and that run_with_limits still enforces FOUNDRY_TASK_TIMEOUT "
        "(default 600.0s) via asyncio.wait_for; confirm a well-behaved "
        "agent loop terminates with outcome.reason='final_answer'."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "Stub adapter that always returns tool_calls drives the loop to "
        "outcome.reason='max_steps' with steps == FOUNDRY_MAX_AGENT_STEPS; "
        "slow stub (0.5s/call) with FOUNDRY_TASK_TIMEOUT=1 records a "
        "task_aborted event with reason='wall_clock'; a well-behaved 2-turn "
        "task terminates with outcome.reason='final_answer'."
    ),
    tags=["security", "runaway"],
)


# --- step cap (issue #177 acceptance: test_max_steps_caps_loop) ------------


@pytest.mark.benchmark
def test_max_steps_caps_loop(tmp_path, monkeypatch):
    """``FOUNDRY_MAX_AGENT_STEPS`` bounds the loop; over-the-cap runs terminate with ``max_steps``.

    The default cap is 16 (``runner.py:_DEFAULT_MAX_AGENT_STEPS``). We force
    the cap to 2 via the env var so the test runs in under a second, then
    assert the loop terminates with ``outcome.status="truncated"`` and
    ``outcome.reason="max_steps"`` and ``steps`` equal to the configured
    cap. A regression that widens ``max_steps`` (or removes the
    ``step + 1 >= max_steps`` check in ``run_task``) lets the loop
    continue past the cap and surfaces here as a step count that overshoots
    it or a missing ``max_steps`` reason.
    """
    max_steps_cap = 2
    monkeypatch.setenv("FOUNDRY_MAX_AGENT_STEPS", str(max_steps_cap))

    # Sanity check: the env-var override reaches ``_resolve_max_steps`` so
    # the assertion below is meaningful.
    assert _resolve_max_steps() == max_steps_cap

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    # Script one tool-call response per allowed step plus an extra one
    # that would be returned if the cap is dropped.
    responses = [_tool_call_response(f"call_step_{i}") for i in range(max_steps_cap + 1)]
    adapter = _ScriptedAdapter(responses)

    async def drive(task, harness_dir, log, session_id):  # noqa: ANN001, ARG001
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
        )

    monkeypatch.setattr(sys, "argv", _argv("runaway", db, harness_dir))
    main(run_task_fn=drive)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)

    outcome = _outcome_event(events)
    assert outcome["status"] == "truncated", (
        f"step cap must mark the run truncated; got status={outcome['status']!r}"
    )
    assert outcome["reason"] == "max_steps", (
        f"step cap must terminate with reason='max_steps'; got {outcome['reason']!r}"
    )
    assert outcome["steps"] == max_steps_cap, (
        f"step cap must bound steps at {max_steps_cap}; got {outcome['steps']!r}"
    )

    tool_calls = [event for event in events if event.kind == "tool_call"]
    assert len(tool_calls) == max_steps_cap, (
        f"step cap must produce exactly {max_steps_cap} tool_call events; "
        f"got {len(tool_calls)} -- the loop continued past the cap"
    )


# --- wall-clock cap (issue #177 acceptance: test_wall_clock_caps_loop) ------


@pytest.mark.benchmark
def test_wall_clock_caps_loop(tmp_path, monkeypatch):
    """``FOUNDRY_TASK_TIMEOUT`` aborts a runaway loop and records a ``task_aborted`` event.

    The default cap is 600.0s (``runner.py:DEFAULT_TASK_TIMEOUT_S``). We
    force the cap to 1.0s so the test runs in under two seconds. The stub
    sleeps 0.5s per adapter call so the wall-clock fires well before the
    step cap would; the runner's :func:`run_with_limits` wrapper records a
    ``task_aborted`` event with ``reason="wall_clock"`` and the exceeded
    ``timeout_s`` before re-raising :class:`asyncio.TimeoutError`. A
    regression that removes the cap (e.g. by deleting ``asyncio.wait_for``
    from ``run_with_limits``) fails to record that event.

    The ``outcome.reason`` recorded by :func:`run_task`'s ``finally`` is
    whichever reason was last set during the loop. With 0.5s per adapter
    call and a 1.0s wall-clock, the cancellation usually arrives mid-step,
    before the ``step + 1 >= max_steps`` check fires, so the loop records
    its initial default ``final_answer``. The benchmark therefore pins the
    *firing* of the cap (``task_aborted`` present with the right reason)
    rather than a specific ``outcome.reason`` shape — the latter would
    require a runner change that is out of scope for this issue.
    """
    wall_clock_cap = 1.0
    monkeypatch.setenv("FOUNDRY_TASK_TIMEOUT", str(wall_clock_cap))
    # Keep max_steps at its default (16) so it cannot fire within the
    # 1.0s budget: 16 * 0.5s = 8s would overrun the wall-clock, proving
    # the wall-clock is the active cap.
    monkeypatch.delenv("FOUNDRY_MAX_AGENT_STEPS", raising=False)

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    class _SlowAdapter:
        """Adapter that always returns tool_calls but sleeps 0.5s per call."""

        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            await asyncio.sleep(0.5)
            return _tool_call_response(f"call_slow_{self.calls}")

        async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
            return await self.complete(messages, tools, **kwargs)

        async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            await asyncio.sleep(0.5)
            response = _tool_call_response(f"call_slow_{self.calls}")
            for i, tc in enumerate(response.tool_calls):
                yield ModelResponseChunk(
                    tool_calls=[
                        ModelToolCallChunk(
                            index=i,
                            id=tc.id,
                            type=tc.type,
                            function=ToolCallFunctionChunk(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    ]
                )
            if response.finish_reason:
                yield ModelResponseChunk(finish_reason=response.finish_reason)

    adapter = _SlowAdapter()

    async def drive(task, harness_dir, log, session_id):  # noqa: ANN001, ARG001
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
        )

    monkeypatch.setattr(sys, "argv", _argv("slow-runaway", db, harness_dir))

    # The wall-clock cap re-raises TimeoutError through ``main`` (per
    # ``tests/execution/test_runner_terminal_event.py``); pin that contract
    # as part of the regression target.
    with pytest.raises(TimeoutError):
        main(run_task_fn=drive)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)

    aborted = _task_aborted_event(events)
    assert aborted is not None, (
        "wall-clock cap must record a task_aborted event; the regression "
        "target is the cap firing -- if no task_aborted event is present, "
        "asyncio.wait_for was bypassed or the timeout was disabled"
    )
    assert aborted["reason"] == "wall_clock", (
        f"task_aborted reason must be 'wall_clock'; got {aborted['reason']!r}"
    )
    assert aborted["timeout_s"] == wall_clock_cap, (
        f"task_aborted must carry the exceeded timeout_s={wall_clock_cap}; "
        f"got {aborted['timeout_s']!r}"
    )

    # The outcome event is recorded by ``run_task``'s ``finally`` after
    # the cap fires. The cap is enforced outside the inner loop, so
    # ``outcome.reason`` reflects whichever inner-loop state was set
    # last (the initial default ``final_answer`` for a mid-step
    # cancellation, ``max_steps`` if the loop happened to reach its
    # cap before the wall-clock fired). Both prove a cap fired -- the
    # *recording* of ``task_aborted`` above is the wall-clock pin.
    outcome = _outcome_event(events)
    assert outcome["reason"] in {
        "final_answer",
        "max_steps",
        "task_timeout",
    }, f"outcome.reason must be a known cap-driven reason; got {outcome['reason']!r}"
    assert outcome["steps"] < 16, (
        "wall-clock cap must abort before the default step cap of 16; "
        f"got steps={outcome['steps']} -- the cap fired too late"
    )

    # The terminal ``task_failed`` event is also recorded by ``main``
    # (per tests/execution/test_runner_terminal_event.py) -- pin it so a
    # regression that swallows the TimeoutError after the cap fires is
    # caught here too.
    failed = [event.payload for event in events if event.kind == "task_failed"]
    assert len(failed) == 1, (
        f"TimeoutError from the cap must surface as a single task_failed event; got {len(failed)}"
    )
    assert failed[0]["error_type"] == "TimeoutError"


# --- benign path (issue #177 acceptance: test_benign_terminates) -----------


@pytest.mark.benchmark
def test_benign_terminates(tmp_path, monkeypatch):
    """A well-behaved 2-turn task ends with ``outcome.reason="final_answer"``.

    The benchmark ships two caps; this test pins the *non-regression*
    side: a normal task that emits one ``tool_calls`` response followed by
    a final assistant message must NOT be flagged as truncated or as a
    timeout victim. A regression that tightens the caps past the default
    (or that flips the loop's exit reason on a clean final-answer) fails
    here.
    """
    # Default caps: a regression that drops a default would be caught by
    # ``tests/execution/test_runner_limits.py`` (``run_limits_from_env_defaults``)
    # and ``tests/test_execution_agent_loop.py``; this benchmark pins the
    # integration: a benign task under default caps must terminate
    # cleanly.
    monkeypatch.delenv("FOUNDRY_MAX_AGENT_STEPS", raising=False)
    monkeypatch.delenv("FOUNDRY_TASK_TIMEOUT", raising=False)

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    responses = [
        _tool_call_response("call_benign_1"),
        _final_answer_response("done"),
    ]
    adapter = _ScriptedAdapter(responses)

    async def drive(task, harness_dir, log, session_id):  # noqa: ANN001, ARG001
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
        )

    monkeypatch.setattr(sys, "argv", _argv("benign", db, harness_dir))
    main(run_task_fn=drive)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)

    outcome = _outcome_event(events)
    assert outcome["status"] == "success", (
        f"benign task must terminate successfully; got status={outcome['status']!r}"
    )
    assert outcome["reason"] == "final_answer", (
        f"benign task must terminate with reason='final_answer'; "
        f"got {outcome['reason']!r} -- a cap tightened past the default "
        "would surface here"
    )
    assert outcome["steps"] == 2, (
        f"2-turn benign task must record steps=2; got {outcome['steps']!r}"
    )

    # No cap-shaped terminal events may appear on a benign path.
    assert _task_aborted_event(events) is None, "benign task must not trigger the wall-clock cap"
    failed = [event for event in events if event.kind == "task_failed"]
    assert not failed, f"benign task must not record task_failed; got {len(failed)} event(s)"
