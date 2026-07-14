"""Benchmark task: Runner-driven stub-ModelAdapter smoke (issue #174).

This task is the gate-level smoke benchmark for the asyncio agent loop
that landed under issue #89 (ADR-0010). Every other benchmark task under
``benchmarks/tasks/`` invokes ``benchmarks.support.run_solution`` -- it
writes a hardcoded golden solution to disk and runs it as a subprocess --
which only verifies that "the gold solution still works". It does NOT
verify that the Runner drives a task through the asyncio agent loop, that
the trace events land in the documented sequence, or that
``HookRegistry.on_error`` stays silent on the happy path. Two of those
three signals are exactly what the improvement-rate KPI (PRD S5) is
supposed to measure, but today the KPI sees only the first signal (the
gold solution's exit code).

This task fills the gap. It drives ``Runner.run_task`` directly against a
stub ``ModelAdapter`` that emits one ``tool_call`` followed by a final
assistant message, captures the trace, and asserts:

1. The captured trace carries every documented event kind from the agent
   loop in a logically consistent order -- ``user_prompt`` first,
   ``outcome`` last, ``tool_call`` precedes its corresponding
   ``tool_result`` (issue #174 acceptance criteria).
2. The terminal ``outcome`` event reports ``reason == 'final_answer'``.
3. ``HookRegistry.on_error`` received zero hook failures on the happy path.

It is registered as a :class:`BenchmarkTask` (issue #108) so the in-process
registry enumerates it alongside every other benchmark, and tagged with
``agent-loop`` so the Critic (ADR-0004) groups it with the integration
suite in ``tests/test_execution_agent_loop.py`` and against ADR-0010.

The harness layout lives entirely under ``benchmark_workspace`` so this
benchmark never touches the repository's evolved harness (AGENTS.md §7
self-reference rule).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger


TASK = BenchmarkTask(
    name="runner_loop_smoke",
    description=(
        "Drive Runner.run_task against a stub ModelAdapter that emits one "
        "tool_call then a final assistant message; assert the documented "
        "trace event vocabulary, outcome.reason == 'final_answer', and "
        "zero HookRegistry.on_error callbacks on the happy path "
        "(issue #174)."
    ),
    tags=["agent-loop"],
    difficulty_tier="easy",
)


class _StubAdapter:
    """Stub ``ModelAdapter`` that replays a scripted response sequence.

    The first ``complete()`` call returns a response carrying one ``bash``
    ``tool_call`` (the "tool step"). The second ``complete()`` call returns
    a plain final assistant message (the "final answer"). If the loop calls
    ``complete()`` more than twice, a ``RuntimeError`` is raised so a
    runaway loop (SECURITY.md "Runaway detection") surfaces as a hard test
    failure instead of silently succeeding against a default response.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls == 1:
            tool_call = ModelToolCall(
                id="call_bash_smoke",
                type="function",
                function=ToolCallFunction(
                    name="bash",
                    arguments=json.dumps({"command": "echo hello"}),
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
        if self.calls == 2:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
            )
        raise RuntimeError(
            f"_StubAdapter exhausted after 2 scripted responses; loop called "
            f"complete() {self.calls} times (possible runaway loop, see "
            f"SECURITY.md 'Runaway detection')"
        )

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        response = await self.complete(messages, tools, **kwargs)
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


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``.

    The Runner reads ``system_prompt.txt`` to seed the system message and
    walks ``skills/`` for the tool surface; both may be empty for this
    smoke benchmark because the stub skill executor acknowledges every call
    without consulting the filesystem.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for runner_loop_smoke\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


def _install_on_error_tracker(tracker):  # noqa: ANN001
    """Install ``tracker`` on the default ``HookRegistry`` for the test.

    Resets the registry so state from prior tests does not leak in, then
    re-registers the prompt-injection firewall so the runtime environment
    matches the production hook surface (SECURITY.md "Prompt-input
    firewall" must run by default). The tracker is the
    :class:`HookErrorCallback` the registry routes any isolated hook
    failure into -- the assertion at the bottom of the test reads from it.

    Returns the registry so the caller can ``reset_default_registry()``
    after the run to avoid leaking the tracker into other tests.
    """
    from harness.hooks import get_registry
    from harness.hooks.base import reset_default_registry
    from harness.hooks.injection_firewall import InjectionFirewallHook

    reset_default_registry()
    registry = get_registry()
    registry.register(InjectionFirewallHook())
    registry._on_error = tracker  # type: ignore[assignment]
    return registry


@pytest.mark.benchmark
def test_runner_loop_smoke(benchmark_workspace: Path) -> None:
    """Runner-driven stub-ModelAdapter smoke benchmark (issue #174).

    Drives ``Runner.run_task`` with a stub ``ModelAdapter`` (one
    ``tool_call`` then final answer) and asserts:

    1. The trace carries every documented event kind from the agent loop
       in a logically consistent order -- ``user_prompt`` first,
       ``outcome`` last, ``tool_call`` precedes its corresponding
       ``tool_result`` (ADR-0010 §Consequences; issue #174 acceptance
       criteria).
    2. The terminal ``outcome`` event reports
       ``reason == 'final_answer'``.
    3. ``HookRegistry.on_error`` received zero hook failures on the happy
       path -- a hook that raises on benign output would silently corrupt
       the model channel; surfacing this here makes the regression
       observable at PR review (ADR-0004).
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    try:
        adapter = _StubAdapter()

        async def _drive() -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as session_id:
                await run_task(
                    "runner-loop-smoke",
                    harness_dir,
                    logger,
                    session_id,
                    model_adapter=adapter,
                )

        asyncio.run(_drive())

        # --- Trace event vocabulary ----------------------------------------
        logger = TraceLogger(db)
        events = logger.load_session(logger.list_sessions()[0].session_id)
        kinds = [event.kind for event in events]

        # Every documented event kind from the agent loop (ADR-0010) appears.
        required_kinds = (
            "user_prompt",
            "model_request",
            "model_response",
            "tool_call",
            "tool_result",
            "outcome",
        )
        for required in required_kinds:
            assert required in kinds, f"event kind {required!r} missing from trace; got {kinds!r}"

        # user_prompt precedes outcome (the loop enters, then terminates).
        assert kinds.index("user_prompt") < kinds.index("outcome"), (
            f"user_prompt must precede outcome; got {kinds!r}"
        )

        # tool_call precedes its corresponding tool_result (the loop records
        # the call before the executor returns).
        assert kinds.index("tool_call") < kinds.index("tool_result"), (
            f"tool_call must precede tool_result; got {kinds!r}"
        )

        # --- Terminal outcome ----------------------------------------------
        outcome_event = next(event for event in events if event.kind == "outcome")
        assert outcome_event.payload["reason"] == "final_answer", (
            f"expected outcome.reason='final_answer'; got {outcome_event.payload!r}"
        )
        assert outcome_event.payload["status"] == "success"
        assert outcome_event.payload["steps"] >= 1

        # --- Hook isolation -------------------------------------------------
        # A benign happy path must not trigger HookRegistry._isolate_failure
        # (issue #21) -- a hook that raises on clean output would silently
        # corrupt the model channel. Surfacing the count here makes that
        # class of regression observable at PR review (ADR-0004).
        assert hook_failures == [], (
            f"expected zero HookRegistry.on_error calls on the happy path; got {hook_failures!r}"
        )

        # --- Adapter exhaustion guard --------------------------------------
        # Regression vs. SECURITY.md "Runaway detection": the loop must exit
        # after the second ``complete()`` call (final answer), not keep
        # invoking the adapter.
        assert adapter.calls == 2, (
            f"expected exactly 2 model round-trips (tool_call then "
            f"final_answer); got {adapter.calls}"
        )
    finally:
        # Drop the test's hooks + tracker so subsequent tests see a clean
        # default registry (no leaked firewall state, no leaked tracker).
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()
