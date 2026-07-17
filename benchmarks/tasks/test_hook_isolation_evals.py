"""Benchmark task: thrown hook exception does not abort the chain (ADR-0004).

Regression target for the hook-failure isolation contract in
``harness/hooks/base.py`` (issue #21). The single most important
property of the hook chain is that one buggy hook cannot take down the
whole agent run: ``HookRegistry.run_pre`` and ``run_post`` catch the
exception, log it through ``harness.hooks.base`` (no silent swallow
per AGENTS.md §2), optionally forward it to the ``on_error`` sink,
and pass the original ``ToolCall`` / ``ToolResult`` through unchanged
so subsequent hooks still observe the original payload. A regression
that turns the ``try/except`` into a re-raise, that swallows silently,
or that stops iterating after the first failure surfaces here as a
failing benchmark and blocks the harness edit at PR review (ADR-0004).
"""

from __future__ import annotations

import asyncio
import logging

import pytest

from benchmarks.models import BenchmarkTask
from harness.hooks.base import HookRegistry, ToolCall, ToolResult

_CALL = ToolCall(name="read_file", arguments={"path": "/tmp/x"})
_RESULT = ToolResult(name="read_file", output="hello")


class _RecordingHook:
    """A passthrough hook that records every call for ordering assertions."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.pre_calls: list[ToolCall] = []
        self.post_calls: list[tuple[ToolCall, ToolResult]] = []

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        self.pre_calls.append(call)
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        self.post_calls.append((call, result))
        return result


class _RaisingHook:
    """A hook that always raises ``RuntimeError`` in both slots."""

    def __init__(self, message: str = "boom") -> None:
        self.message = message

    async def pre_tool(self, call: ToolCall) -> ToolCall:  # noqa: ARG002
        raise RuntimeError(self.message)

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:  # noqa: ARG002
        raise RuntimeError(self.message)


def _run(coro):
    return asyncio.run(coro)


TASK = BenchmarkTask(
    name="hook_isolation",
    description=(
        "HookRegistry.run_pre and run_post isolate a thrown hook "
        "exception so the original call/result survives, every other "
        "hook still runs, and the failure is logged (never silently "
        "swallowed)."
    ),
    prompt=(
        "Inspect harness/hooks/base.py: confirm _isolate_failure still "
        "wraps the try/except, that an Exception (not BaseException) "
        "is caught, that the failure is forwarded to the optional "
        "on_error sink, and that subsequent hooks observe the original "
        "payload unchanged."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "run_pre and run_post return the original ToolCall / ToolResult "
        "by identity even when one registered hook raises; every other "
        "registered hook still runs in order; the failure is logged at "
        "ERROR level on the harness.hooks.base logger (no silent swallow)."
    ),
    tags=["security"],
)


@pytest.mark.benchmark
def test_pre_failure_does_not_abort_chain(caplog) -> None:
    """A raised ``pre_tool`` must not stop the chain or mutate the call."""
    before = _RecordingHook("before")
    bad = _RaisingHook()
    after = _RecordingHook("after")
    registry = HookRegistry()
    registry.register(before)
    registry.register(bad)
    registry.register(after)

    with caplog.at_level(logging.ERROR, logger="harness.hooks.base"):
        out = _run(registry.run_pre(_CALL))

    # Identity pass-through: the failing hook did not mutate the call.
    assert out is _CALL, "run_pre must pass the original call through unchanged"
    # Chain survived end-to-end.
    assert len(before.pre_calls) == 1, "hook before the failing hook ran exactly once"
    assert len(after.pre_calls) == 1, "hook after the failing hook MUST still run"
    # Failure is logged with the hook class name + slot (never silently swallowed).
    assert any(
        "RaisingHook" in record.message and "pre_tool" in record.message
        for record in caplog.records
    ), "the failure must be logged on harness.hooks.base with slot pre_tool"


@pytest.mark.benchmark
def test_post_failure_does_not_abort_chain(caplog) -> None:
    """A raised ``post_tool`` must not stop the chain or mutate the result."""
    before = _RecordingHook("before")
    bad = _RaisingHook()
    after = _RecordingHook("after")
    registry = HookRegistry()
    registry.register(before)
    registry.register(bad)
    registry.register(after)

    with caplog.at_level(logging.ERROR, logger="harness.hooks.base"):
        out = _run(registry.run_post(_CALL, _RESULT))

    assert out is _RESULT, "run_post must pass the original result through unchanged"
    assert len(before.post_calls) == 1, "hook before the failing hook ran exactly once"
    assert len(after.post_calls) == 1, "hook after the failing hook MUST still run"
    assert any(
        "RaisingHook" in record.message and "post_tool" in record.message
        for record in caplog.records
    ), "the failure must be logged on harness.hooks.base with slot post_tool"


@pytest.mark.benchmark
def test_post_failure_preserves_subsequent_hooks() -> None:
    """A ``post_tool`` exception in slot 1 must not stop slot 2 from running.

    The injection firewall sits at post_tool. If it raises, subsequent hooks
    (e.g. audit logging, result caching) must still observe the original result.
    """
    slot1 = _RaisingHook()
    slot2 = _RecordingHook("after")
    registry = HookRegistry()
    registry.register(slot1)
    registry.register(slot2)

    out = _run(registry.run_post(_CALL, _RESULT))

    assert out is _RESULT, "run_post must return the original result by identity"
    assert len(slot2.post_calls) == 1, "hook in slot 2 must still run after slot 1 raises"


@pytest.mark.benchmark
def test_post_failure_original_result_passed(caplog) -> None:
    """A ``post_tool`` exception must not mutate the result returned to the caller.

    The security-critical property of the injection firewall's post_tool slot
    is that a raised exception leaves the original ToolResult untouched —
    subsequent hooks and the agent runner see the original payload.
    """
    bad = _RaisingHook()
    registry = HookRegistry()
    registry.register(bad)

    with caplog.at_level(logging.ERROR, logger="harness.hooks.base"):
        out = _run(registry.run_post(_CALL, _RESULT))

    assert out is _RESULT, "run_post must pass the original result through unchanged by identity"
    assert any(
        "RaisingHook" in record.message and "post_tool" in record.message
        for record in caplog.records
    ), "the failure must be logged on harness.hooks.base with slot post_tool"


@pytest.mark.benchmark
def test_failure_is_forwarded_to_on_error_sink() -> None:
    """A registered ``on_error`` sink observes the isolated failure.

    The optional ``HookErrorCallback`` is the structured sink the runner
    uses to persist hook failures into the project ``TraceLogger``. A
    regression that swallows the failure before forwarding it (or that
    passes the wrong arguments) surfaces here as a wrong-shape payload
    or a missing call.
    """

    captured: list[tuple[str, int, str, BaseException]] = []

    def sink(slot: str, index: int, hook_name: str, exc: BaseException) -> None:
        captured.append((slot, index, hook_name, exc))

    registry = HookRegistry(on_error=sink)
    registry.register(_RecordingHook("before"))
    registry.register(_RaisingHook())
    registry.register(_RecordingHook("after"))

    _run(registry.run_pre(_CALL))

    assert len(captured) == 1, f"on_error sink must be invoked exactly once; got {len(captured)}"
    slot, index, hook_name, exc = captured[0]
    assert slot == "pre_tool"
    assert index == 1, "the failing hook's index (1) must be reported"
    assert hook_name == "_RaisingHook"
    assert isinstance(exc, RuntimeError)
