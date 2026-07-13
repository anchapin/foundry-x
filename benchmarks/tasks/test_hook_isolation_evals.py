"""Benchmark task: hook isolation for exceptions and argument modifications (issues #21, #343).

This module covers two isolation properties of ``HookRegistry``:

1. **Exception isolation** (issue #21, ADR-0004): a hook that raises must not
   abort the chain or mutate the ``ToolCall`` / ``ToolResult``. The failure is
   logged and forwarded to the optional ``on_error`` sink; subsequent hooks
   still run and observe the original payload.

2. **Argument modification isolation** (issue #343): a ``pre_tool`` hook that
   modifies ``call.arguments`` must propagate the modified ``ToolCall`` to
   subsequent hooks and to the tool executor. The agent's output must be
   consistent with the hook-modified state, and assertions must validate that
   the hook was called and the modification occurred.

A regression that breaks either isolation property surfaces here as a failing
benchmark and blocks the harness edit at PR review (ADR-0004).
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

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


class _ArgumentModifyingHook:
    """A hook that modifies ``call.arguments`` before passing them on.

    This exercises the argument-mutation path of the hook chain (issue #343):
    unlike ``_RaisingHook`` which tests exception isolation, this hook verifies
    that argument modifications made in ``pre_tool`` are propagated correctly
    to the tool executor and reflected in subsequent agent behavior.
    """

    def __init__(self, swaps: dict[str, str] | None = None) -> None:
        self.swaps = swaps or {}
        self.pre_calls: list[ToolCall] = []
        self.post_calls: list[tuple[ToolCall, ToolResult]] = []

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        self.pre_calls.append(call)
        modified = ToolCall(name=call.name, arguments=dict(call.arguments))
        for old_val, new_val in self.swaps.items():
            if old_val in modified.arguments:
                modified.arguments[new_val] = modified.arguments.pop(old_val)
            else:
                for key, value in modified.arguments.items():
                    if isinstance(value, str) and old_val in value:
                        modified.arguments[key] = value.replace(old_val, new_val)
        return modified

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        self.post_calls.append((call, result))
        return result


class _ReadFileSimulator:
    """Simulates a read_file tool for deterministic testing."""

    def __init__(self, workspace: dict[str, str]) -> None:
        self.workspace = workspace
        self.calls: list[ToolCall] = []

    async def execute(self, call: ToolCall) -> ToolResult:
        self.calls.append(call)
        path = call.arguments.get("path", "")
        content = self.workspace.get(path, f"file not found: {path}")
        return ToolResult(name=call.name, output=content)


@pytest.mark.benchmark
def test_pre_tool_modifies_arguments() -> None:
    """A ``pre_tool`` hook that modifies ``call.arguments`` propagates the modified call.

    Regression target for issue #343: ``HookRegistry.run_pre`` must pass the
    hook-modified ``ToolCall`` (not the original) to subsequent hooks and to
    the tool executor. A regression that (a) saves the original call before
    running hooks and uses the original, (b) creates a copy but discards it,
    or (c) calls the hook but ignores the return value would all break this
    contract.
    """
    registry = HookRegistry()
    swap_hook = _ArgumentModifyingHook(swaps={"old_path": "new_path"})
    registry.register(swap_hook)

    original_call = ToolCall(name="read_file", arguments={"old_path": "/tmp/secret.txt"})
    result_call = _run(registry.run_pre(original_call))

    assert result_call is not original_call, "run_pre must return the hook-modified call"
    assert (
        "new_path" in result_call.arguments
    ), f"hook must rename 'old_path' key to 'new_path'; got arguments={result_call.arguments}"
    assert "old_path" not in result_call.arguments, "original key must be removed after swap"
    assert len(swap_hook.pre_calls) == 1, "hook must be called exactly once"


@pytest.mark.benchmark
def test_argument_modification_affects_tool_execution() -> None:
    """When a pre-tool hook modifies arguments, the tool executor sees the modified values.

    This is the core issue #343 acceptance criterion: a hook that changes the
    path argument from ``secret.txt`` to ``decoy.txt`` must cause the tool to
    read ``decoy.txt`` instead of ``secret.txt``. The agent's subsequent behavior
    is only consistent with the hook-modified state if the tool actually received
    the modified arguments.
    """
    workspace_contents = {
        "/tmp/secret.txt": "CONFIDENTIAL DATA",
        "/tmp/decoy.txt": "PUBLIC INFORMATION",
    }

    swap_hook = _ArgumentModifyingHook(swaps={"secret.txt": "decoy.txt"})
    simulator = _ReadFileSimulator(workspace_contents)
    registry = HookRegistry()
    registry.register(swap_hook)

    original_call = ToolCall(name="read_file", arguments={"path": "/tmp/secret.txt"})

    modified_call = _run(registry.run_pre(original_call))
    result = _run(simulator.execute(modified_call))

    assert result.output == "PUBLIC INFORMATION", (
        "tool must read decoy.txt (modified path) not secret.txt (original path); "
        f"got output={result.output!r}"
    )
    assert (
        simulator.calls[0].arguments.get("path") == "/tmp/decoy.txt"
    ), "tool executor must receive the hook-modified path"


@pytest.mark.benchmark
def test_argument_modification_preserves_chain_for_subsequent_hooks() -> None:
    """Argument modification by an intermediate hook does not break subsequent hooks.

    The hook chain must survive an argument-modifying hook just as it survives
    a raising hook: subsequent hooks in the chain still run and observe the
    (now modified) call that was passed through.
    """
    before = _RecordingHook("before")
    modifier = _ArgumentModifyingHook(swaps={"original_key": "modified_key"})
    after = _RecordingHook("after")
    registry = HookRegistry()
    registry.register(before)
    registry.register(modifier)
    registry.register(after)

    original_call = ToolCall(name="test_tool", arguments={"original_key": "value"})
    final_call = _run(registry.run_pre(original_call))

    assert (
        final_call.arguments.get("modified_key") == "value"
    ), "final call must carry the modified key from the modifier hook"
    assert len(before.pre_calls) == 1, "hook before modifier must have run"
    assert len(modifier.pre_calls) == 1, "modifier hook must have run"
    assert len(after.pre_calls) == 1, "hook after modifier must have run"
    assert (
        after.pre_calls[0].arguments.get("modified_key") == "value"
    ), "subsequent hook must observe the modified call, not the original"


@pytest.mark.benchmark
def test_hook_modification_recorded_in_trace(benchmark_workspace: Path) -> None:
    """When a pre-tool hook modifies arguments, the modification is visible in trace events.

    The ``benchmark_workspace`` fixture is seeded with ``secret.txt`` and ``decoy.txt``
    so the test can run a realistic scenario: the agent is asked to read ``secret.txt``
    but the hook redirects to ``decoy.txt``. The trace must record the modified
    ``path`` argument so the human-review channel can audit what the hook did.
    """
    workspace_contents = {
        "secret.txt": "CONFIDENTIAL DATA",
        "decoy.txt": "PUBLIC INFORMATION",
    }
    for filename, content in workspace_contents.items():
        (benchmark_workspace / filename).write_text(content, encoding="utf-8")

    swap_hook = _ArgumentModifyingHook(swaps={"secret.txt": "decoy.txt"})
    simulator = _ReadFileSimulator(
        {str(benchmark_workspace / k): v for k, v in workspace_contents.items()}
    )
    registry = HookRegistry()
    registry.register(swap_hook)

    original_call = ToolCall(
        name="read_file",
        arguments={"path": str(benchmark_workspace / "secret.txt")},
    )

    modified_call = _run(registry.run_pre(original_call))
    result = _run(simulator.execute(modified_call))

    assert result.output == "PUBLIC INFORMATION", "hook must redirect from secret.txt to decoy.txt"
    assert swap_hook.pre_calls[0].arguments.get("path") == str(
        benchmark_workspace / "secret.txt"
    ), "hook must have received the original path"
    assert modified_call.arguments.get("path") == str(
        benchmark_workspace / "decoy.txt"
    ), "hook must have returned the modified path"
