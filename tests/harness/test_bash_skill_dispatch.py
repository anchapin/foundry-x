"""Dispatch test for the ``bash`` skill via ``HookRegistry`` (issue #104).

Issue #104 acceptance: a new test under ``tests/harness/`` invokes the
bash skill via ``HookRegistry`` fan-out, mocked against an empty
``HarnessStub``, and asserts the registry dispatches it without raising.

This is the smallest credible wiring: ``HookRegistry.run_pre`` /
``run_post`` walk the registered hooks, so a hook that recognises a
``ToolCall(name="bash", ...)`` and ``pass``-throughs everything else is
enough to prove the registry can route a bash call without a real
subprocess. The empty ``HarnessStub`` stands in for whatever the runner
later binds at startup; the test only cares that the call lands on the
chain.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from harness.hooks.base import HookRegistry, ToolCall, ToolResult


REPO_ROOT = Path(__file__).resolve().parents[2]
BASH_SKILL_PATH = REPO_ROOT / "harness" / "skills" / "bash.json"


class HarnessStub:
    """Empty harness stand-in for dispatch tests (issue #104).

    Holds no hooks and exposes no surface; exists only so the test can
    assert that the ``HookRegistry`` is the layer that fans a
    ``ToolCall`` out to registered hooks regardless of the harness
    backing it. The class deliberately has no behaviour so a future
    edit cannot accidentally add a hook here and silently change the
    test's coverage.
    """


class BashDispatchHook:
    """Records every ``bash`` ``ToolCall`` the registry routes to it.

    Implements the ``Hook`` protocol (pre/post) but is *not* itself a
    skill implementation -- it only proves the registry can deliver a
    bash-named ``ToolCall`` to a hook. The real bash skill implementation
    is the future ``subprocess.run``-backed hook that the runner will
    bind at startup; that work is out of scope for the seeding change
    in issue #104 (the skill JSON lands first; the executor lands in a
    later proposal so the Critic can evaluate it independently).
    """

    def __init__(self) -> None:
        self.pre_calls: list[ToolCall] = []
        self.post_calls: list[tuple[ToolCall, ToolResult]] = []

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        if call.name == "bash":
            self.pre_calls.append(call)
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        if call.name == "bash":
            self.post_calls.append((call, result))
        return result


def _run(coro):
    return asyncio.run(coro)


def test_bash_skill_json_is_loadable() -> None:
    doc = json.loads(BASH_SKILL_PATH.read_text(encoding="utf-8"))
    assert doc["name"] == "bash"


def test_registry_dispatches_bash_call_without_raising() -> None:
    """A ``bash`` ``ToolCall`` flows through ``HookRegistry`` cleanly.

    The empty ``HarnessStub`` proves the test is exercising the registry
    surface in isolation; ``BashDispatchHook`` proves a registered hook
    can observe the call. If either layer raises during ``run_pre`` or
    ``run_post`` the test fails (AGENTS.md \u00a72: never silently swallow).
    """
    HarnessStub()

    registry = HookRegistry()
    hook = BashDispatchHook()
    registry.register(hook)

    call = ToolCall(name="bash", arguments={"command": "echo hello", "cwd": "/tmp"})
    out_call = _run(registry.run_pre(call))

    assert out_call is call
    assert hook.pre_calls == [call], (
        "HookRegistry.run_pre must deliver the bash ToolCall to the "
        "registered hook (issue #104 acceptance)"
    )

    result = ToolResult(
        name="bash",
        output={"stdout": "hello\n", "stderr": "", "exit_code": 0, "truncated": False},
    )
    out_result = _run(registry.run_post(call, result))

    assert out_result is result
    assert hook.post_calls == [(call, result)], (
        "HookRegistry.run_post must deliver the bash ToolCall+result pair "
        "to the registered hook (issue #104 acceptance)"
    )


def test_registry_dispatch_is_a_noop_for_unknown_tools() -> None:
    """The registry must ignore tools it does not recognise.

    Companion to ``test_registry_dispatches_bash_call_without_raising``
    that pins the negative side: a non-bash tool call must not surface
    inside ``BashDispatchHook``. Guards against a future hook that
    forwards every call regardless of ``call.name``.
    """
    HarnessStub()

    registry = HookRegistry()
    hook = BashDispatchHook()
    registry.register(hook)

    other_call = ToolCall(name="read_file", arguments={"path": "/tmp/x"})
    _run(registry.run_pre(other_call))

    assert hook.pre_calls == [], "non-bash calls must not surface in the bash dispatch hook"
