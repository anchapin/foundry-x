from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Protocol

_log = logging.getLogger("harness.hooks.base")


@dataclass
class ToolCall:
    name: str
    arguments: dict[str, Any]


@dataclass
class ToolResult:
    name: str
    output: Any
    error: str | None = None


class Hook(Protocol):
    async def pre_tool(self, call: ToolCall) -> ToolCall: ...

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult: ...


# Optional sink invoked when a hook raises inside ``run_pre`` / ``run_post``.
# The primary failure channel is the module logger (``harness.hooks.base``);
# this callback exists so the runner (or tests) can record the failure into
# the project TraceLogger or any other structured sink without coupling the
# harness hooks package to ``foundry_x``. The self-reference loop in
# AGENTS.md §7 forbids that import direction.
HookErrorCallback = Callable[[str, int, str, BaseException], None]


class HookRegistry:
    def __init__(self, on_error: HookErrorCallback | None = None) -> None:
        self._hooks: list[Hook] = []
        # ``on_error(slot, index, hook_name, exc)`` is invoked once per
        # isolated failure. Stored untyped on purpose so callers can route
        # the failure into any sink without widening the API.
        self._on_error = on_error

    def register(self, hook: Hook) -> None:
        self._hooks.append(hook)

    async def run_pre(self, call: ToolCall) -> ToolCall:
        for index, hook in enumerate(self._hooks):
            try:
                call = await hook.pre_tool(call)
            except Exception as exc:
                self._isolate_failure("pre_tool", index, hook, exc)
        return call

    async def run_post(self, call: ToolCall, result: ToolResult) -> ToolResult:
        for index, hook in enumerate(self._hooks):
            try:
                result = await hook.post_tool(call, result)
            except Exception as exc:
                self._isolate_failure("post_tool", index, hook, exc)
        return result

    def _isolate_failure(self, slot: str, index: int, hook: Hook, exc: Exception) -> None:
        """Log and route a single hook failure without aborting the chain.

        Implements the issue #21 isolation contract: a single buggy hook
        must not crash the entire agent run. The failure is (a) logged via
        the module logger with the hook name, slot, and exception, and
        (b) optionally forwarded to ``on_error`` for structured sinks such
        as the project TraceLogger. ``call`` / ``result`` are passed through
        unchanged so subsequent hooks still observe the original payload.

        AGENTS.md §2 forbids silently swallowing exceptions, so every
        branch either logs or re-raises. We catch ``Exception`` (not
        ``BaseException``) so ``asyncio.CancelledError``, ``KeyboardInterrupt``
        and ``SystemExit`` propagate normally — those are control-flow
        signals that must abort the run, not be hidden behind a failed hook.
        """
        hook_name = type(hook).__qualname__
        _log.exception(
            "hook %r (#%d) raised in %s; isolating and continuing: %r",
            hook_name,
            index,
            slot,
            exc,
        )
        if self._on_error is not None:
            try:
                self._on_error(slot, index, hook_name, exc)
            except Exception as sink_exc:  # pragma: no cover - defensive
                # A misbehaving sink must not undo the isolation. Surface the
                # sink's own failure through the same logger and move on.
                _log.exception(
                    "HookRegistry.on_error callback raised while reporting "
                    "%s failure of hook %r (#%d): %r",
                    slot,
                    hook_name,
                    index,
                    sink_exc,
                )


_REGISTRY = HookRegistry()


def register_hook(hook: Hook) -> Hook:
    _REGISTRY.register(hook)
    return hook


def get_registry() -> HookRegistry:
    return _REGISTRY
