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

    def reset(self, on_error: HookErrorCallback | None = None) -> None:
        """Drop every registered hook and replace ``on_error`` with ``on_error``.

        Lets a Critic sandbox or test wipe a registry back to a known-empty
        state without having to construct a fresh instance — useful when
        the caller already holds a reference and downstream code uses it
        by identity (e.g. the runner's ``get_registry()`` return value).
        The default ``None`` clears any previously bound sink as well; that
        is intentional — ``reset`` is the nuclear option.
        """
        self._hooks.clear()
        self._on_error = on_error

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


# ---------------------------------------------------------------------------
# Module-level default registry
# ---------------------------------------------------------------------------
# A single ``HookRegistry`` instance backs the convenience helpers below so
# existing call sites (``register_hook(h)``, ``get_registry()``) keep working
# unchanged. The Critic sandbox (ADR-0004) needs to evaluate harness
# variants in isolation — hooks from variant A must not leak into variant B's
# evaluation. The reset / swap helpers beneath make that possible without
# forcing every consumer to thread a registry argument through their code.
_DEFAULT_REGISTRY: HookRegistry = HookRegistry()
# Backwards-compatible alias for tests that historically reached into the
# module global directly (see ``tests/test_hook_isolation.py``). New code
# should call :func:`get_registry` so the swap helpers below are honored.
_REGISTRY = _DEFAULT_REGISTRY


def get_registry() -> HookRegistry:
    """Return the process-default registry.

    Returns whatever the most recent :func:`set_default_registry` call
    installed (or the original singleton if no swap has happened). Existing
    code keeps working unchanged; the Critic sandbox calls
    ``set_default_registry`` once at setup time so ``get_registry()`` in
    downstream code transparently returns the sandbox-scoped registry.
    """
    return _DEFAULT_REGISTRY


def set_default_registry(registry: HookRegistry) -> HookRegistry:
    """Replace the process-default registry.

    Returns the *previous* default so callers can restore it (e.g. a Critic
    sandbox evaluation that wants to leave the host process untouched when it
    finishes). Pass an existing instance to share state, or a fresh
    ``HookRegistry()`` to start blank.
    """
    global _DEFAULT_REGISTRY
    previous = _DEFAULT_REGISTRY
    _DEFAULT_REGISTRY = registry
    return previous


def reset_default_registry() -> HookRegistry:
    """Discard every hook from the default registry and return it.

    Convenience for tests: equivalent to ``set_default_registry(HookRegistry())``
    but reuses the existing instance so any caller that already holds a
    reference (via :func:`get_registry`) sees the cleared state immediately.
    """
    _DEFAULT_REGISTRY.reset()
    return _DEFAULT_REGISTRY


def register_hook(
    hook: Hook,
    *,
    registry: HookRegistry | None = None,
) -> Hook:
    """Register ``hook`` against ``registry`` (or the process default).

    The ``registry`` keyword is the Critic-sandbox entry point: pass a
    fresh ``HookRegistry`` to install the hook into an isolated chain that
    will not cross-contaminate the host process. Without ``registry`` the
    call is identical to the pre-issue behavior — hooks land on the
    module-level default, which every existing caller relies on.
    """
    target = registry if registry is not None else _DEFAULT_REGISTRY
    target.register(hook)
    return hook
