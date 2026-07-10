from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


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


class HookRegistry:
    def __init__(self) -> None:
        self._hooks: list[Hook] = []

    def register(self, hook: Hook) -> None:
        self._hooks.append(hook)

    async def run_pre(self, call: ToolCall) -> ToolCall:
        for hook in self._hooks:
            call = await hook.pre_tool(call)
        return call

    async def run_post(self, call: ToolCall, result: ToolResult) -> ToolResult:
        for hook in self._hooks:
            result = await hook.post_tool(call, result)
        return result


_REGISTRY = HookRegistry()


def register_hook(hook: Hook) -> Hook:
    _REGISTRY.register(hook)
    return hook


def get_registry() -> HookRegistry:
    return _REGISTRY
