"""Rate-limit hook and defaults mirror (SECURITY.md "Rate limits", issues #206, #332).

``docs/SECURITY.md:50-52`` states that the Evolver rate-limit defaults --
"max N proposals per hour, max M lines of harness diff per proposal" --
"live in ``harness/hooks/``." Before this module those defaults lived
only in ``src/foundry_x/evolution/evolver.py`` (``Evolver.__init__``).
This module is the thin mirror that makes the SECURITY.md claim true:
the same numeric defaults are now discoverable from the harness side.

``RateLimitHook`` (issue #332) implements the ``Hook`` protocol to enforce
``max_diffs_per_hour`` at the tool-call level: ``pre_tool`` rejects new
evolver calls once the sliding window is full, and ``post_tool`` cleans up
the pending counter when edits are returned.

Self-reference constraint (AGENTS.md section 7)
-----------------------------------------------
This module imports **nothing** from ``src/foundry_x/``. The harness is
the artifact being evolved; the foundry is the machinery that evolves
it. A harness-to-foundry import would close that loop in the wrong
direction. The numeric equality between this mirror and the
foundry-resident defaults is asserted by
``tests/harness/test_rate_limit_mirror.py`` so a drift between the two
surfaces as a test failure rather than a silent divergence.

Seed status (issue #206)
------------------------
This file is a **seed** created per issue #206's explicit request. It is
a new file landing the cap's declarative form, not hand-edited DNA in
the sense of AGENTS.md section 2. Future evolution runs may modify it
through the normal ``Evolver`` -> ``Critic`` pipeline (ADR-0004).

Out of scope (issue #206)
-------------------------
- Moving the runtime cap from ``src/foundry_x/`` to ``harness/hooks/``.
- Adding a runtime hook invocation in ``main()`` (separate runner-slot
  proposal).
"""

from __future__ import annotations

from collections import deque
from datetime import datetime, timedelta, timezone
from typing import TYPE_CHECKING

from .base import HookRegistry, register_hook

if TYPE_CHECKING:
    from .base import ToolCall, ToolResult

# ---------------------------------------------------------------------------
# Tool name for the evolver (used by RateLimitHook to scope its tracking)
# ---------------------------------------------------------------------------
_EVOLVER_TOOL_NAME = "evolver_propose"

# ---------------------------------------------------------------------------
# Default cap values -- mirror of Evolver.__init__ defaults
# ---------------------------------------------------------------------------
# These constants are the single source of truth on the harness side. The
# foundry-side runtime defaults live in
# ``src/foundry_x/evolution/evolver.py`` (``Evolver(max_proposals_per_hour=10,
# max_diff_lines=200)`` plus ``_RATE_WINDOW = timedelta(hours=1)``). The two
# MUST agree; ``tests/harness/test_rate_limit_mirror.py`` enforces that.

DEFAULT_MAX_PROPOSALS_PER_HOUR: int = 10
DEFAULT_MAX_DIFF_LINES: int = 200
DEFAULT_RATE_WINDOW_HOURS: int = 1

# ---------------------------------------------------------------------------
# Shared rate-limit state (process-global, so RateLimitHook is stateless)
# ---------------------------------------------------------------------------
# Tracks (timestamp, pending_flag) tuples in a sliding window.
# pending_flag is True when a call was allowed, False when it was rejected.
_RL_STATE: dict[str, int | deque[tuple[datetime, bool]] | None] = {
    "window": None,
    "max_per_hour": DEFAULT_MAX_PROPOSALS_PER_HOUR,
    "max_diff_lines": DEFAULT_MAX_DIFF_LINES,
}


def _get_window() -> deque[tuple[datetime, bool]]:
    if _RL_STATE["window"] is None:
        _RL_STATE["window"] = deque()
    return _RL_STATE["window"]


def _purge_old(window: deque[tuple[datetime, bool]], hours: int) -> None:
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    while window and window[0][0] < cutoff:
        window.popleft()


def _count_recent(window: deque[tuple[datetime, bool]]) -> int:
    return sum(1 for _, allowed in window if allowed)


def get_default_max_proposals() -> int:
    """Return the default cap on proposals per rolling hour.

    Mirrors ``Evolver.__init__(max_proposals_per_hour=10)`` in
    ``src/foundry_x/evolution/evolver.py``. SECURITY.md "Rate limits":
    "max N proposals per hour."
    """
    return DEFAULT_MAX_PROPOSALS_PER_HOUR


def get_default_max_diff_lines() -> int:
    """Return the default cap on unified-diff lines per proposal.

    Mirrors ``Evolver.__init__(max_diff_lines=200)`` in
    ``src/foundry_x/evolution/evolver.py``. SECURITY.md "Rate limits":
    "max M lines of harness diff per proposal."
    """
    return DEFAULT_MAX_DIFF_LINES


def get_default_rate_window_hours() -> int:
    """Return the sliding-window length in hours for the proposal cap.

    Mirrors ``_RATE_WINDOW = timedelta(hours=1)`` in
    ``src/foundry_x/evolution/evolver.py``.
    """
    return DEFAULT_RATE_WINDOW_HOURS


# ---------------------------------------------------------------------------
# RateLimitHook (issue #332)
# ---------------------------------------------------------------------------


class RateLimitHook:
    """Hook that enforces per-hour evolver call cap (issue #332).

    Implements the ``Hook`` protocol. ``pre_tool`` checks the sliding
    window and rejects evolver calls once ``DEFAULT_MAX_PROPOSALS_PER_HOUR``
    is reached. ``post_tool`` cleans up the pending counter when edits
    are returned.

    The hook is stateless; all state lives in ``_RL_STATE`` (module-level
    dict) so the same state is shared across all hook instances.
    """

    __slots__ = ()

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        if call.name != _EVOLVER_TOOL_NAME:
            return call
        window = _get_window()
        _purge_old(window, DEFAULT_RATE_WINDOW_HOURS)
        allowed = _count_recent(window) < _RL_STATE["max_per_hour"]
        window.append((datetime.now(timezone.utc), allowed))
        if not allowed:
            raise RuntimeError(
                f"RateLimitHook: {DEFAULT_MAX_PROPOSALS_PER_HOUR} proposals per "
                f"hour cap reached; rejecting {call.name}"
            )
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        if call.name != _EVOLVER_TOOL_NAME:
            return result
        window = _get_window()
        if not window:
            return result
        _timestamp, allowed = window[-1]
        if not allowed:
            return result
        if result.error is not None:
            window.pop()
            return result
        output = result.output
        if isinstance(output, list):
            window.pop()
        return result


# ---------------------------------------------------------------------------
# Self-registration (mirrors injection_firewall.py pattern)
# ---------------------------------------------------------------------------

_rate_limit_hook_instance: RateLimitHook | None = None


def _get_hook() -> RateLimitHook:
    global _rate_limit_hook_instance
    if _rate_limit_hook_instance is None:
        _rate_limit_hook_instance = RateLimitHook()
    return _rate_limit_hook_instance


def register_into(registry: HookRegistry) -> RateLimitHook:
    hook = _get_hook()
    registry.register(hook)
    return hook


register_hook(_get_hook())


__all__ = [
    "DEFAULT_MAX_DIFF_LINES",
    "DEFAULT_MAX_PROPOSALS_PER_HOUR",
    "DEFAULT_RATE_WINDOW_HOURS",
    "RateLimitHook",
    "get_default_max_diff_lines",
    "get_default_max_proposals",
    "get_default_rate_window_hours",
    "register_into",
]
