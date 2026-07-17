"""Tests for RateLimitHook and rate-limit enforcement in Evolver.propose() (issue #332).

Acceptance criterion: "Evolver.propose() respects a configurable max_diffs_per_hour
and max_lines_per_diff cap and returns an empty list when the cap is breached;
a tests/evolution/test_rate_limit_hook.py covers the rejection path."

This module tests both:
1. ``RateLimitHook`` — the ``Hook`` implementation in ``harness/hooks/rate_limit.py``
   that tracks evolver calls and enforces the per-hour cap.
2. ``Evolver.propose()`` — returns ``[]`` when either cap is breached.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from harness.hooks.base import ToolCall, ToolResult
from harness.hooks.rate_limit import (
    DEFAULT_MAX_PROPOSALS_PER_HOUR,
    RateLimitHook,
    _RL_STATE,
    _get_window,
    register_into,
)


def _reset_state() -> None:
    """Reset the module-level rate-limit state between tests."""
    _RL_STATE["window"] = None


def _make_diff(*lines: str) -> str:
    header = "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n"
    hunk = "@@ -0,0 +1 @@\n"
    return header + hunk + "".join(f"+{line}\n" for line in lines)


class TestRateLimitHook:
    """Unit tests for ``RateLimitHook``."""

    def setup_method(self) -> None:
        _reset_state()

    def teardown_method(self) -> None:
        _reset_state()

    def test_hook_has_pre_and_post_tool(self) -> None:
        hook = RateLimitHook()
        assert hasattr(hook, "pre_tool")
        assert hasattr(hook, "post_tool")

    def test_pre_tool_accepts_first_call(self) -> None:
        hook = RateLimitHook()
        call = ToolCall(name="evolver_propose", arguments={})
        result = asyncio.run(hook.pre_tool(call))
        assert result is call
        window = _get_window()
        assert len(window) == 1
        ts, allowed = window[0]
        assert allowed is True

    def test_pre_tool_rejects_after_cap(self) -> None:
        hook = RateLimitHook()
        call = ToolCall(name="evolver_propose", arguments={})
        for _ in range(DEFAULT_MAX_PROPOSALS_PER_HOUR):
            asyncio.run(hook.pre_tool(call))
        with pytest.raises(RuntimeError, match="cap reached"):
            asyncio.run(hook.pre_tool(call))

    def test_pre_tool_ignores_non_evolver_calls(self) -> None:
        hook = RateLimitHook()
        call = ToolCall(name="read_file", arguments={"path": "/tmp/x"})
        result = asyncio.run(hook.pre_tool(call))
        assert result is call
        assert len(_get_window()) == 0

    def test_post_tool_decruments_pending_on_success(self) -> None:
        hook = RateLimitHook()
        call = ToolCall(name="evolver_propose", arguments={})
        asyncio.run(hook.pre_tool(call))
        result = ToolResult(
            name="evolver_propose",
            output=[
                ProposedEdit(
                    target_file="harness/system_prompt.txt",
                    rationale="test",
                    unified_diff=_make_diff("new line"),
                )
            ],
        )
        asyncio.run(hook.post_tool(call, result))
        assert len(_get_window()) == 0

    def test_post_tool_decruments_pending_on_empty_output(self) -> None:
        hook = RateLimitHook()
        call = ToolCall(name="evolver_propose", arguments={})
        asyncio.run(hook.pre_tool(call))
        result = ToolResult(name="evolver_propose", output=[])
        asyncio.run(hook.post_tool(call, result))
        assert len(_get_window()) == 0

    def test_post_tool_ignores_non_evolver_calls(self) -> None:
        hook = RateLimitHook()
        call = ToolCall(name="read_file", arguments={"path": "/tmp/x"})
        result = ToolResult(name="read_file", output="content")
        asyncio.run(hook.post_tool(call, result))
        assert len(_get_window()) == 0

    def test_register_into_returns_hook(self) -> None:
        from harness.hooks.base import HookRegistry

        registry = HookRegistry()
        hook = register_into(registry)
        assert isinstance(hook, RateLimitHook)
        assert hook in registry._hooks


class TestEvolverProposeReturnsEmptyOnCapBreach:
    """Test that ``Evolver.propose()`` returns ``[]`` when caps are breached.

    Acceptance criterion: "Evolver.propose() respects a configurable max_diffs_per_hour
    and max_lines_per_diff cap and returns an empty list when the cap is breached."
    """

    def test_propose_returns_empty_list_on_rate_limit_breach(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("original\n", encoding="utf-8")
        e = Evolver(max_proposals_per_hour=1, max_diff_lines=200)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        e.propose(harness_dir, failure=failure)
        result = e.propose(harness_dir, failure=failure)
        assert result == []

    def test_propose_returns_empty_list_on_diff_line_cap_breach(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("original\n", encoding="utf-8")
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=3)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        result = e.propose(harness_dir, failure=failure)
        assert result == []

    def test_propose_returns_edit_when_within_cap(self, tmp_path: Path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("original\n", encoding="utf-8")
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        result = e.propose(harness_dir, failure=failure)
        assert len(result) == 1
        assert isinstance(result[0], ProposedEdit)
