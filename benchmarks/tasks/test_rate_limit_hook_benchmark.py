"""Benchmark task: RateLimitHook enforces the per-hour evolver cap (issue #874).

Regression target for ``harness.hooks.rate_limit.RateLimitHook`` (issues #206
and #332). The hook is the harness-side mirror of the Evolver's
``max_proposals_per_hour`` cap and is wired into the default ``HookRegistry``
via ``register_hook(_get_hook())`` at import time. Its three contracts the
benchmark pins are:

1. ``pre_tool`` accepts the first ``DEFAULT_MAX_PROPOSALS_PER_HOUR`` calls
   with ``call.name == "evolver_propose"`` and rejects the next one with
   ``RuntimeError("cap reached")``. The benchmark drives the calls through
   ``HookRegistry.run_pre`` so the assertion matches the production code
   path the Runner uses (``runner.py:1702``).
2. ``pre_tool`` short-circuits on any other tool name so benign work like
   ``read_file`` never consumes the cap. A regression that broadens the
   scope would silently throttle the agent loop.
3. ``post_tool`` decrements the sliding window when the evolver returns a
   list of edits, so the cap reflects ``inflight + recent``, not
   ``all-time attempts``.

The benchmark also measures ``pre_tool`` latency so a regression that turns
the hook into synchronous I/O surfaces as a wall-clock budget breach at PR
review (ADR-0004). Results are logged at INFO in the same shape other hook
benchmarks use for cross-run comparability.

The fixture directory ``benchmarks/fixtures/rate_limit_hook/`` exists so
``tests/benchmarks/test_hygiene.py::test_every_benchmark_task_has_matching_fixture_directory``
has a target; the benchmark reads no seed data.
"""

from __future__ import annotations

import asyncio
import logging
import time

import pytest

from benchmarks.models import BenchmarkTask
from harness.hooks.base import HookRegistry, ToolCall, ToolResult
from harness.hooks.rate_limit import (
    DEFAULT_MAX_PROPOSALS_PER_HOUR,
    DEFAULT_RATE_WINDOW_HOURS,
    RateLimitHook,
    _RL_STATE,
    _get_window,
)

_log = logging.getLogger(__name__)

TASK = BenchmarkTask(
    name="rate_limit_hook",
    description=(
        "RateLimitHook enforces DEFAULT_MAX_PROPOSALS_PER_HOUR proposals "
        "per rolling hour on evolver_propose tool calls through the "
        "HookRegistry, ignores other tool names, releases the slot when "
        "post_tool sees an edit list, and runs each pre_tool call under "
        "a tight latency budget (issues #206, #332, #874)."
    ),
    prompt=(
        "Inspect harness/hooks/rate_limit.py: confirm RateLimitHook.pre_tool "
        "allows up to DEFAULT_MAX_PROPOSALS_PER_HOUR evolver_propose calls "
        "per DEFAULT_RATE_WINDOW_HOURS-hour rolling window, raises "
        "RuntimeError on the next one, leaves non-evolver_propose calls "
        "untouched, and that post_tool releases the slot when the result "
        "carries a list of edits."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "First N evolver_propose pre_tool calls succeed (N=DEFAULT_MAX_PROPOSALS_PER_HOUR); "
        "(N+1)th raises RuntimeError matching 'cap reached'; non-evolver_propose "
        "calls pass through identity and leave the sliding window empty; "
        "post_tool decrements the window when the result returns a list; "
        "pre_tool latency stays under 50 ms per call."
    ),
    tags=["security", "rate-limit", "harness-hook"],
)


def _reset_rl_state() -> None:
    """Reset the module-level rate-limit state between tests.

    ``_RL_STATE`` and the window deque are process-global so the same hook
    instance can observe state across calls. A test that forgets to reset
    would leak its window into the next benchmark and produce a flaky pass.
    """
    _RL_STATE["window"] = None
    _RL_STATE["max_per_hour"] = DEFAULT_MAX_PROPOSALS_PER_HOUR


@pytest.fixture(autouse=True)
def _clean_rl_state() -> None:
    """Auto-reset rate-limit state around every test in this module."""
    _reset_rl_state()
    yield
    _reset_rl_state()


def _evolver_call() -> ToolCall:
    return ToolCall(name="evolver_propose", arguments={})


def _edit_payload() -> list[dict[str, str]]:
    """Minimal synthetic edit list to feed ``post_tool``."""
    return [
        {
            "target_file": "harness/hooks/x.py",
            "rationale": "synthetic benchmark edit",
            "unified_diff": (
                "--- a/harness/hooks/x.py\n+++ b/harness/hooks/x.py\n@@ -0,0 +1 @@\n+synthetic\n"
            ),
        }
    ]


# --- benchmark tests ------------------------------------------------------


@pytest.mark.benchmark
def test_rate_limit_hook_cap_enforced_at_configured_threshold() -> None:
    """``DEFAULT_MAX_PROPOSALS_PER_HOUR`` calls succeed; the next one is rejected.

    This is the core SECURITY.md "max N proposals per hour" guarantee
    surfaced as a deterministic benchmark. The first N evolver_propose
    pre_tool calls must pass through identity (the hook records a slot in
    the sliding window for each) and the (N+1)th must raise
    ``RuntimeError("... cap reached ...")``.

    Driven through ``HookRegistry.run_pre`` so the assertion matches the
    production wiring (``runner.py:1702``). The hook-failure isolation
    contract (issue #21) means the registry forwards the RuntimeError to
    the ``on_error`` sink and returns the original ``ToolCall`` by
    identity; the benchmark asserts both halves of that contract.
    """
    registry = HookRegistry()
    registry.register(RateLimitHook())
    call = _evolver_call()

    for i in range(DEFAULT_MAX_PROPOSALS_PER_HOUR):
        out = asyncio.run(registry.run_pre(call))
        assert out is call, (
            f"call #{i + 1}/{DEFAULT_MAX_PROPOSALS_PER_HOUR}: hook must pass "
            f"the ToolCall through identity; got {out!r}"
        )

    assert len(_get_window()) == DEFAULT_MAX_PROPOSALS_PER_HOUR, (
        f"window must record exactly {DEFAULT_MAX_PROPOSALS_PER_HOUR} slots "
        f"after N successful calls; got {len(_get_window())}"
    )

    # The (N+1)th call trips the cap. Bind a sink so we can prove the
    # RuntimeError was emitted with the expected shape (the registry
    # catches it under the hook-isolation contract; it never reaches
    # ``runner.py`` callers as an exception).
    hook_failures: list[tuple[str, int, str, BaseException]] = []

    def _sink(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, exc))

    registry_with_sink = HookRegistry(on_error=_sink)
    registry_with_sink.register(RateLimitHook())

    out = asyncio.run(registry_with_sink.run_pre(call))

    assert out is call, (
        "the (N+1)th call must still be returned by identity (issue #21 "
        "hook-isolation contract); the failure must be routed through the "
        "on_error sink, not re-raised to the caller"
    )
    assert len(hook_failures) == 1, (
        f"the cap must trip exactly once; got {len(hook_failures)} failures: {hook_failures!r}"
    )
    slot, _index, hook_name, exc = hook_failures[0]
    assert slot == "pre_tool", f"failure slot must be pre_tool; got {slot!r}"
    assert hook_name == "RateLimitHook", (
        f"failure must identify the RateLimitHook; got {hook_name!r}"
    )
    assert isinstance(exc, RuntimeError), (
        f"cap rejection must be a RuntimeError; got {type(exc).__name__}"
    )
    assert "cap reached" in str(exc), (
        f"rejection message must mention 'cap reached' (issue #332); got {exc!r}"
    )
    assert str(DEFAULT_MAX_PROPOSALS_PER_HOUR) in str(exc), (
        f"rejection message must quote the configured cap ({DEFAULT_MAX_PROPOSALS_PER_HOUR}); "
        f"got {exc!r}"
    )


@pytest.mark.benchmark
def test_rate_limit_hook_ignores_non_evolver_calls() -> None:
    """Non-``evolver_propose`` tool calls bypass the hook entirely.

    ``RateLimitHook.pre_tool`` short-circuits on ``call.name != _EVOLVER_TOOL_NAME``
    so benign work (read_file, write_file, bash, etc.) does not consume
    the cap. A regression that broadens the scope would silently throttle
    every tool the agent uses; the benchmark pins the contract by
    submitting many non-evolver calls and asserting the window stays
    empty.
    """
    registry = HookRegistry()
    registry.register(RateLimitHook())
    call = ToolCall(name="read_file", arguments={"path": "/tmp/x"})

    # Well in excess of the cap so a regression that "almost" scopes
    # correctly still surfaces here.
    over_cap_calls = DEFAULT_MAX_PROPOSALS_PER_HOUR + 5
    for _ in range(over_cap_calls):
        out = asyncio.run(registry.run_pre(call))
        assert out is call, "non-evolver_propose calls must pass through identity"

    assert len(_get_window()) == 0, (
        f"sliding window must remain empty for non-evolver calls; "
        f"got {len(_get_window())} entries (a regression broadened the "
        f"scope beyond 'evolver_propose')"
    )


@pytest.mark.benchmark
def test_rate_limit_hook_post_tool_releases_slot_on_edit_return() -> None:
    """``post_tool`` pops the slot when the evolver returns a list of edits.

    The cap exists to bound the rate of *attempted* proposals, not the
    rate of *successful* edits. When the evolver returns a non-empty edit
    list, the corresponding pre_tool slot is consumed (the proposal
    happened) and post_tool pops it so the window reflects pending +
    recent, not all-time. An empty list also releases the slot (the
    proposal produced no edit, so it shouldn't count).
    """
    registry = HookRegistry()
    registry.register(RateLimitHook())

    call = _evolver_call()

    # Fill the cap exactly so a regression that forgets to pop leaves a
    # full window and the next test cannot run cleanly.
    for _ in range(DEFAULT_MAX_PROPOSALS_PER_HOUR):
        asyncio.run(registry.run_pre(call))
    assert len(_get_window()) == DEFAULT_MAX_PROPOSALS_PER_HOUR

    # post_tool with a non-empty edit list must release the slot.
    edit_result = ToolResult(name="evolver_propose", output=_edit_payload())
    out = asyncio.run(registry.run_post(call, edit_result))
    assert out is edit_result, "post_tool must pass the ToolResult through identity"
    assert len(_get_window()) == DEFAULT_MAX_PROPOSALS_PER_HOUR - 1, (
        f"post_tool must pop one slot when the result carries an edit list; "
        f"window expected size {DEFAULT_MAX_PROPOSALS_PER_HOUR - 1}, got "
        f"{len(_get_window())}"
    )

    # And the empty-list path also releases a slot.
    asyncio.run(registry.run_pre(call))
    assert len(_get_window()) == DEFAULT_MAX_PROPOSALS_PER_HOUR
    empty_result = ToolResult(name="evolver_propose", output=[])
    asyncio.run(registry.run_post(call, empty_result))
    assert len(_get_window()) == DEFAULT_MAX_PROPOSALS_PER_HOUR - 1, (
        "post_tool must pop one slot when the result is an empty list"
    )

    # Non-evolver calls do not interact with the window.
    non_evolver_result = ToolResult(name="read_file", output="content")
    asyncio.run(registry.run_post(ToolCall(name="read_file", arguments={}), non_evolver_result))
    assert len(_get_window()) == DEFAULT_MAX_PROPOSALS_PER_HOUR - 1, (
        "post_tool on a non-evolver call must leave the window untouched"
    )


@pytest.mark.benchmark
def test_rate_limit_hook_window_uses_configured_rolling_window() -> None:
    """The sliding-window length matches ``DEFAULT_RATE_WINDOW_HOURS``.

    SECURITY.md "max N proposals per hour" specifies a one-hour rolling
    window. A regression that tightens the window (e.g. minutes instead of
    hours) or widens it (e.g. daily) would diverge from the prose without
    the cap number itself changing; the benchmark pins the window length
    as a separate assertion.
    """
    assert DEFAULT_RATE_WINDOW_HOURS == 1, (
        f"DEFAULT_RATE_WINDOW_HOURS must remain 1 hour (SECURITY.md 'max N "
        f"proposals per hour'); got {DEFAULT_RATE_WINDOW_HOURS}"
    )

    registry = HookRegistry()
    registry.register(RateLimitHook())

    # One successful call records a slot; the slot's timestamp is what
    # _purge_old compares against the rolling cutoff. We don't synthesize
    # timestamps here (the hook owns its own clock); we just assert the
    # contract that the recorded slot is the only thing in the window
    # after a single call.
    asyncio.run(registry.run_pre(_evolver_call()))

    window = _get_window()
    assert len(window) == 1, f"window must contain exactly one slot; got {window!r}"
    _ts, allowed = window[0]
    assert allowed is True, (
        "the recorded slot must be flagged allowed=True; rejected calls "
        "must not append to the window (see issue #332 / "
        "tests/evolution/test_rate_limit_hook.py::test_rejected_call_not_appended_to_window)"
    )


@pytest.mark.benchmark
def test_rate_limit_hook_pre_tool_latency_under_budget() -> None:
    """Each ``pre_tool`` call stays under a 50 ms latency budget.

    The cap runs on the Runner's critical path
    (``runner.py:1700-1705``: ``hook_start = time.monotonic()``). A
    regression that introduces I/O or unbounded work inside the hook would
    silently degrade the agent loop. 50 ms is generous enough to absorb
    scheduling jitter on a busy CI box but tight enough to catch a
    regression that turns the hook into a sync DB write.

    The latency numbers are logged at INFO so the benchmark report carries
    comparable values across runs (matches the format used by
    ``test_evolver_guardrail_evals`` for cross-run diffing).
    """
    registry = HookRegistry()
    registry.register(RateLimitHook())
    call = _evolver_call()

    samples_ms: list[float] = []
    for _ in range(DEFAULT_MAX_PROPOSALS_PER_HOUR):
        start = time.perf_counter()
        asyncio.run(registry.run_pre(call))
        samples_ms.append((time.perf_counter() - start) * 1000.0)

    max_ms = max(samples_ms)
    mean_ms = sum(samples_ms) / len(samples_ms)
    _log.info(
        "rate_limit_hook pre_tool latency: n=%d mean=%.3fms max=%.3fms",
        len(samples_ms),
        mean_ms,
        max_ms,
    )

    assert max_ms < 50.0, (
        f"RateLimitHook.pre_tool latency must stay under 50 ms; got "
        f"max={max_ms:.3f}ms mean={mean_ms:.3f}ms across {len(samples_ms)} "
        f"calls. A regression that adds I/O to the hot path surfaces here."
    )
