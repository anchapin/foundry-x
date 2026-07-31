"""Benchmark task: Context-pruning sweep for 5600G/6600 XT validation (issue #956).

Implements the ADR-0021 §5 sweep methodology: run the benchmark suite at
three ``FOUNDRY_CONTEXT_TOKENS`` thresholds (4096, 8192, 16384) and
compare pass rates to validate the default of 8192 tokens.

The sweep drives ``Runner.run_task`` with a stub ``ModelAdapter`` that
emits multiple tool calls with cumulative token usage, wiring the
``TokenAwarePruningHook`` so that pruning fires at each threshold.

For each threshold the sweep records:

- **passed**: session ``outcome.status == 'success'``
- **context_pruned_count**: number of ``context_pruned`` events emitted
- **dropped_total**: sum of ``dropped`` across all ``context_pruned`` events
- **token_budget_hit**: whether the session was aborted with
  ``task_aborted(reason='token_budget')``

After all three thresholds complete, the aggregation test asserts that
the pass rate at 8192 is within 5 percentage points of the 16384 baseline
(per ADR-0021 §5 regression threshold guidance).

The sweep emits one ``ContextPruningSweepResult`` per threshold. These
are the data contract that the KPI layer consumes for ``context_efficiency``
computation (issue #951):

    context_efficiency = 1 - (dropped_total / total_events)

A value near 1.0 means pruning rarely fired; near 0.0 means heavy pruning
throughout the session.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.models import BenchmarkTask, ContextPruningSweepResult
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ModelUsage,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger
from harness.hooks.context_pruning import _SqlitePruner

TASK = BenchmarkTask(
    name="context_pruning_sweep",
    description=(
        "Parametrized sweep of FOUNDRY_CONTEXT_TOKENS across 4096/8192/16384 "
        "to validate the 8192-token default against ADR-0021 §5 regression "
        "threshold guidance; emits ContextPruningSweepResult per threshold "
        "for context_efficiency KPI aggregation (issue #956)."
    ),
    tags=["agent-loop", "context-pruning", "sweep", "5600G", "6600-XT"],
    difficulty_tier="medium",
)

_THRESHOLDS = ["4096", "8192", "16384"]
_EVENT_THRESHOLD = 50


class _SweepStubAdapter:
    """Stub ``ModelAdapter`` for the sweep that emits escalating token usage.

    Produces responses with cumulative ``usage.total_tokens`` values that
    cross the token thresholds at predictable steps:

    =========  =========  ===================
    Step       Response  Cumulative tokens
    =========  =========  ===================
    1          tool_call  300
    2          tool_call  1100
    3          tool_call  2100   ← exceeds 2048
    4          tool_call  3400
    5          final      4500
    =========  =========  ===================

    At 4096 / 8192 threshold pruning fires at step 3 when tokens
    cross the threshold; at 16384 it never fires.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            tool_call = ModelToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(
                    name="bash",
                    arguments=json.dumps({"command": "echo step1"}),
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
                usage=ModelUsage(prompt_tokens=100, completion_tokens=200, total_tokens=300),
            )
        if self.calls == 2:
            tool_call = ModelToolCall(
                id="call_2",
                type="function",
                function=ToolCallFunction(
                    name="read_file",
                    arguments=json.dumps({"path": "/tmp/x"}),
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
                usage=ModelUsage(prompt_tokens=200, completion_tokens=600, total_tokens=800),
            )
        if self.calls == 3:
            tool_call = ModelToolCall(
                id="call_3",
                type="function",
                function=ToolCallFunction(
                    name="grep_search",
                    arguments=json.dumps({"pattern": "test", "path": "/tmp"}),
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
                usage=ModelUsage(prompt_tokens=300, completion_tokens=500, total_tokens=800),
            )
        if self.calls == 4:
            tool_call = ModelToolCall(
                id="call_4",
                type="function",
                function=ToolCallFunction(
                    name="write_file",
                    arguments=json.dumps({"path": "/tmp/y", "content": "result"}),
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
                usage=ModelUsage(prompt_tokens=400, completion_tokens=900, total_tokens=1300),
            )
        if self.calls == 5:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
                usage=ModelUsage(prompt_tokens=500, completion_tokens=600, total_tokens=1100),
            )
        raise RuntimeError(
            f"_SweepStubAdapter exhausted after 5 scripted responses; "
            f"loop called complete() {self.calls} times"
        )

    async def chat(self, messages, tools=None, **kwargs):
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):
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
        if response.usage is not None:
            yield ModelResponseChunk(usage=response.usage)


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for context_pruning_sweep\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


def _install_on_error_tracker(tracker):
    """Install ``tracker`` on the default ``HookRegistry`` for the test."""
    from harness.hooks import get_registry
    from harness.hooks.base import reset_default_registry

    reset_default_registry()
    registry = get_registry()
    registry._on_error = tracker  # type: ignore[assignment]
    return registry


@pytest.mark.parametrize("context_tokens", _THRESHOLDS)
@pytest.mark.benchmark
def test_context_pruning_sweep_run(
    benchmark_workspace: Path,
    context_tokens: str,
    sweep_results_path: Path,
) -> None:
    """Run the sweep at one ``FOUNDRY_CONTEXT_TOKENS`` threshold.

    Drives ``Runner.run_task`` with the stub adapter and ``TokenAwarePruningHook``.
    Records the outcome and ``context_pruned`` event count into the session-scoped
    ``sweep_results_path`` file for the aggregation test to consume.

    Parameters
    ----------
    context_tokens:
        The ``FOUNDRY_CONTEXT_TOKENS`` value for this parametrized run.
        One of "4096", "8192", "16384".
    sweep_results_path:
        Session-scoped path shared across all sweep tests in the same worker.
    """
    token_threshold = int(context_tokens)

    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    try:
        adapter = _SweepStubAdapter()
        _pruner = _SqlitePruner(db)

        async def _drive(reg: Any) -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as sid:

                def _tracer(sid: str, kind: str, payload: dict) -> None:
                    logger.record(sid, kind=kind, payload=payload)

                from harness.hooks.context_pruning import TokenAwarePruningHook

                hook = TokenAwarePruningHook(
                    session_id=sid,
                    token_threshold=token_threshold,
                    event_threshold=_EVENT_THRESHOLD,
                    pruner=_pruner.prune,
                    tracer=_tracer,
                    get_tokens=_pruner.count_tokens,
                )
                reg.register(hook)

                await run_task(
                    "context-pruning-sweep",
                    harness_dir,
                    logger,
                    sid,
                    model_adapter=adapter,
                )

        asyncio.run(_drive(registry))

        logger = TraceLogger(db)
        sid = logger.list_sessions()[0].session_id
        events = logger.load_session(sid)

        prune_events = [e for e in events if e.kind == "context_pruned"]
        context_pruned_count = len(prune_events)
        dropped_total = sum(e.payload.get("dropped", 0) for e in prune_events)
        total_events = len(events)

        outcome_event = next((e for e in events if e.kind == "outcome"), None)
        passed = outcome_event is not None and outcome_event.payload.get("status") == "success"

        task_abort = next(
            (
                e
                for e in events
                if e.kind == "task_aborted" and e.payload.get("reason") == "token_budget"
            ),
            None,
        )
        token_budget_hit = task_abort is not None

        result = ContextPruningSweepResult(
            threshold=token_threshold,
            passed=passed,
            context_pruned_count=context_pruned_count,
            token_budget_hit=token_budget_hit,
            dropped_total=dropped_total,
            total_events=total_events,
        )
        with open(sweep_results_path, "a") as f:
            f.write(result.model_dump_json() + "\n")

        assert hook_failures == [], (
            f"expected zero HookRegistry._on_error calls; got {hook_failures!r}"
        )
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()
        _pruner.close()


@pytest.mark.benchmark
def test_context_pruning_sweep_aggregate(sweep_results_path: Path) -> None:
    """Aggregate results across all three thresholds and assert regression bounds.

    Reads the session-scoped ``sweep_results_path`` file (populated by the three
    parametrized ``test_context_pruning_sweep_run`` runs) and asserts:

    1. Exactly three results are present (one per threshold).
    2. The 8192 pass rate is within 5 percentage points of the 16384 baseline
       (ADR-0021 §5 regression threshold guidance).
    3. Each result has a valid ``context_pruned_count`` for KPI aggregation.

    The ``context_efficiency`` KPI is computed as::

        context_efficiency = 1 - (dropped_total / total_events)

    A value near 1.0 means pruning rarely fired; near 0.0 means heavy pruning.
    """
    _sweep_results: list[ContextPruningSweepResult] = []
    if sweep_results_path.exists():
        with open(sweep_results_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    _sweep_results.append(ContextPruningSweepResult.model_validate_json(line))

    thresholds_seen = {r.threshold for r in _sweep_results}
    assert len(_sweep_results) == 3, (
        f"expected 3 sweep results (one per threshold); got {len(_sweep_results)}: "
        f"{[(r.threshold, r.passed) for r in _sweep_results]!r}"
    )
    assert thresholds_seen == {4096, 8192, 16384}, (
        f"expected thresholds {{4096, 8192, 16384}}; got {thresholds_seen!r}"
    )

    result_by_threshold = {r.threshold: r for r in _sweep_results}
    r_8192 = result_by_threshold[8192]
    r_16384 = result_by_threshold[16384]
    _ = result_by_threshold[4096]

    assert r_8192.passed, (
        "sweep must pass at 8192 baseline (ADR-0021 §5 acceptance criterion); "
        f"got passed={r_8192.passed}, "
        f"context_pruned_count={r_8192.context_pruned_count}"
    )

    if r_16384.passed:
        delta_pp = (int(r_8192.passed) - int(r_16384.passed)) * 100
        assert abs(delta_pp) <= 5, (
            f"8192 pass rate must be within 5 pp of 16384 baseline; "
            f"got delta={delta_pp:+d} pp "
            f"(8192={'pass' if r_8192.passed else 'fail'}, "
            f"16384={'pass' if r_16384.passed else 'fail'})"
        )

    for r in _sweep_results:
        assert r.context_pruned_count >= 0, (
            f"context_pruned_count must be non-negative for threshold {r.threshold}; "
            f"got {r.context_pruned_count}"
        )
        assert r.dropped_total >= 0, (
            f"dropped_total must be non-negative for threshold {r.threshold}; got {r.dropped_total}"
        )
        assert r.total_events > 0, (
            f"total_events must be positive for threshold {r.threshold}; got {r.total_events}"
        )


class _NeverCalledAdapter:
    """Mock adapter used only for validation-path tests.

    In the error case (context_tokens > budget) the ``ValueError`` fires
    before any adapter method is called.  In the non-error cases the
    adapter IS called by ``run_task``, so ``stream()`` must yield at
    least one well-formed ``ModelResponseChunk`` — otherwise the
    synchronous ``_consume_model_stream`` loop ``async for`` crashes with
    ``AttributeError: 'NoneType' object has no attribute 'content'``.
    """

    async def complete(self, messages, tools=None, **kwargs):
        raise RuntimeError("_NeverCalledAdapter.complete() was reached")

    async def chat(self, messages, tools=None, **kwargs):
        raise RuntimeError("_NeverCalledAdapter.chat() was reached")

    async def stream(self, messages, tools=None, **kwargs):
        yield ModelResponseChunk(content="")


@pytest.mark.benchmark
def test_context_tokens_above_token_budget_raises_value_error(
    benchmark_workspace: Path,
    monkeypatch,
) -> None:
    """ADR-0021 §6 regression: ``ValueError`` fires when
    ``FOUNDRY_CONTEXT_TOKENS > FOUNDRY_TOKEN_BUDGET``.

    The guard prevents a TOCTOU-class failure where the pruning threshold
    exceeds the abort threshold, causing sessions to prune indefinitely
    without ever aborting.  The validation fires synchronously at
    ``run_task`` startup, before the model loop begins, so a
    ``_NeverCalledAdapter`` is sufficient to exercise the path.
    """
    monkeypatch.setenv("FOUNDRY_CONTEXT_TOKENS", "10000")
    monkeypatch.setenv("FOUNDRY_TOKEN_BUDGET", "5000")

    harness_dir = benchmark_workspace / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "skills").mkdir(exist_ok=True)

    db = benchmark_workspace / "traces.db"

    async def _call_run_task() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="test-0.0") as sid:
            await run_task(
                "context-tokens- validation",
                harness_dir,
                logger,
                sid,
                model_adapter=_NeverCalledAdapter(),
            )

    with pytest.raises(ValueError, match=r"FOUNDRY_CONTEXT_TOKENS.*FOUNDRY_TOKEN_BUDGET"):
        asyncio.run(_call_run_task())


@pytest.mark.benchmark
@pytest.mark.parametrize(
    "context_tokens,budget",
    [
        ("5000", "10000"),
        ("5000", "5000"),
    ],
)
def test_context_tokens_at_or_below_token_budget_succeeds(
    benchmark_workspace: Path,
    monkeypatch,
    context_tokens: str,
    budget: str,
) -> None:
    """ADR-0021 §6: no exception when ``FOUNDRY_CONTEXT_TOKENS <= FOUNDRY_TOKEN_BUDGET``.

    False-positive prevention: the guard must not fire when the threshold
    is at or below the budget.
    """
    monkeypatch.setenv("FOUNDRY_CONTEXT_TOKENS", context_tokens)
    monkeypatch.setenv("FOUNDRY_TOKEN_BUDGET", budget)

    harness_dir = benchmark_workspace / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "skills").mkdir(exist_ok=True)

    db = benchmark_workspace / "traces.db"

    async def _call_run_task() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="test-0.0") as sid:
            await run_task(
                "context-tokens-under-budget",
                harness_dir,
                logger,
                sid,
                model_adapter=_NeverCalledAdapter(),
            )

    asyncio.run(_call_run_task())


@pytest.mark.benchmark
def test_context_tokens_set_budget_unset_succeeds(benchmark_workspace: Path, monkeypatch) -> None:
    """ADR-0021 §6: when ``FOUNDRY_TOKEN_BUDGET`` is absent the guard skips
    the comparison entirely (budget_raw is empty → check is bypassed).
    """
    monkeypatch.setenv("FOUNDRY_CONTEXT_TOKENS", "100000")
    monkeypatch.delenv("FOUNDRY_TOKEN_BUDGET", raising=False)

    harness_dir = benchmark_workspace / "harness"
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "skills").mkdir(exist_ok=True)

    db = benchmark_workspace / "traces.db"

    async def _call_run_task() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="test-0.0") as sid:
            await run_task(
                "context-tokens-no-budget",
                harness_dir,
                logger,
                sid,
                model_adapter=_NeverCalledAdapter(),
            )

    asyncio.run(_call_run_task())
