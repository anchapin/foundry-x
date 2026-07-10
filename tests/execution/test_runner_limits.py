from __future__ import annotations

import asyncio

import pytest

from foundry_x.execution.runner import (
    DEFAULT_TASK_TIMEOUT_S,
    RunLimits,
    run_limits_from_env,
    run_with_limits,
)
from foundry_x.trace.logger import TraceLogger


async def _stub_slow() -> str:
    """Stub coroutine that runs longer than any test timeout."""
    await asyncio.sleep(1.0)
    return "done"


@pytest.mark.asyncio
async def test_wall_clock_timeout_aborts_and_records(tmp_path):
    """Acceptance test for issue #4: exceeding the cap aborts the run,
    writes a task_aborted trace event, and propagates TimeoutError."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    limits = RunLimits(task_timeout_s=0.1, token_budget=None)

    with logger.session(harness_version="test-0.0") as session_id:
        with pytest.raises(TimeoutError):
            await run_with_limits(_stub_slow(), logger, session_id, limits)

        events = logger.load_session(session_id)

    aborted = [e for e in events if e.kind == "task_aborted"]
    assert len(aborted) == 1
    assert aborted[0].payload["reason"] == "wall_clock"
    assert aborted[0].payload["timeout_s"] == 0.1


@pytest.mark.asyncio
async def test_result_returned_when_under_cap(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    limits = RunLimits(task_timeout_s=1.0, token_budget=None)

    async def quick() -> str:
        await asyncio.sleep(0.01)
        return "ok"

    with logger.session(harness_version="test-0.0") as session_id:
        result = await run_with_limits(quick(), logger, session_id, limits)

    assert result == "ok"


@pytest.mark.asyncio
async def test_no_timeout_when_cap_disabled(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    limits = RunLimits(task_timeout_s=None)

    with logger.session(harness_version="test-0.0") as session_id:
        result = await run_with_limits(_stub_slow(), logger, session_id, limits)

    assert result == "done"


def test_run_limits_from_env_defaults():
    limits = run_limits_from_env(env={})
    assert limits.task_timeout_s == DEFAULT_TASK_TIMEOUT_S
    assert limits.token_budget is None


def test_run_limits_from_env_override():
    limits = run_limits_from_env(env={"FOUNDRY_TASK_TIMEOUT": "30", "FOUNDRY_TOKEN_BUDGET": "4096"})
    assert limits.task_timeout_s == 30
    assert limits.token_budget == 4096


def test_run_limits_from_env_disable_timeout():
    limits = run_limits_from_env(env={"FOUNDRY_TASK_TIMEOUT": "0"})
    assert limits.task_timeout_s is None
