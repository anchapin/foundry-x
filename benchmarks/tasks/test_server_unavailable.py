"""Benchmark task: ``server_unavailable`` is emitted when the server becomes unavailable (issue #1006).

This benchmark verifies that the Runner correctly emits a ``server_unavailable``
trace event when :meth:`FoundryServerManager.is_healthy` returns ``False`` at
the start of an agent-loop iteration, and that the session terminates with
``outcome_reason="server_unavailable"`` when the restart attempt fails
(:meth:`FoundryServerManager.restart` raises :class:`ServerLaunchError`).

The stub :class:`_UnhealthyServerManager` simulates a mid-session server crash:
``is_healthy()`` returns ``False`` on the first call, and ``restart()`` raises
:class:`ServerLaunchError` so the runner's supervisor aborts the session.

No existing benchmark exercises this path; a regression that breaks the
health-check or restart supervisor would silently pass every benchmark in
``benchmarks/tasks/`` (issue #1006).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution import runner as runner_mod
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
)
from foundry_x.execution.runner import main
from foundry_x.infra.server_manager import (
    ServerConfig,
    ServerLaunchError,
)
from foundry_x.trace.logger import TraceLogger


class _UnhealthyServerManager:
    """Stub :class:`FoundryServerManager` that reports the server as unavailable.

    ``is_healthy()`` returns ``False`` on the first call to trigger the
    ``server_unavailable`` emission path. ``restart()`` raises
    :class:`ServerLaunchError` so the runner aborts the session with
    ``outcome_reason="server_unavailable"``.
    """

    def __init__(self) -> None:
        self._is_healthy_called = False
        self._config = ServerConfig(
            host="localhost",
            model_path=None,
            n_gpu_layers="0",
            ctx_size="8192",
            autostart=True,
            server_bin=None,
        )

    @property
    def config(self) -> ServerConfig:
        return self._config

    @property
    def host(self) -> str:
        return self._config.host

    @property
    def health_url(self) -> str:
        return f"http://{self._config.host}/health"

    @property
    def restart_count(self) -> int:
        return 0

    async def is_healthy(self) -> bool:
        self._is_healthy_called = True
        return False

    async def restart(self) -> bool:
        raise ServerLaunchError("stub: server unavailable for benchmark")


def _argv(task: str, trace_path: Path, harness_dir: Path) -> list[str]:
    return [
        "fx-runner",
        "--task",
        task,
        "--harness-dir",
        str(harness_dir),
        "--trace-path",
        str(trace_path),
    ]


def _stub_harness(harness_dir: Path) -> None:
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for server_unavailable\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def _server_unavailable_event(events) -> dict:
    unavailable = [event.payload for event in events if event.kind == "server_unavailable"]
    assert len(unavailable) == 1, (
        f"expected exactly one server_unavailable event; got {len(unavailable)}. "
        "If 0: the health-check hook was bypassed. "
        "If >1: _handle_server_unavailable was called multiple times per session."
    )
    return unavailable[0]


def _outcome_event(events) -> dict:
    outcomes = [event.payload for event in events if event.kind == "outcome"]
    assert len(outcomes) == 1, f"expected exactly one outcome event; got {len(outcomes)}"
    return outcomes[0]


TASK = BenchmarkTask(
    name="server_unavailable",
    description=(
        "Verifies the Runner emits server_unavailable when the model server becomes "
        "unavailable mid-session and aborts with outcome_reason=server_unavailable."
    ),
    prompt=(
        "This benchmark does not run an agent; it drives Runner.run_task with a "
        "stub server manager that reports is_healthy=False to verify the "
        "server_unavailable event path and session abort."
    ),
    difficulty_tier="smoke",
    expected_outcome=(
        "With a stub manager that reports is_healthy=False and restart() raising "
        "ServerLaunchError, the Runner records exactly one server_unavailable event "
        "and terminates with outcome.status=failed and "
        "outcome.reason=server_unavailable."
    ),
    tags=["server", "health-check", "infrastructure"],
)


class _FinalAnswerAdapter:
    """Stub ``ModelAdapter`` that returns a single final answer immediately."""

    async def complete(self, messages, tools=None, **kwargs):
        return ModelResponse(
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        )

    async def chat(self, messages, tools=None, **kwargs):
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):
        response = await self.complete(messages, tools, **kwargs)
        yield response.message.content or ""


@pytest.mark.benchmark
def test_server_unavailable_emits_event_and_aborts_session(tmp_path, monkeypatch):
    """``server_unavailable`` is recorded and the session aborts when restart fails.

    This pins three regression targets:

    1. Exactly one ``server_unavailable`` event is present in the trace.
       A regression that bypasses the mid-session health-check hook would
       produce zero events.
    2. ``outcome.status`` is ``"failed"`` and ``outcome.reason`` is
       ``"server_unavailable"``. A regression in the abort logic would
       produce a different reason (e.g. ``"final_answer"``).
    3. ``_handle_server_unavailable`` calls ``restart()`` and the
       ``ServerLaunchError`` propagates to set ``should_abort=True``.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    server_manager = _UnhealthyServerManager()
    adapter = _FinalAnswerAdapter()

    async def drive(task, harness_dir, log, session_id):
        await runner_mod.run_task(
            task,
            harness_dir,
            log,
            session_id,
            model_adapter=adapter,
            server_manager=server_manager,
        )

    monkeypatch.setattr(sys, "argv", _argv("server-unavailable", db, harness_dir))
    main(run_task_fn=drive)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)

    # --- server_unavailable event ------------------------------------------
    unavailable = _server_unavailable_event(events)
    assert unavailable["step"] == 0, (
        f"server_unavailable should fire at step 0; got step={unavailable['step']}"
    )
    assert unavailable["host"] == "localhost"
    assert unavailable["restart_attempted"] is True, (
        "restart_attempted must be True because autostart=True on the stub manager"
    )

    # --- outcome event ----------------------------------------------------
    outcome = _outcome_event(events)
    assert outcome["status"] == "failed", (
        f"outcome.status must be 'failed'; got {outcome['status']!r}"
    )
    assert outcome["reason"] == "server_unavailable", (
        f"outcome.reason must be 'server_unavailable'; got {outcome['reason']!r}"
    )
