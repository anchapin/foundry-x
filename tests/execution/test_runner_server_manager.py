"""Integration tests for the runner's mid-session server health-check (issue #899).

``FoundryServerManager.ensure_healthy`` is called pre-session, and a
mid-session ``is_healthy()`` returning ``False`` records a
``server_unavailable`` trace event and calls ``restart()``. These
tests mock the manager so no real subprocess is spawned.

Pattern mirrors ``tests/execution/test_runner_event_limit.py`` — a
scripted ``ModelAdapter`` yields a single turn, the runner is invoked
with an injected ``FoundryServerManager``, and the trace store is
inspected for the expected events.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry_x.execution.model_adapter import (
    ModelResponseChunk,
)
from foundry_x.execution.runner import RunLimits, run_task as real_run_task
from foundry_x.infra.server_manager import (
    FoundryServerManager,
    ServerConfig,
    ServerLaunchError,
)
from foundry_x.trace.logger import TraceLogger


def _stub_harness(harness_dir: Path) -> None:
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


class _FinalAnswerAdapter:
    """Adapter that yields a final-answer turn (no tool calls)."""

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        yield ModelResponseChunk(content="done")
        yield ModelResponseChunk(finish_reason="stop")

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream()")

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        raise AssertionError("run_task must call stream()")


class _FakeServerManager:
    """Minimal stand-in for :class:`FoundryServerManager` used by the runner.

    The runner only calls ``is_healthy()`` and ``restart()``; both
    are recorded here so the integration tests can assert on call
    count and return sequence. ``config`` is exposed so
    ``_handle_server_unavailable`` can branch on ``autostart``.
    """

    def __init__(
        self,
        *,
        healthy: bool = True,
        autostart: bool = True,
        restart_outcome: bool = True,
    ) -> None:
        self._healthy = healthy
        self._restart_outcome = restart_outcome
        self.config = ServerConfig(
            host="http://127.0.0.1:8080",
            model_path="/tmp/x.gguf",
            n_gpu_layers="0",
            ctx_size="8192",
            autostart=autostart,
            server_bin=None,
        )
        self.is_healthy_calls = 0
        self.restart_calls = 0

    @property
    def host(self) -> str:
        return self.config.host

    @property
    def health_url(self) -> str:
        return "http://127.0.0.1:8080/health"

    async def is_healthy(self) -> bool:
        self.is_healthy_calls += 1
        return self._healthy

    async def restart(self) -> bool:
        self.restart_calls += 1
        return self._restart_outcome


@pytest.mark.asyncio
async def test_runner_records_server_unavailable_when_mid_session_unhealthy(
    tmp_path: Path,
) -> None:
    """Mid-session ``is_healthy() == False`` records ``server_unavailable``
    and calls ``restart()``. Successful restart lets the loop continue.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    manager = _FakeServerManager(healthy=False, autostart=True, restart_outcome=True)

    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "task",
            harness_dir,
            logger,
            session_id,
            model_adapter=_FinalAnswerAdapter(),
            skill_executor=None,
            limits=RunLimits(),
            workspace_root=None,
            server_manager=manager,  # type: ignore[arg-type]
        )

    events = logger.load_session(session_id)
    unavailable = [e for e in events if e.kind == "server_unavailable"]
    assert len(unavailable) == 1, f"expected 1 server_unavailable event, got {unavailable}"
    payload = unavailable[0].payload
    assert payload["step"] == 0
    assert payload["host"] == "http://127.0.0.1:8080"
    assert payload["health_url"] == "http://127.0.0.1:8080/health"
    assert payload["restart_attempted"] is True

    # The manager was probed for health AND its restart loop was called.
    assert manager.is_healthy_calls >= 1
    assert manager.restart_calls == 1

    # Outcome is success because the restart recovered in time.
    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "success"
    assert outcome.payload["reason"] == "final_answer"


@pytest.mark.asyncio
async def test_runner_fails_session_when_restart_attempts_exhausted(
    tmp_path: Path,
) -> None:
    """When ``restart()`` returns ``False``, the loop terminates with
    ``outcome.status="failed"`` and ``reason="server_unavailable"``.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    manager = _FakeServerManager(healthy=False, autostart=True, restart_outcome=False)

    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "task",
            harness_dir,
            logger,
            session_id,
            model_adapter=_FinalAnswerAdapter(),
            skill_executor=None,
            limits=RunLimits(),
            workspace_root=None,
            server_manager=manager,  # type: ignore[arg-type]
        )

    events = logger.load_session(session_id)
    unavailable = [e for e in events if e.kind == "server_unavailable"]
    assert len(unavailable) == 1
    assert unavailable[0].payload["restart_attempted"] is True

    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "failed"
    assert outcome.payload["reason"] == "server_unavailable"


@pytest.mark.asyncio
async def test_runner_skips_restart_when_autostart_off(tmp_path: Path) -> None:
    """Autostart-off mode is a passive prober — no restart is attempted."""
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    manager = _FakeServerManager(healthy=False, autostart=False, restart_outcome=False)

    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "task",
            harness_dir,
            logger,
            session_id,
            model_adapter=_FinalAnswerAdapter(),
            skill_executor=None,
            limits=RunLimits(),
            workspace_root=None,
            server_manager=manager,  # type: ignore[arg-type]
        )

    events = logger.load_session(session_id)
    unavailable = [e for e in events if e.kind == "server_unavailable"]
    assert len(unavailable) == 1
    # Autostart is off — the runner recorded the detection but did not
    # call ``restart()``.
    assert unavailable[0].payload["restart_attempted"] is False
    assert manager.restart_calls == 0

    # The session continues because the runner respects the operator's
    # choice to manage the server out-of-band — the model adapter's
    # own retry policy handles the outage.
    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "success"
    assert outcome.payload["reason"] == "final_answer"


@pytest.mark.asyncio
async def test_runner_does_not_probe_when_no_manager(tmp_path: Path) -> None:
    """When ``server_manager=None`` (the default), no health probe runs.

    This guards against accidental behaviour change for callers that
    do not opt in to the supervisor — the loop runs as before.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "task",
            harness_dir,
            logger,
            session_id,
            model_adapter=_FinalAnswerAdapter(),
            skill_executor=None,
            limits=RunLimits(),
            workspace_root=None,
            server_manager=None,
        )

    events = logger.load_session(session_id)
    unavailable = [e for e in events if e.kind == "server_unavailable"]
    assert unavailable == []
    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "success"


@pytest.mark.asyncio
async def test_runner_surfaces_serverlauncherror_as_model_error(tmp_path: Path) -> None:
    """``ServerLaunchError`` from ``restart()`` is surfaced as a
    ``model_error`` trace event (AGENTS.md §2 — never silently swallow).
    """

    class _RaisingManager(_FakeServerManager):
        async def restart(self) -> bool:  # noqa: D401
            self.restart_calls += 1
            raise ServerLaunchError("binary missing")

    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    manager = _RaisingManager(healthy=False, autostart=True)

    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "task",
            harness_dir,
            logger,
            session_id,
            model_adapter=_FinalAnswerAdapter(),
            skill_executor=None,
            limits=RunLimits(),
            workspace_root=None,
            server_manager=manager,  # type: ignore[arg-type]
        )

    events = logger.load_session(session_id)
    unavailable = [e for e in events if e.kind == "server_unavailable"]
    assert len(unavailable) == 1
    model_errors = [e for e in events if e.kind == "model_error"]
    assert any(e.payload["error_type"] == "ServerLaunchError" for e in model_errors), model_errors
    # The exception did not escape the runner; the session terminated
    # cleanly with the supervisor-recorded reason.
    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "failed"
    assert outcome.payload["reason"] == "server_unavailable"


@pytest.mark.asyncio
async def test_runner_passes_real_foundry_server_manager(
    tmp_path: Path,
) -> None:
    """A real ``FoundryServerManager`` instance is accepted by ``run_task``
    (no isinstance check rejects it). The probe path uses an offline
    endpoint so the manager reports unhealthy but does not raise.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    # Use a closed port — ``is_healthy`` returns ``False`` without
    # raising, but ``restart`` would fail; autostart is on so the
    # server_unavailable event still surfaces.
    manager = FoundryServerManager(
        ServerConfig(
            host="http://127.0.0.1:1",  # closed port → connect refused
            model_path="/tmp/x.gguf",
            n_gpu_layers="0",
            ctx_size="8192",
            autostart=True,
            server_bin="/nonexistent/llama-server",
            health_ready_timeout_s=0.05,
            max_restart_attempts=1,
            restart_backoff_base_s=0.0,
            restart_backoff_cap_s=0.0,
        )
    )

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as session_id:
        await real_run_task(
            "task",
            harness_dir,
            logger,
            session_id,
            model_adapter=_FinalAnswerAdapter(),
            skill_executor=None,
            limits=RunLimits(),
            workspace_root=None,
            server_manager=manager,
        )

    events = logger.load_session(session_id)
    unavailable = [e for e in events if e.kind == "server_unavailable"]
    assert len(unavailable) == 1
    # The real manager's ``restart()`` catches ``ServerLaunchError``
    # inside ``_restart_sync`` and returns ``False`` (rather than
    # propagating), so the runner records no ``model_error`` event —
    # the supervisor's bounded retry loop already owned the failure.
    # The session still aborts with outcome.failed / server_unavailable
    # because the runner treats ``not restart()`` as "abort".
    assert manager.restart_count == 0  # all attempts failed
    outcome = next(e for e in events if e.kind == "outcome")
    assert outcome.payload["status"] == "failed"
    assert outcome.payload["reason"] == "server_unavailable"
