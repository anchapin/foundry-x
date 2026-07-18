"""Benchmark task: hook_registry_error is emitted when get_registry() raises (issue #619).

Regression target for the ``_resolve_hook_registry`` failure path in
``src/foundry_x/execution/runner.py`` (issue #260). When the harness IS
importable but ``get_registry()`` raises, every security-critical hook —
including the ``InjectionFirewallHook`` mandated by ``SECURITY.md`` —
would be silently disabled for the entire session (AGENTS.md §2 — never
silently swallow an exception). The ``hook_registry_error`` trace event
records that the firewall layer is off so the Digester and operator can
observe the degraded mode. A regression that removes the try/except around
the ``get_registry()`` call, that drops the ``hook_registry_error`` record,
or that raises before the trace signal surfaces here as a failing
benchmark and blocks the harness edit at PR review (ADR-0004).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCallChunk,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import _resolve_hook_registry, main
from foundry_x.trace.logger import TraceLogger

REPO_ROOT = Path(__file__).resolve().parents[2]
REPO_HARNESS_DIR = REPO_ROOT / "harness"


class _ScriptedAdapter:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = list(responses)

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        if not self._responses:
            raise RuntimeError("_ScriptedAdapter exhausted")
        return self._responses.pop(0)

    async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
        return await self.complete(messages, tools, **kwargs)

    async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        if not self._responses:
            raise RuntimeError("_ScriptedAdapter exhausted")
        response = self._responses.pop(0)
        if response.message.content:
            yield ModelResponseChunk(content=response.message.content)
        for i, tc in enumerate(response.tool_calls or []):
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


def _argv(task: str, db_path: Path, harness_dir: Path) -> list[str]:
    return [
        "fx-runner",
        "--task",
        task,
        "--harness-dir",
        str(harness_dir),
        "--trace-path",
        str(db_path),
    ]


TASK = BenchmarkTask(
    name="hook_registry_error",
    description=(
        "_resolve_hook_registry emits a hook_registry_error trace event "
        "when harness.hooks.get_registry() raises, and returns None so the "
        "session continues in degraded mode (no hook fan-out)."
    ),
    prompt=(
        "Inspect src/foundry_x/execution/runner.py: confirm "
        "_resolve_hook_registry still wraps the get_registry() call in a "
        "try/except, records a hook_registry_error event with error_type "
        "and message payload when it catches, and returns None so the "
        "session survives in degraded mode."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "hook_registry_error is recorded with error_type and message "
        "payload; _resolve_hook_registry returns None; no exception "
        "propagates to the caller."
    ),
    tags=["security"],
)


# --- Integration-level benchmark (full run_task path) -------------------------


@pytest.mark.benchmark
def test_hook_registry_error_emits_single_event_and_completes_degraded(
    tmp_path, monkeypatch
) -> None:
    """When get_registry() raises, exactly one hook_registry_error is emitted
    and the session completes in degraded mode (outcome='success').

    This is the full integration path that unit tests in
    tests/execution/test_runner_hook_registry.py cannot cover: it
    exercises the complete run_task -> _resolve_hook_registry ->
    HookRegistry fan-out path end-to-end. A regression that, e.g.,
    calls log.record() twice or raises instead of returning None
    would pass the unit tests but fail here.
    """
    import foundry_x.execution.runner as runner_mod
    import harness.hooks as harness_hooks

    db = tmp_path / "traces.db"
    responses = [
        ModelResponse(
            message=ModelMessage(role="assistant", content="degraded-ok"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("degraded-task", db, REPO_HARNESS_DIR))

    def _boom() -> None:
        raise RuntimeError("registry blew up")

    monkeypatch.setattr(harness_hooks, "get_registry", _boom)

    main()

    session_id = TraceLogger(db).list_sessions()[0].session_id
    events = TraceLogger(db).load_session(session_id)
    kinds = [event.kind for event in events]

    assert "hook_registry_error" in kinds, (
        f"expected a hook_registry_error event when get_registry() raises; kinds={kinds!r}"
    )

    err_events = [event for event in events if event.kind == "hook_registry_error"]
    assert len(err_events) == 1, (
        f"exactly one hook_registry_error event expected; got {len(err_events)}"
    )
    assert err_events[0].payload["error_type"] == "RuntimeError"
    assert err_events[0].payload["message"] == "registry blew up"

    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "success", (
        f"session must complete in degraded mode (status=success); "
        f"got status={outcome_event.payload['status']!r}"
    )


@pytest.mark.benchmark
def test_hook_registry_error_captures_any_exception_type(tmp_path, monkeypatch) -> None:
    """Any Exception subclass from get_registry() is captured with its
    type name and message intact.
    """
    import foundry_x.execution.runner as runner_mod
    import harness.hooks as harness_hooks

    db = tmp_path / "traces.db"
    responses = [
        ModelResponse(
            message=ModelMessage(role="assistant", content="done"),
            finish_reason="stop",
        ),
    ]
    adapter = _ScriptedAdapter(responses)
    monkeypatch.setattr(runner_mod, "build_model_adapter", lambda: adapter)
    monkeypatch.setattr(sys, "argv", _argv("exc-type-task", db, REPO_HARNESS_DIR))

    def _boom() -> None:
        raise ValueError("config invalid")

    monkeypatch.setattr(harness_hooks, "get_registry", _boom)

    main()

    session_id = TraceLogger(db).list_sessions()[0].session_id
    events = TraceLogger(db).load_session(session_id)

    err_events = [event for event in events if event.kind == "hook_registry_error"]
    assert len(err_events) == 1
    assert err_events[0].payload["error_type"] == "ValueError"
    assert err_events[0].payload["message"] == "config invalid"

    outcome_event = next(event for event in events if event.kind == "outcome")
    assert outcome_event.payload["status"] == "success"


@pytest.mark.benchmark
def test_hook_registry_error_is_recorded_and_returns_none(
    tmp_trace_logger: TraceLogger,
) -> None:
    """When get_registry() raises, hook_registry_error is recorded and None is returned.

    The test patches ``harness.hooks.get_registry`` to raise a
    ``RuntimeError``, calls ``_resolve_hook_registry``, and asserts:
    1. The function returns ``None`` (degraded mode).
    2. A ``hook_registry_error`` event is recorded with the correct payload.
    3. No exception propagates to the caller.

    A regression that drops the try/except, swallows the event, or raises
    before the record call fails one of the assertions.
    """
    import unittest.mock
    import harness.hooks

    def _failing_get_registry() -> object:
        raise RuntimeError("registry unavailable: disk full")

    with tmp_trace_logger.session(harness_version="test") as session_id:
        with unittest.mock.patch.object(
            harness.hooks,
            "get_registry",
            side_effect=_failing_get_registry,
        ):
            result = _resolve_hook_registry(tmp_trace_logger, session_id)

    assert result is None, (
        "_resolve_hook_registry must return None when get_registry() raises "
        f"so the session continues in degraded mode; got {result!r}"
    )

    events = tmp_trace_logger.load_session(session_id)
    hook_error_events = [e for e in events if e.kind == "hook_registry_error"]
    assert len(hook_error_events) == 1, (
        f"exactly one hook_registry_error event must be recorded; got {len(hook_error_events)}"
    )
    payload = hook_error_events[0].payload
    assert payload.get("error_type") == "RuntimeError", (
        f"error_type must be RuntimeError; got {payload.get('error_type')!r}"
    )
    assert "disk full" in payload.get("message", ""), (
        f"message must contain the exception text; got {payload.get('message')!r}"
    )


@pytest.fixture
def tmp_trace_logger(tmp_path: Path) -> TraceLogger:
    """Provide a TraceLogger backed by a temporary JSONL file."""
    db_path = tmp_path / "trace.jsonl"
    logger = TraceLogger(db_path, backend="jsonl")
    yield logger
    logger.close()
