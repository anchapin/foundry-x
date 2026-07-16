"""Benchmark task: hook_registry_error is emitted when get_registry() raises (issue #647).

Regression target for ``src/foundry_x/execution/runner._resolve_hook_registry``
(issue #260). When ``harness.hooks.get_registry()`` raises after a successful
lazy import, the runner must:

1. Record exactly one ``hook_registry_error`` trace event (with
   ``error_type`` and ``message``) so the Digester and operator observe
   that the security-critical injection firewall is off.
2. Return ``None`` so the session continues in degraded mode (no hooks
   fan-out), rather than crashing or silently swallowing the exception
   (AGENTS.md §2 — never silently swallow an exception).

A regression that removes the event, emits it more than once, or raises
instead of returning ``None`` surfaces here as a failing benchmark and
blocks the harness edit at PR review (ADR-0004).

SECURITY.md §"Prompt-input firewall" names ``InjectionFirewallHook`` as
security-critical. When the hook registry is unavailable, that entire
layer is disabled for the session. The ``hook_registry_error`` event is
the operator's only signal that this has happened.
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
from foundry_x.execution.runner import main
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
        "When get_registry() raises, the runner records exactly one "
        "hook_registry_error trace event and completes the session in "
        "degraded mode (no hooks, including the security-critical "
        "InjectionFirewallHook)."
    ),
    prompt=(
        "Inspect src/foundry_x/execution/runner.py: confirm "
        "_resolve_hook_registry still catches Exception from "
        "get_registry(), records hook_registry_error with error_type "
        "and message, and returns None so the session continues in "
        "degraded mode."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "When get_registry() raises: exactly one hook_registry_error "
        "event is emitted with error_type and message payload; the "
        "session outcome is 'success' (degraded mode, not task_failed)."
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
