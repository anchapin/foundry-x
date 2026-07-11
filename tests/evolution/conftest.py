from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import ProposedEdit
from foundry_x.trace.logger import TraceEvent

_BASE_TS = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def _event(kind: str, offset: float, payload: dict, *, event_id: str) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        session_id="sess-fixture",
        timestamp=(_BASE_TS + timedelta(seconds=offset)).isoformat(),
        kind=kind,
        payload=payload,
    )


@pytest.fixture
def failing_trace() -> list[TraceEvent]:
    return [
        _event(
            "user_prompt",
            0.0,
            {"prompt": "Fix the bug in auth.py"},
            event_id="evt-prompt",
        ),
        _event(
            "tool_call",
            0.3,
            {"name": "read_file"},
            event_id="evt-call",
        ),
        _event(
            "tool_result",
            0.8,
            {"name": "read_file", "status": "ok"},
            event_id="evt-result",
        ),
        _event(
            "error",
            1.2,
            {"message": "permission denied"},
            event_id="evt-error",
        ),
        _event(
            "outcome",
            2.0,
            {"status": "failed"},
            event_id="evt-outcome",
        ),
    ]


@pytest.fixture
def failure_report() -> FailureReport:
    return FailureReport(
        session_id="sess-fixture",
        summary="Agent called the wrong tool for file deletion.",
        failed_steps=[
            {"step": 3, "kind": "tool_call", "detail": "called rm instead of edit"},
            {"step": 5, "kind": "state_leak", "detail": "temp file left behind"},
        ],
        suspected_causes=[
            "System prompt does not list the edit_file skill.",
            "No hook validates tool selection before execution.",
        ],
        proposed_class="bad-prompt",
    )


@pytest.fixture
def proposed_edit() -> ProposedEdit:
    return ProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="Add edit_file to the available-tool list so the agent stops invoking rm.",
        unified_diff="@@ -1 +1 @@\n-- old line\n++ new line\n",
    )
