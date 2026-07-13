from __future__ import annotations

import sys
from pathlib import Path

import pytest

from foundry_x.execution.runner import main
from foundry_x.trace.logger import TraceLogger


def _argv(task: str, trace_path: Path, harness_dir: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fx-runner",
            "--task",
            task,
            "--trace-path",
            str(trace_path),
            "--harness-dir",
            str(harness_dir),
        ],
    )


def _stub_harness(harness_dir: Path) -> None:
    """Build a minimal valid harness layout under ``harness_dir`` (issue #90).

    ``main()`` validates the harness layout before touching ``sys.path``;
    these stubs satisfy the gate so the terminal-event unit under test runs.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "_version.txt").write_text("0.1.0-test\n", encoding="utf-8")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def test_main_records_task_failed_and_reraises(tmp_path, monkeypatch):
    """Acceptance test for issue #10: a failing run_task records both
    task_received and task_failed (with error_type) and still propagates."""
    db = tmp_path / "traces.db"
    _stub_harness(tmp_path)
    _argv("do something risky", db, tmp_path, monkeypatch)

    async def failing_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError):
        main(run_task_fn=failing_run_task)

    logger = TraceLogger(db)
    sessions = logger.list_sessions()
    assert len(sessions) == 1
    events = logger.load_session(sessions[0].session_id)
    kinds = [e.kind for e in events]
    assert kinds.count("task_received") == 1
    assert "task_failed" in kinds

    failed = [e for e in events if e.kind == "task_failed"][0]
    assert failed.payload["error_type"] == "RuntimeError"
    assert failed.payload["message"] == "boom"
    assert failed.payload["duration_ms"] >= 0


def test_main_records_task_completed_on_success(tmp_path, monkeypatch):
    """Acceptance test for issue #10: a succeeding run_task records a
    task_completed event with a non-negative duration_ms."""
    db = tmp_path / "traces.db"
    _stub_harness(tmp_path)
    _argv("no-op task", db, tmp_path, monkeypatch)

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    logger = TraceLogger(db)
    sessions = logger.list_sessions()
    assert len(sessions) == 1
    events = logger.load_session(sessions[0].session_id)
    kinds = [e.kind for e in events]
    assert kinds.count("task_received") == 1
    assert "task_failed" not in kinds

    completed = [e for e in events if e.kind == "task_completed"][0]
    assert completed.payload["duration_ms"] >= 0


def test_main_terminal_event_after_timeout(tmp_path, monkeypatch):
    """A wall-clock timeout (re-raised TimeoutError) still produces a
    terminal task_failed event, so the session always has an outcome."""
    db = tmp_path / "traces.db"
    _stub_harness(tmp_path)
    monkeypatch.setenv("FOUNDRY_TASK_TIMEOUT", "0.05")
    _argv("slow task", db, tmp_path, monkeypatch)

    async def slow_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        import asyncio

        await asyncio.sleep(1.0)

    with pytest.raises(TimeoutError):
        main(run_task_fn=slow_run_task)

    logger = TraceLogger(db)
    sessions = logger.list_sessions()
    events = logger.load_session(sessions[0].session_id)
    kinds = [e.kind for e in events]
    assert "task_aborted" in kinds
    assert "task_failed" in kinds
    failed = [e for e in events if e.kind == "task_failed"][0]
    assert failed.payload["error_type"] == "TimeoutError"
