from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

import pytest

from foundry_x.execution.runner import _DEFAULT_MAX_AGENT_STEPS, _resolve_max_steps, main


def _stub_harness(harness_dir: Path) -> None:
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n", encoding="utf-8")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


def _argv(task, trace_path, harness_dir, monkeypatch, validate=False):
    argv = [
        "fx-runner",
        "--task",
        task,
        "--trace-path",
        str(trace_path),
        "--harness-dir",
        str(harness_dir),
    ]
    if validate:
        argv.append("--validate")
    monkeypatch.setattr(sys, "argv", argv)


class TestValidateMode:
    def test_validate_harness_validation_error(self, tmp_path, monkeypatch):
        harness_dir = tmp_path / "missing_harness"
        harness_dir.mkdir(parents=True, exist_ok=True)
        trace_path = tmp_path / "traces.db"
        _argv("noop task", trace_path, harness_dir, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2

    def test_validate_adapter_missing_env_vars(self, tmp_path, monkeypatch):
        _stub_harness(tmp_path)
        trace_path = tmp_path / "traces.db"
        for key in ("OPENCODE_SERVER_URL", "LLAMACPP_HOST"):
            monkeypatch.delenv(key, raising=False)
        _argv("noop task", trace_path, tmp_path, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2

    def test_validate_trace_store_not_writable(self, tmp_path, monkeypatch):
        _stub_harness(tmp_path)
        trace_path = tmp_path / "traces.db"
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:8080")
        _argv("noop task", trace_path, tmp_path, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        def mock_open(self, *args, **kwargs):
            raise OSError("Permission denied")

        with mock.patch.object(Path, "open", mock_open), pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2

    def test_validate_success(self, tmp_path, monkeypatch):
        _stub_harness(tmp_path)
        trace_path = tmp_path / "traces.db"
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:8080")
        _argv("noop task", trace_path, tmp_path, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 0


class TestResolveMaxSteps:
    def test_resolve_max_steps_non_positive_warns(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_MAX_AGENT_STEPS", "-1")
        with pytest.warns(UserWarning, match="FOUNDRY_MAX_AGENT_STEPS=-1.*non-positive"):
            result = _resolve_max_steps()
        assert result == _DEFAULT_MAX_AGENT_STEPS

    def test_resolve_max_steps_zero_warns(self, monkeypatch):
        monkeypatch.setenv("FOUNDRY_MAX_AGENT_STEPS", "0")
        with pytest.warns(UserWarning, match="FOUNDRY_MAX_AGENT_STEPS=0.*non-positive"):
            result = _resolve_max_steps()
        assert result == _DEFAULT_MAX_AGENT_STEPS
