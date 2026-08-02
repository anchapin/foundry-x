from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest

from foundry_x.execution.runner import _DEFAULT_MAX_AGENT_STEPS, _resolve_max_steps, main

REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_LOAD_CHECK = REPO_ROOT / "harness" / "scripts" / "load_check.py"

_MINIMAL_HOOKS_INIT = (
    "class HookRegistry:\n"
    "    def __init__(self):\n"
    "        self._hooks = []\n"
    "\n"
    "    def register(self, hook):\n"
    "        self._hooks.append(hook)\n"
    "\n"
    "\n"
    "def get_registry():\n"
    "    return HookRegistry()\n"
)

_MINIMAL_MANIFEST = json.dumps(
    {"version": "0.0.0", "model_target": "test", "hooks": [], "skills": []}
)


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal harness that passes both path and strict validation.

    Creates system_prompt.txt, hooks/__init__.py (with get_registry),
    skills/, manifest.json, and scripts/load_check.py (issue #1468).
    The directory must be named ``harness`` so that load_check's
    ``import harness.hooks`` resolves correctly.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n", encoding="utf-8")
    (harness_dir / "skills").mkdir(exist_ok=True)

    hooks_dir = harness_dir / "hooks"
    hooks_dir.mkdir(exist_ok=True)
    (hooks_dir / "__init__.py").write_text(_MINIMAL_HOOKS_INIT, encoding="utf-8")

    (harness_dir / "manifest.json").write_text(_MINIMAL_MANIFEST, encoding="utf-8")

    scripts_dir = harness_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    if REAL_LOAD_CHECK.exists():
        shutil.copyfile(REAL_LOAD_CHECK, scripts_dir / "load_check.py")

    return harness_dir


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
    @pytest.fixture(autouse=True)
    def _clean_harness_imports(self):
        """Remove cached harness.* modules and restore sys.path after each test.

        ``_check_hooks_importable`` (inside load_check.py) imports
        ``harness.hooks`` in-process when strict validation runs. Without
        this cleanup the cached module from one test's fixture would
        shadow the next test's fixture (issue #1468).
        """
        original_path = list(sys.path)
        yield
        for key in [k for k in sys.modules if k == "harness" or k.startswith("harness.")]:
            del sys.modules[key]
        sys.path[:] = original_path

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
        harness_dir = _stub_harness(tmp_path / "harness")
        trace_path = tmp_path / "traces.db"
        for key in ("OPENCODE_SERVER_URL", "LLAMACPP_HOST"):
            monkeypatch.delenv(key, raising=False)
        _argv("noop task", trace_path, harness_dir, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2

    def test_validate_trace_store_not_writable(self, tmp_path, monkeypatch):
        harness_dir = _stub_harness(tmp_path / "harness")
        trace_path = tmp_path / "traces.db"
        trace_path.mkdir()
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:8080")
        _argv("noop task", trace_path, harness_dir, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2

    def test_validate_success(self, tmp_path, monkeypatch):
        harness_dir = _stub_harness(tmp_path / "harness")
        trace_path = tmp_path / "traces.db"
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:8080")
        _argv("noop task", trace_path, harness_dir, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 0

    # --- issue #1468: strict content validation ---------------------------

    def test_validate_strict_fails_on_malformed_skill_json(self, tmp_path, monkeypatch, capsys):
        """A malformed skill JSON must produce a clear, named error -- not a
        deep traceback inside ``run_task`` (issue #1468 criterion 1)."""
        harness_dir = _stub_harness(tmp_path / "harness")
        (harness_dir / "skills" / "broken.json").write_text("{ not json", encoding="utf-8")
        manifest = json.loads((harness_dir / "manifest.json").read_text())
        manifest["skills"] = ["broken.json"]
        (harness_dir / "manifest.json").write_text(json.dumps(manifest))

        trace_path = tmp_path / "traces.db"
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:8080")
        _argv("noop task", trace_path, harness_dir, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "broken.json" in captured.err
        assert "Traceback" not in captured.err

    def test_validate_strict_fails_on_dangling_manifest_ref(self, tmp_path, monkeypatch, capsys):
        """A ``manifest.json`` that references a non-existent skill file must
        fail at ``--validate``, not deep inside ``run_task`` (issue #1468
        criterion 2)."""
        harness_dir = _stub_harness(tmp_path / "harness")
        manifest = json.loads((harness_dir / "manifest.json").read_text())
        manifest["skills"] = ["nonexistent.json"]
        (harness_dir / "manifest.json").write_text(json.dumps(manifest))

        trace_path = tmp_path / "traces.db"
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:8080")
        _argv("noop task", trace_path, harness_dir, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "nonexistent.json" in captured.err
        assert "Traceback" not in captured.err

    def test_validate_strict_fails_on_hook_import_error(self, tmp_path, monkeypatch, capsys):
        """When ``import harness.hooks`` raises ``ImportError``, ``--validate``
        must fail with a clear message (issue #1468 criterion 3)."""
        harness_dir = _stub_harness(tmp_path / "harness")
        (harness_dir / "hooks" / "__init__.py").write_text(
            "raise ImportError('deliberately broken')\n", encoding="utf-8"
        )

        trace_path = tmp_path / "traces.db"
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://localhost:8080")
        _argv("noop task", trace_path, harness_dir, monkeypatch, validate=True)

        async def noop_run_task(task, harness_dir, log, session_id):
            return None

        with pytest.raises(SystemExit) as exc_info:
            main(run_task_fn=noop_run_task)
        assert exc_info.value.code == 2
        captured = capsys.readouterr()
        assert "harness.hooks" in captured.err
        assert "Traceback" not in captured.err


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
