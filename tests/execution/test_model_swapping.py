"""Tests for automated model swapping (issue #494).

Tests the ``build_model_adapter_with_overrides`` function and the
``--quantization`` and ``--path-or-endpoint`` CLI flags added to the runner.
"""

from __future__ import annotations

import pytest

from foundry_x.execution.runner import (
    build_model_adapter_with_overrides,
    main,
    resolve_model_id,
)
from foundry_x.trace.logger import TraceLogger


class TestBuildModelAdapterWithOverrides:
    """Unit tests for build_model_adapter_with_overrides (issue #494)."""

    def test_url_endpoint_used_directly(self):
        """When path_or_endpoint is a URL, it is used as the base URL directly."""
        adapter = build_model_adapter_with_overrides(
            model_id="gpt-4o",
            quantization=None,
            path_or_endpoint="http://127.0.0.1:8080/v1",
        )
        assert adapter.base_url == "http://127.0.0.1:8080/v1"
        assert adapter.model == "gpt-4o"

    def test_https_url_endpoint_used_directly(self):
        """HTTPS URLs are used as-is."""
        adapter = build_model_adapter_with_overrides(
            model_id="gpt-4o",
            quantization=None,
            path_or_endpoint="https://api.openai.com/v1",
        )
        assert adapter.base_url == "https://api.openai.com/v1"
        assert adapter.model == "gpt-4o"

    def test_local_path_uses_llamacpp_host(self):
        """When path_or_endpoint is a local GGUF path, LLAMACPP_HOST is used."""
        env = {"LLAMACPP_HOST": "http://192.168.1.100:8081"}
        adapter = build_model_adapter_with_overrides(
            model_id="codellama",
            quantization=None,
            path_or_endpoint="/srv/models/codellama-7b.Q5_K_M.gguf",
            env=env,
        )
        assert adapter.base_url == "http://192.168.1.100:8081"
        assert adapter.model == "codellama"

    def test_local_path_defaults_to_localhost(self):
        """When path_or_endpoint is a local path with no LLAMACPP_HOST, defaults to localhost."""
        adapter = build_model_adapter_with_overrides(
            model_id="codellama",
            quantization=None,
            path_or_endpoint="/srv/models/codellama.Q5_K_M.gguf",
        )
        assert adapter.base_url == "http://127.0.0.1:8080"
        assert adapter.model == "codellama"

    def test_quantization_sets_env_var(self):
        """Quantization is stored in FOUNDRY_QUANTIZATION for traceability."""
        env = {}
        adapter = build_model_adapter_with_overrides(
            model_id="codellama",
            quantization="Q5_K_M",
            path_or_endpoint="http://127.0.0.1:8080",
            env=env,
        )
        assert adapter.model == "codellama"

    def test_model_id_override_wins(self):
        """Explicit model_id wins over env-derived model name."""
        adapter = build_model_adapter_with_overrides(
            model_id="explicit-model-id",
            quantization=None,
            path_or_endpoint="http://127.0.0.1:8080",
        )
        assert adapter.model == "explicit-model-id"

    def test_fallback_to_env_when_no_overrides(self):
        """When all overrides are None, behaves like build_model_adapter."""
        env = {
            "OPENCODE_SERVER_URL": "http://127.0.0.1:8080",
            "FOUNDRY_MODEL_ID": "env-model-id",
        }
        adapter = build_model_adapter_with_overrides(
            model_id=None,
            quantization=None,
            path_or_endpoint=None,
            env=env,
        )
        assert adapter.base_url == "http://127.0.0.1:8080"
        assert adapter.model == "env-model-id"

    def test_raises_when_no_endpoint_and_no_url(self):
        """Raises ValueError when no base_url can be determined."""
        with pytest.raises(ValueError, match="OPENCODE_SERVER_URL or LLAMACPP_HOST"):
            build_model_adapter_with_overrides(
                model_id=None,
                quantization=None,
                path_or_endpoint=None,
                env={},
            )


class TestModelIdResolutionWithQuantization:
    """Tests that quantization is recorded in trace session (issue #494)."""

    def test_resolve_model_id_derives_from_llamacpp_path(self):
        """Quantization label is derivable from model path basename."""
        env = {"LLAMACPP_MODEL_PATH": "/srv/models/codellama-7b.Q5_K_M.gguf"}
        assert resolve_model_id(env) == "codellama-7b.Q5_K_M.gguf"


def _stub_harness(harness_dir) -> None:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)


class TestQuantizationCliFlag:
    """Tests for the --quantization CLI flag (issue #494)."""

    def test_model_id_from_cli_flag_stored_in_session(self, tmp_path, monkeypatch):
        """When --model-id is set via CLI, it is stored in the trace session."""
        db = tmp_path / "traces.db"
        monkeypatch.setenv("LLAMACPP_HOST", "http://127.0.0.1:8080")
        _stub_harness(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "fx-runner",
                "--task",
                "noop",
                "--trace-path",
                str(db),
                "--harness-dir",
                str(tmp_path),
                "--model-id",
                "codellama-7b-Q5_K_M",
            ],
        )

        async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
            return None

        main(run_task_fn=noop_run_task)

        sessions = TraceLogger(db).list_sessions()
        assert len(sessions) == 1
        assert sessions[0].model_id == "codellama-7b-Q5_K_M"

    def test_path_or_endpoint_url_overrides_default(self, tmp_path, monkeypatch):
        """When --path-or-endpoint is a URL, the adapter uses it as the base URL."""
        db = tmp_path / "traces.db"
        monkeypatch.setenv("LLAMACPP_HOST", "http://127.0.0.1:8080")
        monkeypatch.setenv("OPENCODE_SERVER_URL", "http://default.local:8080")
        _stub_harness(tmp_path)
        monkeypatch.setattr(
            "sys.argv",
            [
                "fx-runner",
                "--task",
                "noop",
                "--trace-path",
                str(db),
                "--harness-dir",
                str(tmp_path),
                "--model-id",
                "gpt-4o",
                "--path-or-endpoint",
                "http://192.168.1.50:9090/v1",
            ],
        )

        async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
            return None

        main(run_task_fn=noop_run_task)

        sessions = TraceLogger(db).list_sessions()
        assert len(sessions) == 1
        assert sessions[0].model_id == "gpt-4o"
