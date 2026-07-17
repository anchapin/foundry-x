"""Subprocess coverage for infra/scripts/run_benchmark.sh.

Asserts the acceptance criteria from issue #207:
  * --help exits 0 and documents the flags.
  * --dry-run prints the resolved docker compose argv without executing.
  * Missing required args exit with code 2.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "infra" / "scripts" / "run_benchmark.sh"


def _bash_binary() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash binary is not on PATH; skipping run_benchmark.sh subprocess tests")
    return bash


def _run_script(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    if env is not None:
        run_env.update(env)
    return subprocess.run(
        [_bash_binary(), str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=run_env,
    )


# ---------------------------------------------------------------------------
# --help
# ---------------------------------------------------------------------------


def test_help_exits_zero() -> None:
    result = _run_script("--help")
    assert result.returncode == 0
    for flag in ("--model", "--task", "--compose-extra", "--keep-server", "--dry-run"):
        assert flag in result.stdout


def test_short_help_flag_works() -> None:
    result = _run_script("-h")
    assert result.returncode == 0
    assert "usage:" in result.stdout


# ---------------------------------------------------------------------------
# Missing required arguments
# ---------------------------------------------------------------------------


def test_missing_task_exits_two() -> None:
    result = _run_script()
    assert result.returncode == 2
    assert "--task is required" in result.stderr


def test_task_without_value_exits_two() -> None:
    result = _run_script("--task")
    assert result.returncode == 2
    assert "--task requires a prompt argument" in result.stderr


def test_model_without_value_exits_two() -> None:
    result = _run_script("--task", "x", "--model")
    assert result.returncode == 2
    assert "--model requires a path argument" in result.stderr


def test_unknown_flag_exits_two() -> None:
    result = _run_script("--task", "x", "--bogus")
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


# ---------------------------------------------------------------------------
# --dry-run: resolved argv
# ---------------------------------------------------------------------------


def test_dry_run_prints_compose_invocation() -> None:
    result = _run_script("--task", "Summarize the README", "--dry-run")
    assert result.returncode == 0
    assert "docker" in result.stdout
    assert "compose" in result.stdout
    assert "run" in result.stdout
    assert "foundryx" in result.stdout
    assert "--task" in result.stdout
    # %q shell-quoting escapes spaces, so check for the first word.
    assert "Summarize" in result.stdout
    assert "docker-compose.yml" in result.stdout


def test_dry_run_includes_compose_extra() -> None:
    result = _run_script(
        "--task",
        "hello",
        "--compose-extra",
        "-f,infra/docker/docker-compose.rocm.yml",
        "--dry-run",
    )
    assert result.returncode == 0
    assert "docker-compose.rocm.yml" in result.stdout
    # The base file must still be present alongside the override.
    assert "docker-compose.yml" in result.stdout


def test_dry_run_does_not_invoke_docker() -> None:
    """--dry-run must not actually start a container."""
    result = _run_script("--task", "x", "--dry-run")
    assert result.returncode == 0
    assert "Running sandbox" not in result.stdout


# ---------------------------------------------------------------------------
# Model validation (issue #745)
# ---------------------------------------------------------------------------


def _run_with_mock_curl(
    mock_responses: dict[str, str],
    extra_args: tuple[str, ...] = (),
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run script with a mocked curl that returns predefined responses per URL."""
    import tempfile
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        mock_curl = Path(tmpdir) / "curl"
        mock_script_lines = ["#!/usr/bin/env bash"]
        for url, resp in mock_responses.items():
            escaped_resp = json.dumps(resp)
            mock_script_lines.append(f'if [[ "$@" == *"{url}"* ]]; then echo {escaped_resp}; fi')
        mock_script_lines.append("exit 0")
        mock_curl.write_text("\n".join(mock_script_lines))
        mock_curl.chmod(0o755)

        path_env = f"{tmpdir}:" + os.environ.get("PATH", "")
        run_env: dict[str, str] = {"PATH": path_env}
        if extra_env:
            run_env.update(extra_env)

        try:
            return _run_script(
                "--task",
                "test task",
                "--model",
                "/srv/models/test-model.Q5_K_M.gguf",
                *extra_args,
                env=run_env,
            )
        except subprocess.TimeoutExpired as e:
            stdout = e.stdout.decode("utf-8", errors="replace") if e.stdout else ""
            stderr = e.stderr.decode("utf-8", errors="replace") if e.stderr else ""
            return subprocess.CompletedProcess(
                args=e.args,
                returncode=-9,
                stdout=stdout,
                stderr=stderr,
            )


def test_model_mismatch_fails_when_server_running_different_model() -> None:
    """When server is healthy but serving a different model, script must fail."""
    result = _run_with_mock_curl(
        mock_responses={
            "http://127.0.0.1:8080/health": '{"ok":true}',
            "http://127.0.0.1:8080/v1/models": '{"data":[{"id":"other-model.Q4_K_M.gguf"}]}',
        },
    )
    assert result.returncode == 1
    assert "other-model.Q4_K_M.gguf" in result.stderr
    assert "test-model.Q5_K_M.gguf" in result.stderr
    assert "benchmark targets" in result.stderr


def test_model_mismatch_succeeds_with_override() -> None:
    """When model mismatch but LLAMACPP_MODEL_OVERRIDE=1, script proceeds with warning."""
    result = _run_with_mock_curl(
        mock_responses={
            "http://127.0.0.1:8080/health": '{"ok":true}',
            "http://127.0.0.1:8080/v1/models": '{"data":[{"id":"other-model.Q4_K_M.gguf"}]}',
        },
        extra_env={"LLAMACPP_MODEL_OVERRIDE": "1"},
    )
    assert "WARNING" in result.stderr
    assert "LLAMACPP_MODEL_OVERRIDE" in result.stderr
    assert "other-model.Q4_K_M.gguf" in result.stderr


def test_model_match_proceeds_past_validation() -> None:
    """When server serves the expected model, script proceeds past model validation.

    The script will timeout on the docker run step (docker not available in test env),
    but that proves it got past the model validation block.
    """
    result = _run_with_mock_curl(
        mock_responses={
            "http://127.0.0.1:8080/health": '{"ok":true}',
            "http://127.0.0.1:8080/v1/models": '{"data":[{"id":"test-model.Q5_K_M.gguf"}]}',
        },
    )
    assert "Verified running server" in result.stdout
    assert "test-model.Q5_K_M.gguf" in result.stdout


def test_model_override_env_var_in_help() -> None:
    """LLAMACPP_MODEL_OVERRIDE must be documented in --help output."""
    result = _run_script("--help")
    assert result.returncode == 0
    assert "LLAMACPP_MODEL_OVERRIDE" in result.stdout
