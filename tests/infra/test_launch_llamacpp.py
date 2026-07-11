"""Subprocess coverage for infra/scripts/launch_llamacpp.sh.

Asserts --help, the missing-required-arg path, and the resolved argv
shape using a stub llama-server binary (no real server required).
"""

from __future__ import annotations

import os
import re
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "infra" / "scripts" / "launch_llamacpp.sh"


def _bash_binary() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash binary is not on PATH; skipping launch_llamacpp.sh tests")
    return bash


def _run_launch(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
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


def _write_executable(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def test_launch_help_exits_zero() -> None:
    result = _run_launch("--help")

    assert result.returncode == 0
    assert "usage: launch_llamacpp.sh --model <gguf>" in result.stdout
    for flag in ("--model", "--host", "--port", "--n-gpu-layers", "--ctx-size"):
        assert flag in result.stdout


def test_launch_model_is_required() -> None:
    result = _run_launch("--host", "127.0.0.1", "--port", "9000")

    assert result.returncode == 2
    assert "error: --model is required" in result.stderr


def test_launch_unknown_arg_rejected() -> None:
    result = _run_launch("--model", "x.gguf", "--bogus")

    assert result.returncode == 2
    assert "error: unknown argument: --bogus" in result.stderr


def test_launch_resolves_argv_and_reports_pid_health(tmp_path: Path) -> None:
    record = tmp_path / "argv.txt"
    pid_file = tmp_path / "server.pid"
    log_file = tmp_path / "server.log"
    model = tmp_path / "model.Q5_K_M.gguf"
    model.write_text("not a real model", encoding="utf-8")

    stub_bin = tmp_path / "fake-llama-server"
    _write_executable(
        stub_bin,
        f"""#!/usr/bin/env bash
# Record exactly what we were exec'd with so the test can diff the shape.
printf '%s\\n' "$@" > {record!s}
exit 0
""",
    )

    result = _run_launch(
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--n-gpu-layers",
        "35",
        "--ctx-size",
        "4096",
        "--log-file",
        str(log_file),
        "--pid-file",
        str(pid_file),
        env={"LLAMACPP_SERVER_BIN": str(stub_bin)},
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output

    # PID/health are printed to the script's own stdout before exec.
    pid_match = re.search(r"^pid:\s+(\d+)$", result.stdout, re.MULTILINE)
    assert pid_match is not None, f"no pid line on stdout: {result.stdout!r}"
    assert "health: http://127.0.0.1:8080/health" in result.stdout

    # The pid-file mirrors the reported PID.
    assert pid_file.read_text(encoding="utf-8").strip() == pid_match.group(1)

    # The stub saw exactly the resolved flag set, in the documented order.
    argv = record.read_text(encoding="utf-8").splitlines()
    assert argv == [
        "--model",
        str(model),
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--n-gpu-layers",
        "35",
        "--ctx-size",
        "4096",
    ]
