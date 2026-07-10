"""Subprocess coverage for infra/llama-cpp/rocm_setup.sh parser behavior."""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "infra" / "llama-cpp" / "rocm_setup.sh"


def _bash_binary() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash binary is not on PATH; skipping rocm_setup.sh subprocess tests")
    return bash


def _run_rocm_setup(
    *args: str, env: dict[str, str] | None = None
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.pop("LLAMACPP_SMOKE_MODEL", None)
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


def test_rocm_setup_help_exits_zero() -> None:
    result = _run_rocm_setup("--help")

    assert result.returncode == 0
    assert "usage: rocm_setup.sh [--smoke-test <gguf>]" in result.stdout
    assert "--smoke-test <gguf>" in result.stdout


def test_rocm_setup_smoke_test_requires_model_path() -> None:
    result = _run_rocm_setup("--smoke-test")

    assert result.returncode == 2
    assert "error: --smoke-test requires a model path argument" in result.stderr


def test_rocm_setup_cli_smoke_model_overrides_env_var(tmp_path: Path) -> None:
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    llamacpp_dir = tmp_path / "llama.cpp"
    server_bin = llamacpp_dir / "build" / "bin" / "llama-server"
    server_bin.parent.mkdir(parents=True)
    cli_model = tmp_path / "cli.gguf"
    env_model = tmp_path / "env.gguf"
    cli_model.write_text("fake model", encoding="utf-8")
    env_model.write_text("env model", encoding="utf-8")

    _write_executable(fake_bin / "git", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "cmake", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "nproc", "#!/usr/bin/env bash\nprintf '1\\n'\n")
    _write_executable(
        fake_bin / "curl",
        """#!/usr/bin/env bash
case "$*" in
  *"/health"*) printf '{"status":"ok"}' ;;
  *"/completion"*) printf '{"content":"hello"}' ;;
  *"/v1/models"*) printf '{"data":[{"id":"fake-model"}]}' ;;
  *) exit 1 ;;
esac
""",
    )
    _write_executable(
        server_bin,
        """#!/usr/bin/env bash
trap 'exit 0' TERM INT
while true; do
  sleep 1
done
""",
    )

    result = _run_rocm_setup(
        "--smoke-test",
        str(cli_model),
        env={
            "LLAMACPP_DIR": str(llamacpp_dir),
            "LLAMACPP_SMOKE_MODEL": str(env_model),
            "LLAMACPP_SMOKE_TIMEOUT": "1",
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"model:   {cli_model}" in result.stdout
    assert str(env_model) not in output
