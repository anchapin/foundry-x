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
    assert "usage: rocm_setup.sh [--check-rocm] [--smoke-test <gguf>]" in result.stdout
    assert "--smoke-test <gguf>" in result.stdout
    assert "--check-rocm" in result.stdout


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

    # The build path now runs the ROCm pre-flight gate (issue #210) before
    # touching git/cmake. Satisfy all four checks with a fake /opt/rocm tree
    # so the smoke-test path reaches the build the way a real host would.
    fake_rocm = tmp_path / "rocm"
    (fake_rocm / ".info").mkdir(parents=True)
    (fake_rocm / "bin").mkdir(parents=True)
    (fake_rocm / ".info" / "version").write_text("6.2.0\n", encoding="utf-8")
    _write_executable(
        fake_rocm / "bin" / "rocminfo",
        "#!/usr/bin/env bash\ncat <<'EOF'\n  Name:                    gfx1032\nEOF\n",
    )
    amdgpu_probe = tmp_path / "sys_module_amdgpu"
    amdgpu_probe.mkdir()
    kfd_probe = tmp_path / "dev_kfd"
    kfd_probe.touch()

    result = _run_rocm_setup(
        "--smoke-test",
        str(cli_model),
        env={
            "LLAMACPP_DIR": str(llamacpp_dir),
            "LLAMACPP_SMOKE_MODEL": str(env_model),
            "LLAMACPP_SMOKE_TIMEOUT": "1",
            "ROCM_PATH": str(fake_rocm),
            "AMDGPU_PROBE": str(amdgpu_probe),
            "KFD_PROBE": str(kfd_probe),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert f"model:   {cli_model}" in result.stdout
    assert str(env_model) not in output


# --- SHA256 verification tests (issue #284) ---


def test_rocm_setup_sha256_match_passes(tmp_path: Path) -> None:
    """When LLAMACPP_MODEL_SHA256 matches the file, smoke test proceeds."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    llamacpp_dir = tmp_path / "llama.cpp"
    server_bin = llamacpp_dir / "build" / "bin" / "llama-server"
    server_bin.parent.mkdir(parents=True)
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake model content", encoding="utf-8")

    # Fake sha256sum returns a fixed hash; we pass the same hash as env.
    fake_hash = "a" * 64
    _write_executable(fake_bin / "git", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "cmake", "#!/usr/bin/env bash\nexit 0\n")
    _write_executable(fake_bin / "nproc", "#!/usr/bin/env bash\nprintf '1\\n'\n")
    _write_executable(
        fake_bin / "sha256sum",
        f"#!/usr/bin/env bash\nprintf '{fake_hash}  %s\\n' \"$1\"\n",
    )
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

    fake_rocm = tmp_path / "rocm"
    (fake_rocm / ".info").mkdir(parents=True)
    (fake_rocm / "bin").mkdir(parents=True)
    (fake_rocm / ".info" / "version").write_text("6.2.0\n", encoding="utf-8")
    _write_executable(
        fake_rocm / "bin" / "rocminfo",
        "#!/usr/bin/env bash\ncat <<'EOF'\n  Name:                    gfx1032\nEOF\n",
    )
    amdgpu_probe = tmp_path / "sys_module_amdgpu"
    amdgpu_probe.mkdir()
    kfd_probe = tmp_path / "dev_kfd"
    kfd_probe.touch()

    result = _run_rocm_setup(
        "--smoke-test",
        str(model_file),
        env={
            "LLAMACPP_DIR": str(llamacpp_dir),
            "LLAMACPP_MODEL_SHA256": fake_hash,
            "LLAMACPP_SMOKE_TIMEOUT": "1",
            "ROCM_PATH": str(fake_rocm),
            "AMDGPU_PROBE": str(amdgpu_probe),
            "KFD_PROBE": str(kfd_probe),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    assert "sha256:" in result.stdout
    assert "verified" in result.stdout


def test_rocm_setup_sha256_mismatch_fails(tmp_path: Path) -> None:
    """When LLAMACPP_MODEL_SHA256 mismatches, script exits non-zero."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    llamacpp_dir = tmp_path / "llama.cpp"
    llamacpp_dir.mkdir()
    server_bin = llamacpp_dir / "build" / "bin" / "llama-server"
    server_bin.parent.mkdir(parents=True)
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake model content", encoding="utf-8")

    # Fake sha256sum returns a fixed hash that won't match the expected one.
    fake_hash = "b" * 64
    _write_executable(
        fake_bin / "sha256sum",
        f"#!/usr/bin/env bash\nprintf '{fake_hash}  %s\\n' \"$1\"\n",
    )
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

    fake_rocm = tmp_path / "rocm"
    (fake_rocm / ".info").mkdir(parents=True)
    (fake_rocm / "bin").mkdir(parents=True)
    (fake_rocm / ".info" / "version").write_text("6.2.0\n", encoding="utf-8")
    _write_executable(
        fake_rocm / "bin" / "rocminfo",
        "#!/usr/bin/env bash\ncat <<'EOF'\n  Name:                    gfx1032\nEOF\n",
    )
    amdgpu_probe = tmp_path / "sys_module_amdgpu"
    amdgpu_probe.mkdir()
    kfd_probe = tmp_path / "dev_kfd"
    kfd_probe.touch()

    result = _run_rocm_setup(
        "--smoke-test",
        str(model_file),
        env={
            "LLAMACPP_DIR": str(llamacpp_dir),
            "LLAMACPP_MODEL_SHA256": "0000000000000000000000000000000000000000000000000000000000000000",
            "LLAMACPP_SMOKE_TIMEOUT": "1",
            "ROCM_PATH": str(fake_rocm),
            "AMDGPU_PROBE": str(amdgpu_probe),
            "KFD_PROBE": str(kfd_probe),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    output = result.stdout + result.stderr
    assert result.returncode != 0, output
    assert "SHA256 mismatch" in result.stderr
    assert "expected:" in result.stderr
    assert "actual:" in result.stderr


def test_rocm_setup_empty_sha256_skips_verification(tmp_path: Path) -> None:
    """When LLAMACPP_MODEL_SHA256 is empty, verification is skipped."""
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    llamacpp_dir = tmp_path / "llama.cpp"
    server_bin = llamacpp_dir / "build" / "bin" / "llama-server"
    server_bin.parent.mkdir(parents=True)
    model_file = tmp_path / "model.gguf"
    model_file.write_text("fake model content", encoding="utf-8")

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

    fake_rocm = tmp_path / "rocm"
    (fake_rocm / ".info").mkdir(parents=True)
    (fake_rocm / "bin").mkdir(parents=True)
    (fake_rocm / ".info" / "version").write_text("6.2.0\n", encoding="utf-8")
    _write_executable(
        fake_rocm / "bin" / "rocminfo",
        "#!/usr/bin/env bash\ncat <<'EOF'\n  Name:                    gfx1032\nEOF\n",
    )
    amdgpu_probe = tmp_path / "sys_module_amdgpu"
    amdgpu_probe.mkdir()
    kfd_probe = tmp_path / "dev_kfd"
    kfd_probe.touch()

    result = _run_rocm_setup(
        "--smoke-test",
        str(model_file),
        env={
            "LLAMACPP_DIR": str(llamacpp_dir),
            "LLAMACPP_MODEL_SHA256": "",  # Empty = skip verification
            "LLAMACPP_SMOKE_TIMEOUT": "1",
            "ROCM_PATH": str(fake_rocm),
            "AMDGPU_PROBE": str(amdgpu_probe),
            "KFD_PROBE": str(kfd_probe),
            "PATH": f"{fake_bin}:{os.environ['PATH']}",
        },
    )

    output = result.stdout + result.stderr
    assert result.returncode == 0, output
    # Should not contain sha256 verification output
    assert "sha256:" not in result.stdout
    assert "verified" not in result.stdout
    assert "SHA256 mismatch" not in result.stderr
