"""ROCm pre-flight sanity checks for infra/llama-cpp/rocm_setup.sh (issue #210).

rocm_setup.sh used to jump straight to ``git clone`` and
``cmake -DGGML_HIP=ON`` with no pre-flight checks. On a host missing
ROCm entirely, with ROCm < 5.7, without the amdgpu kernel module, or
with the wrong HSA agent, the build either failed deep inside cmake
with a cryptic HIP error or produced a binary that silently fell back
to CPU at runtime.

Issue #210 adds ``rocm_setup.sh --check-rocm`` plus a pre-build gate.
These tests assert each of the four checks (ROCm version, amdgpu
module, /dev/kfd, gfx agent) fires against a fake /opt/rocm tree in
tmp_path, with no host ROCm required. The script honors the
ROCM_PATH / AMDGPU_PROBE / KFD_PROBE env vars precisely so these
checks are hermetic.
"""

from __future__ import annotations

import os
import shutil
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "infra" / "llama-cpp" / "rocm_setup.sh"


def _bash() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not on PATH; skipping rocm_setup.sh subprocess tests")
    return bash


def _write_exec(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _tools_dir(root: Path) -> Path:
    """A directory containing ONLY tr/grep/sort/cat (symlinked from the
    host) and NO rocminfo. Using this as PATH lets us exercise the
    "rocminfo binary absent" branch hermetically even on a host that
    happens to have a real ROCm install on its PATH."""
    tools = root / "tools"
    tools.mkdir()
    for tool in ("tr", "grep", "sort", "cat"):
        found = shutil.which(tool)
        if found:
            (tools / tool).symlink_to(found)
    return tools


class FakeRocm:
    """A throwaway ROCm install tree for hermetic check_rocm runs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.rocm = root / "opt" / "rocm"
        (self.rocm / ".info").mkdir(parents=True)
        (self.rocm / "bin").mkdir(parents=True)
        self.amdgpu_probe = root / "sys_module_amdgpu"
        self.amdgpu_probe.mkdir()
        self.kfd_probe = root / "dev_kfd"
        self.kfd_probe.touch()

    def set_version(self, version: str) -> None:
        (self.rocm / ".info" / "version").write_text(version, encoding="utf-8")

    def drop_version(self) -> None:
        (self.rocm / ".info" / "version").unlink(missing_ok=True)

    def set_rocminfo(self, gfx_agents: list[str]) -> None:
        # rocminfo prints several "Name:" lines; only the gfx* ones matter.
        # A quoted heredoc keeps the payload verbatim (no shell expansion).
        lines = [f"  Name:                    {g}" for g in gfx_agents]
        lines.append("  Marketing Name: AMD Radeon RX 6600 XT")
        payload = "\n".join(lines)
        _write_exec(
            self.rocm / "bin" / "rocminfo",
            "#!/usr/bin/env bash\ncat <<'ROCMINFO_EOF'\n" + payload + "\nROCMINFO_EOF\n",
        )

    def drop_rocminfo(self) -> None:
        (self.rocm / "bin" / "rocminfo").unlink(missing_ok=True)

    def drop_amdgpu(self) -> None:
        self.amdgpu_probe.rmdir()

    def drop_kfd(self) -> None:
        self.kfd_probe.unlink()

    def env(self, **extra: str) -> dict[str, str]:
        base = os.environ.copy()
        base.pop("LLAMACPP_SMOKE_MODEL", None)
        base.update(
            {
                "ROCM_PATH": str(self.rocm),
                "AMDGPU_PROBE": str(self.amdgpu_probe),
                "KFD_PROBE": str(self.kfd_probe),
                "PATH": base.get("PATH", ""),
            }
        )
        base.update(extra)
        return base


def _healthy_fake(root: Path) -> FakeRocm:
    fake = FakeRocm(root)
    fake.set_version("6.2.0")
    fake.set_rocminfo(["gfx1032"])
    return fake


def _run(args: list[str], env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
        env=env,
    )


# ---------------------------------------------------------------------------
# Issue #210 acceptance: --check-rocm reports all four preconditions and
# exits 0 only when every one holds.
# ---------------------------------------------------------------------------


def test_check_rocm_passes_when_all_preconditions_hold(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env())

    out = result.stdout
    assert result.returncode == 0, out + result.stderr
    assert "ROCm version:" in out and "6.2.0" in out
    assert "amdgpu module:" in out and "loaded" in out
    assert "/dev/kfd:" in out and "present" in out
    assert "gfx agents:" in out and "gfx1032" in out


def test_check_rocm_does_not_build_when_invoked_alone(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 0, result.stdout + result.stderr
    assert "Building llama.cpp" not in result.stdout


def test_check_rocm_reports_one_line_per_check(tmp_path: Path) -> None:
    """Each of the four preconditions gets its own labeled line so an
    operator can see exactly which one failed."""
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env())

    out = result.stdout
    assert "ROCm version:" in out
    assert "amdgpu module:" in out
    assert "/dev/kfd:" in out
    assert "gfx agents:" in out


# ---------------------------------------------------------------------------
# Check 1: ROCm version present and >= ROCM_MIN_VERSION (5.7).
# ---------------------------------------------------------------------------


def test_check_fails_when_rocm_version_file_missing(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.drop_version()
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 1, result.stdout
    assert "ROCm version:" in result.stdout and "MISSING" in result.stdout


def test_check_fails_when_rocm_version_below_minimum(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.set_version("5.6.1")
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 1, result.stdout
    assert "5.6.1" in result.stdout
    assert "FAIL" in result.stdout


def test_check_accepts_version_with_git_suffix(tmp_path: Path) -> None:
    """ROCm's .info/version can carry a git-describe suffix (e.g.
    '6.2.0-12345'); the comparison must still see 6.2.0 >= 5.7."""
    fake = _healthy_fake(tmp_path)
    fake.set_version("5.7.0-12345")
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Check 2: amdgpu kernel module loaded.
# ---------------------------------------------------------------------------


def test_check_fails_when_amdgpu_module_missing(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.drop_amdgpu()
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 1, result.stdout
    assert "amdgpu module:" in result.stdout and "MISSING" in result.stdout


# ---------------------------------------------------------------------------
# Check 3: /dev/kfd present.
# ---------------------------------------------------------------------------


def test_check_fails_when_kfd_missing(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.drop_kfd()
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 1, result.stdout
    assert "/dev/kfd:" in result.stdout and "MISSING" in result.stdout


# ---------------------------------------------------------------------------
# Check 4: rocminfo lists a gfx agent.
# ---------------------------------------------------------------------------


def test_check_fails_when_no_gfx_agent_reported(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.set_rocminfo([])  # rocminfo runs but reports no gfx agent
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 1, result.stdout
    assert "gfx agents:" in result.stdout and "NONE" in result.stdout


def test_check_fails_when_rocminfo_binary_absent(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.drop_rocminfo()
    # Controlled PATH: only tr/grep/sort/cat; no rocminfo anywhere, so the
    # script's PATH fallback cannot find a host rocminfo.
    result = _run(["--check-rocm"], fake.env(PATH=str(_tools_dir(tmp_path))))

    assert result.returncode == 1, result.stdout
    assert "gfx agents:" in result.stdout and "NONE" in result.stdout


# ---------------------------------------------------------------------------
# Acceptance: the main script refuses to build when --check-rocm would
# fail, and prints a one-line hint pointing at the pitfalls doc.
# ---------------------------------------------------------------------------


def test_build_refuses_to_build_when_preconditions_unmet(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.drop_version()  # break exactly one precondition
    result = _run([], fake.env())

    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "refusing to build" in result.stderr
    # The build must never start.
    assert "Building llama.cpp" not in result.stdout
    assert "git clone" not in combined


def test_check_rocm_prints_hint_on_failure(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    fake.drop_kfd()
    result = _run(["--check-rocm"], fake.env())

    assert result.returncode == 1, result.stdout
    assert "hint:" in result.stderr
    assert "ROCm pitfalls" in result.stderr
