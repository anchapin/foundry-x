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


# ---------------------------------------------------------------------------
# Issue #754 acceptance: GPU responsiveness is checked (5th precondition).
# ---------------------------------------------------------------------------


def test_check_rocm_includes_gpu_responsive_check_in_output(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env(GPU_RESPONSIVE_PROBE="echo ok"))

    out = result.stdout
    assert "GPU responsive:" in out, f"Expected 'GPU responsive:' in output: {out}"


def test_check_rocm_passes_when_gpu_responsive(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env(GPU_RESPONSIVE_PROBE="echo ok"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPU responsive:" in result.stdout and "OK" in result.stdout


def test_check_rocm_fails_when_gpu_not_responsive(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env(GPU_RESPONSIVE_PROBE="false"))

    assert result.returncode == 1, result.stdout
    assert "GPU responsive:" in result.stdout and "FAIL" in result.stdout


def test_check_rocm_fails_when_responsive_probe_times_out(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env(GPU_RESPONSIVE_PROBE="timeout 1 sleep 10"))

    assert result.returncode == 1, result.stdout
    assert "GPU responsive:" in result.stdout and "FAIL" in result.stdout


def test_check_gpu_responsive_flag_passes_when_responsive(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-gpu-responsive"], fake.env(GPU_RESPONSIVE_PROBE="echo ok"))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPU responsive: OK" in result.stdout


def test_check_gpu_responsive_flag_fails_when_not_responsive(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-gpu-responsive"], fake.env(GPU_RESPONSIVE_PROBE="false"))

    assert result.returncode == 1, result.stdout + result.stderr
    assert "GPU responsive: FAIL" in result.stderr


def test_check_rocm_uses_rocminfo_when_gpu_responsive_probe_not_set(
    tmp_path: Path,
) -> None:
    fake = _healthy_fake(tmp_path)
    result = _run(["--check-rocm"], fake.env(GPU_RESPONSIVE_PROBE=""))

    assert result.returncode == 0, result.stdout + result.stderr
    assert "GPU responsive:" in result.stdout and "OK" in result.stdout


# ---------------------------------------------------------------------------
# Issue #815 acceptance: GPU-active inference assertion in smoke test.
# When LLAMACPP_SMOKE_NGL > 0 the smoke test must verify the server log
# contains a ROCm GPU agent identifier; otherwise it exits non-zero.
# ---------------------------------------------------------------------------


class _FakeLlamaServer:
    """A throwaway llama-server that listens on a port and writes a marker
    to stderr so the caller can verify GPU-active inference was asserted.
    Combine with a FakeRocm instance via ``env_from_fake()``."""

    def __init__(self, root: Path, *, rocmmark: str = "ROCm initialized\n") -> None:
        self.root = root
        self.rocmmark = rocmmark
        self.port = 18765
        self.server_dir = root / "llama.cpp"
        self.build_dir = self.server_dir / "build" / "bin"
        self.build_dir.mkdir(parents=True)
        self.model = root / "model.gguf"
        self.model.touch()
        self.log = root / "server.log"

    def write_binary(self) -> None:
        import subprocess as _subprocess

        self.server_dir.mkdir(parents=True, exist_ok=True)
        _subprocess.run(
            ["git", "init", "-q", str(self.server_dir)],
            capture_output=True, text=True, timeout=10,
        )
        _subprocess.run(
            ["git", "config", "user.email", "test@test.test"],
            capture_output=True, text=True, timeout=5,
            cwd=str(self.server_dir),
        )
        _subprocess.run(
            ["git", "config", "user.name", "test"],
            capture_output=True, text=True, timeout=5,
            cwd=str(self.server_dir),
        )
        readme = self.server_dir / "README"
        readme.write_text("mock", encoding="utf-8")
        _subprocess.run(
            ["git", "add", "README"],
            capture_output=True, text=True, timeout=5,
            cwd=str(self.server_dir),
        )
        _subprocess.run(
            ["git", "commit", "-q", "-m", "init"],
            capture_output=True, text=True, timeout=5,
            cwd=str(self.server_dir),
        )
        _subprocess.run(
            ["git", "tag", "b9957", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(self.server_dir),
        )
        _subprocess.run(
            ["git", "remote", "add", "origin", str(self.server_dir)],
            capture_output=True, text=True, timeout=5,
            cwd=str(self.server_dir),
        )

        self.build_dir.mkdir(parents=True, exist_ok=True)
        cmakeLists = self.server_dir / "CMakeLists.txt"
        cmakeLists.write_text(
            "cmake_minimum_required(VERSION 3.16)\n"
            "add_custom_target(llama-server ALL DEPENDS)\n",
            encoding="utf-8",
        )
        self.build_dir.joinpath("bin").mkdir(parents=True, exist_ok=True)
        srvbin = self.build_dir / "bin" / "llama-server"
        srvbin.write_text("#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n", encoding="utf-8")
        srvbin.chmod(srvbin.stat().st_mode | stat.S_IXUSR)

        content = f"""#!/usr/bin/env python3
import os, sys, time, threading, http.server

PORT = {self.port}
MARKER = {self.rocmmark!r}
LOG = {str(self.log)!r}

class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args): pass

    def do_GET(self):
        if self.path == "/health":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{{"ok": true}}')
        elif self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{{"data":[{{"id":"test"}}]}}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        if self.path == "/completion":
            sys.stderr.write(MARKER)
            sys.stderr.flush()
            time.sleep(0.2)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{{"content":"hello"}}')
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(30)
"""
        (self.build_dir / "llama-server").write_text(content, encoding="utf-8")
        (self.build_dir / "llama-server").chmod(
            self.build_dir.stat().st_mode | stat.S_IXUSR
        )

    def env_from_fake(self, fake: FakeRocm, **extra: str) -> dict[str, str]:
        marker_file = self.root / ".gpu_marker"
        marker_file.write_text(self.rocmmark, encoding="utf-8")
        base = fake.env()
        base.update(
            {
                "LLAMACPP_DIR": str(self.server_dir),
                "LLAMACPP_SMOKE_MODEL": str(self.model),
                "LLAMACPP_SMOKE_PORT": str(self.port),
                "SERVER_LOG": str(self.log),
                "LLAMACPP_REF": "HEAD",
                "LLAMACPP_SKIP_BUILD": "1",
                "LLAMACPP_GPU_MARKER_FILE": str(marker_file),
            }
        )
        base.update(extra)
        return base


def _smoke_run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [_bash(), str(SCRIPT), "--smoke-test", env["LLAMACPP_SMOKE_MODEL"]],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
        env=env,
    )


def test_smoke_skips_gpu_assertion_when_ngl_is_zero(tmp_path: Path) -> None:
    fake = _healthy_fake(tmp_path)
    srv = _FakeLlamaServer(tmp_path)
    srv.write_binary()
    result = _smoke_run(srv.env_from_fake(fake))
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Smoke test PASSED" in combined
    assert "no ROCm GPU agent found" not in combined


def test_smoke_fails_gpu_assertion_when_ngl_positive_but_no_rocm_in_log(
    tmp_path: Path,
) -> None:
    fake = _healthy_fake(tmp_path)
    srv = _FakeLlamaServer(tmp_path, rocmmark="CPU mode active\n")
    srv.write_binary()
    result = _smoke_run(srv.env_from_fake(fake, LLAMACPP_SMOKE_NGL="35"))
    combined = result.stdout + result.stderr
    assert result.returncode == 1, combined
    assert "no ROCm GPU agent found" in combined
    assert "ROCm pitfalls" in combined


def test_smoke_passes_gpu_assertion_when_ngl_positive_and_rocm_in_log(
    tmp_path: Path,
) -> None:
    fake = _healthy_fake(tmp_path)
    srv = _FakeLlamaServer(tmp_path, rocmmark="ROCm initialized\namdgpu: gfx1032\n")
    srv.write_binary()
    result = _smoke_run(srv.env_from_fake(fake, LLAMACPP_SMOKE_NGL="35"))
    combined = result.stdout + result.stderr
    assert result.returncode == 0, combined
    assert "Smoke test PASSED" in combined
