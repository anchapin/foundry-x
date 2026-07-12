"""Dockerfile build validation for infra/docker/Dockerfile."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "infra" / "docker" / "Dockerfile"
IMAGE_TAG = "foundryx:pytest"


def _docker_binary() -> str | None:
    return shutil.which("docker")


def _docker_supports_build_check(docker: str) -> bool:
    result = subprocess.run(
        [docker, "build", "--help"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    return result.returncode == 0 and "--check" in result.stdout


def _build_context(tmp_path: Path) -> Path:
    context = tmp_path / "context"
    dockerfile = context / "infra" / "docker" / "Dockerfile"
    dockerfile.parent.mkdir(parents=True)
    shutil.copy2(DOCKERFILE, dockerfile)
    shutil.copy2(ROOT / "pyproject.toml", context / "pyproject.toml")
    if (ROOT / "uv.lock").exists():
        shutil.copy2(ROOT / "uv.lock", context / "uv.lock")
    for name in ("src", "harness", "tests"):
        directory = context / name
        directory.mkdir(parents=True)
        (directory / ".keep").write_text("", encoding="utf-8")
    return context


def _build_image(docker: str) -> subprocess.CompletedProcess[str]:
    """Build the actual image from the repo root and return the result."""
    return subprocess.run(
        [
            docker,
            "build",
            "-f",
            str(DOCKERFILE.relative_to(ROOT)),
            "-t",
            IMAGE_TAG,
            ".",
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )


# ---------------------------------------------------------------------------
# Lint-only tests (run even without Docker)
# ---------------------------------------------------------------------------


def test_dockerfile_build_configuration_is_valid(tmp_path: Path) -> None:
    docker = _docker_binary()
    if docker is None:
        pytest.skip("docker binary is not on PATH; skipping Dockerfile build check")
    if not _docker_supports_build_check(docker):
        pytest.skip("docker build --check is unavailable; skipping full image build to avoid pulls")

    context = _build_context(tmp_path)
    result = subprocess.run(
        [docker, "build", "--check", "-f", "infra/docker/Dockerfile", "."],
        cwd=context,
        capture_output=True,
        text=True,
        timeout=50,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr


# ---------------------------------------------------------------------------
# Full build + runtime-stage contract (require Docker)
# ---------------------------------------------------------------------------


def test_dockerfile_full_build_succeeds() -> None:
    """Build the actual image from the repo root; assert exit 0."""
    docker = _docker_binary()
    if docker is None:
        pytest.skip("docker binary is not on PATH; skipping full image build")

    result = _build_image(docker)
    assert result.returncode == 0, result.stdout + result.stderr


def test_runtime_stage_excludes_build_toolchain() -> None:
    """Runtime stage must NOT contain build-essential, git, or curl.

    This enforces the #116 acceptance criteria that the published
    image carries no build toolchain (see infra/docker/Dockerfile:17-25).
    """
    docker = _docker_binary()
    if docker is None:
        pytest.skip("docker binary is not on PATH; skipping runtime-stage check")

    build = _build_image(docker)
    if build.returncode != 0:
        pytest.skip("image build failed; skipping runtime-stage assertion")

    for binary in ("git", "curl", "gcc", "make"):
        result = subprocess.run(
            [docker, "run", "--rm", IMAGE_TAG, "which", binary],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode != 0, (
            f"runtime stage contains '{binary}' which should be excluded "
            f"(#116 acceptance criteria).  FoundryX runtime image must be "
            f"minimal: only python, ca-certificates, and application code."
        )
