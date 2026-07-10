"""Dockerfile build validation for infra/docker/Dockerfile."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = ROOT / "infra" / "docker" / "Dockerfile"


def _docker_binary() -> str:
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker binary is not on PATH; skipping Dockerfile build check")
    return docker


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


def test_dockerfile_build_configuration_is_valid(tmp_path: Path) -> None:
    docker = _docker_binary()
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
