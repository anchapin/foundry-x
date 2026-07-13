"""Automated GPU sandbox smoke test for the ROCm override (issue #211).

infra/docker/README.md:107-139 documents a manual smoke test that
confirms the GPU passthrough override works end-to-end:

    docker compose \\
        -f infra/docker/docker-compose.yml \\
        -f infra/docker/docker-compose.rocm.yml \\
        run --rm \\
        --entrypoint curl foundryx http://llamacpp:8080/health

Without an automated equivalent, the override can silently regress --
a missing ``group_add`` entry, an ``HSA_OVERRIDE_GFX_VERSION`` typo, a
``/dev/dri`` device removal -- and the operator finds out at the first
real benchmark run, hours into Phase-3 work.
``tests/test_compose_rocm.py`` only parses the YAML statically.

These tests are gated on ``LLAMACPP_E2E=1`` so they only run when an
operator explicitly opts in (requires a live llama-server + GPU on the
host).  CI runners have no GPU, so these tests are skipped in CI.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = ROOT / "infra" / "docker"
BASE_COMPOSE = COMPOSE_DIR / "docker-compose.yml"
ROCM_COMPOSE = COMPOSE_DIR / "docker-compose.rocm.yml"
GATE_ENV = "LLAMACPP_E2E"

pytestmark = pytest.mark.skipif(
    os.environ.get(GATE_ENV) != "1",
    reason=(
        f"Set {GATE_ENV}=1 to run GPU sandbox e2e tests "
        "(requires live llama-server + AMD GPU on the host)"
    ),
)


def _compose_flags() -> list[str]:
    """Same compose file pair as the manual smoke test (README.md:115-119)."""
    return ["-f", str(BASE_COMPOSE), "-f", str(ROCM_COMPOSE)]


def _dump_compose_config() -> str:
    """Render the merged compose config (base + override) for diagnostics.

    Called on test failure so the operator can inspect what Docker Compose
    actually merged, not just what the YAML files say on disk.  Never
    raises -- a config-dump failure is surfaced as a diagnostic string so
    the original test failure remains visible.
    """
    try:
        result = subprocess.run(
            ["docker", "compose", *_compose_flags(), "config"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        return f"--- docker compose config: {type(exc).__name__}: {exc} ---"
    parts: list[str] = ["--- docker compose config (merged) ---"]
    if result.stdout:
        parts.append(result.stdout)
    if result.stderr:
        parts.append(f"[stderr]\n{result.stderr}")
    return "\n".join(parts)


def _assert_compose_files_exist() -> None:
    """Fail fast if either compose file is missing."""
    assert BASE_COMPOSE.exists(), f"missing base compose file: {BASE_COMPOSE}"
    assert ROCM_COMPOSE.exists(), f"missing override compose file: {ROCM_COMPOSE}"


def test_health_endpoint_reachable() -> None:
    """Run the same curl smoke test as README.md:114-120.

    ``docker compose -f base -f rocm run --rm \\
        --entrypoint curl foundryx http://llamacpp:8080/health``

    Expect a JSON body containing ``ok``.  On failure, dumps the merged
    compose config and the full curl response so the operator can
    diagnose whether the regression is in the override file, the
    llama-server, or the network alias.
    """
    _assert_compose_files_exist()
    cmd = [
        "docker",
        "compose",
        *_compose_flags(),
        "run",
        "--rm",
        "--entrypoint",
        "curl",
        "foundryx",
        "--connect-timeout",
        "10",
        "--max-time",
        "30",
        "http://llamacpp:8080/health",
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"curl /health timed out after 120 s\n{_dump_compose_config()}")
    if result.returncode != 0:
        pytest.fail(
            f"curl /health failed (exit {result.returncode})\n"
            f"{_dump_compose_config()}\n"
            f"--- curl stdout ---\n{result.stdout}\n"
            f"--- curl stderr ---\n{result.stderr}"
        )
    assert "ok" in result.stdout.lower(), (
        "/health response did not contain 'ok'\n"
        f"{_dump_compose_config()}\n"
        f"--- curl stdout ---\n{result.stdout}\n"
        f"--- curl stderr ---\n{result.stderr}"
    )


def test_hsa_override_gfx_version_honoured() -> None:
    """Assert ``HSA_OVERRIDE_GFX_VERSION=10.3.0`` is set inside the container.

    Runs a container with the merged compose files (base + override) and
    checks the env var is actually present in the container's environment.
    This catches regressions that static YAML parsing cannot: e.g. if an
    ``env_file`` entry silently shadows the override, or if a future
    Compose version changes merge semantics for ``environment`` dicts.
    """
    _assert_compose_files_exist()
    cmd = [
        "docker",
        "compose",
        *_compose_flags(),
        "run",
        "--rm",
        "--entrypoint",
        "sh",
        "foundryx",
        "-c",
        'printf "%s" "$HSA_OVERRIDE_GFX_VERSION"',
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"env var probe timed out after 60 s\n{_dump_compose_config()}")
    if result.returncode != 0:
        pytest.fail(
            f"env var probe failed (exit {result.returncode})\n"
            f"{_dump_compose_config()}\n"
            f"--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    value = result.stdout.strip()
    assert value == "10.3.0", (
        f"HSA_OVERRIDE_GFX_VERSION must be 10.3.0 inside the container; "
        f"got {value!r}\n"
        f"{_dump_compose_config()}"
    )
