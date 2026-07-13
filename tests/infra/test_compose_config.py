"""Canonical Compose-spec validation of the sandbox compose files (issue #285).

tests/test_compose_sandbox.py and tests/test_compose_rocm.py parse the
compose files with ``yaml.safe_load`` and assert on individual keys.
That catches missing keys but NOT Compose-spec schema violations: a
malformed ``deploy.resources`` shape, a bad ``tmpfs`` option, or a
service referencing an undefined network would all pass
``yaml.safe_load`` yet fail at ``docker compose up``.

The canonical validator is ``docker compose config --quiet``, which the
repo never invoked. This module runs it against:

  1. the base file alone (``docker-compose.yml``), and
  2. the merged base + ROCm override (``docker-compose.yml`` +
     ``docker-compose.rocm.yml``).

The merged validation is the one that matters in practice: the override
is an additive fragment designed to be merged with the base, and on its
own it is not a valid Compose project (the ``foundryx-runner`` service has no
``image``/``build`` until merged). Validating each file in isolation is
therefore necessary but not sufficient — the merged config is what
``docker compose up`` actually consumes.

Both tests skip (not fail) when the ``docker`` binary or the Compose v2
plugin is unavailable, mirroring the docker-gated skip precedent in
tests/infra/test_docker_build.py:15-19. This keeps the default
``uv run pytest`` run in ci.yml green on machines without Docker while
still catching schema regressions wherever Docker is present.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
COMPOSE_DIR = ROOT / "infra" / "docker"
BASE_COMPOSE = COMPOSE_DIR / "docker-compose.yml"
ROCM_COMPOSE = COMPOSE_DIR / "docker-compose.rocm.yml"


def _docker_compose() -> str:
    """Return the ``docker`` binary path, skipping the test if Docker is absent.

    Two skip cases, both non-fatal:
      * ``docker`` is not on PATH at all (e.g. CI without Docker).
      * ``docker`` exists but the Compose v2 plugin (``docker compose``)
        is not installed — the issue specifies the ``docker compose``
        subcommand, so we need the plugin specifically.
    """
    docker = shutil.which("docker")
    if docker is None:
        pytest.skip("docker binary is not on PATH; skipping compose config validation")
    probe = subprocess.run(
        [docker, "compose", "version"],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    if probe.returncode != 0:
        pytest.skip("docker compose v2 plugin is unavailable; skipping compose config validation")
    return docker


def _compose_config(docker: str, compose_files: list[Path]) -> subprocess.CompletedProcess:
    """Run ``docker compose -f ... config --quiet`` and return the completed process.

    ``cwd`` is the compose directory so the relative paths inside the
    compose files (``build.context: ../..``, ``env_file: ../../.env``)
    resolve exactly as they do in a real run. ``check=False`` lets the
    caller assert on ``returncode`` rather than raising.
    """
    cmd: list[str] = [docker, "compose"]
    for f in compose_files:
        cmd.extend(["-f", f.name])
    cmd.extend(["config", "--quiet"])
    proc = subprocess.run(
        cmd,
        cwd=COMPOSE_DIR,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    # Stash the command on the result so failure assertions can echo the exact
    # invocation that failed (a CI reader otherwise only has stdout/stderr).
    proc.cmd = cmd  # type: ignore[attr-defined]
    return proc


def test_base_compose_config_is_valid() -> None:
    """The CPU-only base compose file must pass ``docker compose config``.

    This is the Compose-spec schema check that ``yaml.safe_load`` in
    test_compose_sandbox.py cannot perform: it catches malformed
    ``deploy.resources`` shapes, bad ``tmpfs`` options, dangling network
    references, and any other structural violation that would fail at
    ``docker compose up``.
    """
    docker = _docker_compose()
    assert BASE_COMPOSE.exists(), f"missing base compose file: {BASE_COMPOSE}"
    result = _compose_config(docker, [BASE_COMPOSE])
    assert result.returncode == 0, (
        f"{' '.join(result.cmd)} failed (Compose-spec violation):\n"  # type: ignore[attr-defined]
        f"{result.stdout}{result.stderr}"
    )


def test_merged_base_and_rocm_override_config_is_valid() -> None:
    """The merged base + ROCm override config must pass ``docker compose config``.

    Issue #285 acceptance criterion 1 verbatim: run
    ``docker compose -f docker-compose.yml -f docker-compose.rocm.yml
    config --quiet``. The merged config — not either file in isolation
    — is what ``docker compose up`` consumes, and it is the only place
    a merge-time error (e.g. the override shadowing a base key into an
    invalid shape) would surface. The override alone is intentionally
    not a valid standalone project, so this merged check is essential.
    """
    docker = _docker_compose()
    assert BASE_COMPOSE.exists(), f"missing base compose file: {BASE_COMPOSE}"
    assert ROCM_COMPOSE.exists(), f"missing override file: {ROCM_COMPOSE}"
    result = _compose_config(docker, [BASE_COMPOSE, ROCM_COMPOSE])
    assert result.returncode == 0, (
        f"{' '.join(result.cmd)} failed (Compose-spec violation in the "  # type: ignore[attr-defined]
        f"merged config):\n"
        f"{result.stdout}{result.stderr}"
    )
