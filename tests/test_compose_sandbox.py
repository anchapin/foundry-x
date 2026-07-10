"""Static validation of the sandboxed run path (infra/docker/docker-compose.yml).

docs/SECURITY.md:54-59 mandates running benchmarks and evolution inside a
Docker container with read-only mounts for the host filesystem. These tests
guard the Compose file so the guardrail cannot regress silently. They parse
the YAML only (no Docker daemon required), mirroring the acceptance criteria
of issue #6.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[1] / "infra" / "docker" / "docker-compose.yml"


@pytest.fixture(scope="module")
def compose() -> dict:
    assert COMPOSE.exists(), f"missing compose file: {COMPOSE}"
    with COMPOSE.open() as fh:
        return yaml.safe_load(fh)


def _service(compose: dict) -> dict:
    services = compose["services"]
    assert "foundryx" in services, "expected a 'foundryx' service"
    return services["foundryx"]


def test_src_and_harness_mounts_are_read_only(compose: dict) -> None:
    """Host source must be bind-mounted :ro (SECURITY.md threat #6)."""
    volumes = _service(compose)["volumes"]
    ro = {v.split(":")[1] for v in volumes if v.endswith(":ro")}
    assert {"/app/src", "/app/harness"} <= ro, (
        "src/ and harness/ must be read-only so a buggy hook cannot write " "outside the workspace"
    )


def test_logs_mount_is_writable(compose: dict) -> None:
    """logs/ is the only read-write host mount (TraceLogger, ADR-0003)."""
    volumes = _service(compose)["volumes"]
    logs_mounts = [v for v in volumes if v.split(":")[1] == "/app/logs"]
    assert logs_mounts, "logs/ must be mounted so traces reach the host"
    assert not logs_mounts[0].endswith(":ro"), "logs/ must be writable"


def test_no_other_writable_host_mounts(compose: dict) -> None:
    """No host bind mount other than logs/ may be writable."""
    volumes = _service(compose)["volumes"]
    writable = [v for v in volumes if not v.endswith(":ro") and v.split(":")[1] != "/app/logs"]
    # Named volumes (driver-prefixed) are acceptable; host bind paths (../..)
    # that are writable are not. Only logs/ may be a writable bind mount.
    bad = [v for v in writable if v.startswith("../..")]
    assert not bad, f"unexpected writable host mount(s): {bad}"


def test_memory_limit_is_set(compose: dict) -> None:
    """A memory cap must contain runaway runs (SECURITY.md threat #5)."""
    limits = _service(compose)["deploy"]["resources"]["limits"]
    assert limits.get("memory"), "deploy.resources.limits.memory must be set"


def test_network_is_not_host_mode(compose: dict) -> None:
    """The container must NOT use broad host networking."""
    svc = _service(compose)
    assert (
        svc.get("network_mode") != "host"
    ), "network_mode: host grants broad host access; use a bridge network"
    # A dedicated (non-default-alias) network should be declared.
    assert "networks" in svc, "service should attach to a dedicated network"


def test_capabilities_dropped(compose: dict) -> None:
    """All Linux capabilities should be dropped (supplementary hardening)."""
    assert "ALL" in _service(compose).get(
        "cap_drop", []
    ), "cap_drop: ALL removes ambient capabilities a buggy hook could abuse"
