"""Static validation of the ROCm sandbox override (issue #115).

docs/SECURITY.md:54-59 mandates running benchmarks and evolution inside
a Docker container. The CPU-only compose path added by issue #6 covers
read-only mounts, tmpfs caps, and capability drops, but exposes no GPU
devices -- operators who want to evolve the harness against the local
ROCm-built llama-server (infra/llama-cpp) had to leave the sandbox.

Issue #115 ships `infra/docker/docker-compose.rocm.yml` as an *override*
file that re-attaches the host AMD GPU. These tests guard the override
so the acceptance criteria cannot regress silently. They parse the YAML
only (no Docker daemon required), mirroring the style of
test_compose_sandbox.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

COMPOSE_DIR = Path(__file__).resolve().parents[1] / "infra" / "docker"
BASE_COMPOSE = COMPOSE_DIR / "docker-compose.yml"
ROCM_COMPOSE = COMPOSE_DIR / "docker-compose.rocm.yml"


@pytest.fixture(scope="module")
def override() -> dict:
    assert ROCM_COMPOSE.exists(), f"missing override file: {ROCM_COMPOSE}"
    with ROCM_COMPOSE.open() as fh:
        return yaml.safe_load(fh)


@pytest.fixture(scope="module")
def base_text() -> str:
    assert BASE_COMPOSE.exists(), f"missing base compose file: {BASE_COMPOSE}"
    return BASE_COMPOSE.read_text(encoding="utf-8")


def _service(compose: dict) -> dict:
    services = compose["services"]
    assert "foundryx" in services, "override must declare a 'foundryx' service"
    return services["foundryx"]


def _device_names(svc: dict) -> list[str]:
    """Return the host-side device paths declared on a service.

    Compose accepts both short syntax (list of strings) and long syntax
    (dict). For short syntax, entries can be either "/dev/kfd" (bind to
    the same path) or "/dev/kfd:/dev/foo" (remap to a container path).
    We compare against the host side, before the optional ":".
    """
    devices = svc.get("devices", [])
    if isinstance(devices, dict):
        return list(devices.keys())
    names: list[str] = []
    for entry in devices:
        if isinstance(entry, str):
            names.append(entry.split(":", 1)[0])
        else:
            # Long-form dict: {"source": "...", "target": "..."}
            names.append(str(entry.get("source", "")))
    return names


# ---------------------------------------------------------------------------
# Issue #115 acceptance criteria: override adds devices + 2 env vars, and
# the rest is inherited from the base file via Compose merge-by-service.
# ---------------------------------------------------------------------------


def test_override_uses_same_service_name_as_base(override: dict) -> None:
    """The override must target the same service (`foundryx`) so that
    `docker compose -f base -f override` merges the ROCm bits into the
    base service rather than declaring a duplicate."""
    assert "foundryx" in override.get("services", {}), (
        "override must declare the 'foundryx' service so "
        "`docker compose -f docker-compose.yml -f docker-compose.rocm.yml` "
        "merges the ROCm bits via Compose v2's merge-by-service"
    )


def test_devices_passthrough_kfd_and_dri(override: dict) -> None:
    """Acceptance: devices: [/dev/kfd, /dev/dri].

    /dev/kfd is the Kernel Fusion Driver (HSA compute entry point);
    /dev/dri is the Direct Rendering Manager root (the ROCm runtime
    submits work via /dev/dri/renderD*). Without these, in-container
    ROCm cannot reach the host GPU.
    """
    names = _device_names(_service(override))
    assert "/dev/kfd" in names, f"devices must include /dev/kfd; got {names!r}"
    assert "/dev/dri" in names, f"devices must include /dev/dri; got {names!r}"


def test_group_add_unlocks_render_nodes(override: dict) -> None:
    """/dev/dri/renderD* is gated by group membership.

    Device cgroup rules from `devices:` alone do NOT grant the right to
    open the node -- the container process must also be in the `video`
    and `render` groups. Without these, llama.cpp falls back to CPU
    silently. This is the single most common "GPU passthrough doesn't
    work" pitfall on Linux.
    """
    svc = _service(override)
    group_add = svc.get("group_add", [])
    assert (
        "video" in group_add
    ), f"group_add must include 'video' to access /dev/dri/renderD*; got {group_add!r}"
    assert (
        "render" in group_add
    ), f"group_add must include 'render' to access /dev/dri/renderD*; got {group_add!r}"


def test_hsa_override_gfx_version_for_rx_6600_xt(override: dict) -> None:
    """Acceptance: HSA_OVERRIDE_GFX_VERSION=10.3.0.

    The RX 6600 XT reports as gfx1032 but HSA sometimes rejects the
    agent on older kernels with `agent refused`; the override forces
    the version string HSA uses for capability negotiation. Matches
    infra/llama-cpp/README.md "ROCm pitfalls".
    """
    env = _service(override).get("environment", {})
    assert env.get("HSA_OVERRIDE_GFX_VERSION") == "10.3.0", (
        f"HSA_OVERRIDE_GFX_VERSION must be 10.3.0 for the RX 6600 XT; "
        f"got {env.get('HSA_OVERRIDE_GFX_VERSION')!r}"
    )


def test_rocm_path_points_at_host_install(override: dict) -> None:
    """Acceptance: ROCM_PATH=/opt/rocm.

    The standard ROCm install path; lets in-container tooling that
    probes the runtime find the host install. ROCm users build
    llama.cpp against this path on the host (infra/llama-cpp/rocm_setup.sh).
    """
    env = _service(override).get("environment", {})
    assert (
        env.get("ROCM_PATH") == "/opt/rocm"
    ), f"ROCM_PATH must be /opt/rocm; got {env.get('ROCM_PATH')!r}"


# ---------------------------------------------------------------------------
# Guard rails: the override must not silently undo base-file hardening or
# the CPU-only default path.
# ---------------------------------------------------------------------------


def test_override_does_not_shadow_base_hardening(override: dict) -> None:
    """The override MUST NOT redeclare any base-file hardening key.

    Compose v2 merges services by name. For most top-level keys
    (`volumes`, `tmpfs`, `read_only`, `cap_drop`, `pids_limit`,
    `ulimits`, `networks`, `extra_hosts`, `security_opt`) the override
    value REPLACES the base value; for mappings like `environment` and
    `labels` the override merges by key. If the override silently
    redeclared `cap_drop: ALL` it would still be inherited, but a copy
    that forgot to include `cap_drop` (or that set it to `[]`) would
    silently relax the hardening -- a regression that is hard to spot
    in review. This test forbids the override from declaring any
    base-file key at all; the override's job is purely additive.
    """
    svc = _service(override)
    base_owned = {
        "cap_drop",
        "cap_add",
        "read_only",
        "tmpfs",
        "volumes",
        "networks",
        "extra_hosts",
        "pids_limit",
        "ulimits",
        "security_opt",
        "deploy",
        "build",
        "image",
        "init",
        "env_file",
    }
    leaked = sorted(base_owned & set(svc.keys()))
    assert not leaked, (
        f"override must NOT redeclare base-file keys {leaked!r}; "
        f"doing so risks shadowing the base values via Compose's "
        f"merge-by-service. Override top-level keys: {sorted(svc.keys())!r}"
    )


def test_base_compose_is_unchanged(base_text: str) -> None:
    """Issue #115 keeps the CPU-only default compose file unchanged.

    The acceptance criteria are explicit: "no change to the CPU-only
    default compose file". We spot-check the structural guardrails that
    issue #6 / #123 / #118 added so a later edit can't silently relax
    the CPU path while touching the override. This is a textual check
    because the base file has no machine-readable `version` field --
    the structural facts are the contract.
    """
    assert "read_only: true" in base_text, (
        "base compose must keep read_only: true (issue #123); the ROCm "
        "override must not relax this on the CPU-only path"
    )
    assert "cap_drop:" in base_text and "ALL" in base_text, (
        "base compose must keep cap_drop: ALL (SECURITY.md threat #6); "
        "the override must not relax this on the CPU-only path"
    )
    assert "tmpfs:" in base_text, "base compose must keep tmpfs caps (issue #123)"
    assert "pids_limit:" in base_text, "base compose must keep pids_limit (issue #118)"
    # Negative check: the base file must not have grown a `devices` block
    # (the GPU passthrough is exclusive to the override).
    assert "devices:" not in base_text, (
        "base compose must NOT add a `devices:` block; GPU passthrough "
        "is exclusive to docker-compose.rocm.yml (issue #115)"
    )
    assert "HSA_OVERRIDE_GFX_VERSION" not in base_text, (
        "base compose must NOT set HSA_OVERRIDE_GFX_VERSION; that env "
        "is exclusive to the ROCm override (issue #115)"
    )
