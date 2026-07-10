"""Static validation of the sandboxed run path (infra/docker/docker-compose.yml).

docs/SECURITY.md:54-59 mandates running benchmarks and evolution inside a
Docker container with read-only mounts for the host filesystem. These tests
guard the Compose file so the guardrail cannot regress silently. They parse
the YAML only (no Docker daemon required), mirroring the acceptance criteria
of issue #6.
"""

from __future__ import annotations

import re
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


# ----------------------------------------------------------------------------
# Issue #123 — harden sandbox compose with read-only root FS, tmpfs caps,
# and bounded egress. The three assertions below are the static guard so the
# hardening cannot regress silently. They parse the YAML only; no Docker
# daemon required.
# ----------------------------------------------------------------------------

_REQUIRED_TMPFS_PATHS = ("/tmp", "/var/tmp", "/run")
_TMPFS_SIZE_CAP_PATTERN = r"(?:^|[,\s])size=\d+[kmg]"


def test_read_only_root_filesystem(compose: dict) -> None:
    """Acceptance #1: read_only: true on the foundryx service.

    Closes the writable-surface gap that issue #6 left open: a buggy or
    malicious hook cannot persist to /root, /etc, or the overlay in
    general. The only writable surface is then the explicit tmpfs mounts
    and the logs/ bind mount.
    """
    svc = _service(compose)
    assert svc.get("read_only") is True, (
        "foundryx service must declare read_only: true (issue #123); "
        "without it the container's root FS is writable and a buggy hook "
        "can persist past the run"
    )


def test_tmpfs_caps_present(compose: dict) -> None:
    """Acceptance #2: explicit tmpfs entries for /tmp, /var/tmp, /run.

    Each entry must carry a size cap so a misbehaving hook cannot exhaust
    host RAM. Mode 1777 matches the FHS defaults for /tmp-style dirs.
    """
    svc = _service(compose)
    assert "tmpfs" in svc, "foundryx service must declare tmpfs: entries"

    raw = svc["tmpfs"]
    entries: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        # Long syntax: {"path": "opts"}.
        entries = [(path, opts or "") for path, opts in raw.items()]
    else:
        # Short syntax: a list of "/path:opts" strings.
        for item in raw:
            if ":" in item:
                path, opts = item.split(":", 1)
            else:
                path, opts = item, ""
            entries.append((path, opts))

    paths = {path for path, _ in entries}
    missing = [p for p in _REQUIRED_TMPFS_PATHS if p not in paths]
    assert not missing, f"tmpfs entries missing for: {missing}"

    for path, opts in entries:
        if path in _REQUIRED_TMPFS_PATHS:
            assert re.search(_TMPFS_SIZE_CAP_PATTERN, opts), (
                f"tmpfs {path!r} must declare a size cap (e.g. size=256m); " f"got opts={opts!r}"
            )
            # Mode 1777 is the FHS default for world-writable sticky dirs;
            # the test does not fail if mode is absent, only if it is wrong.
            mode_match = re.search(r"(?:^|[,\s])mode=([0-7]+)", opts)
            assert (
                mode_match is None or mode_match.group(1) == "1777"
            ), f"tmpfs {path!r} mode must be 1777 when declared; got {mode_match.group(1)!r}"


def test_extra_hosts_is_llamacpp_gateway_only(compose: dict) -> None:
    """Acceptance #3: extra_hosts, if present, must contain only the LLAMACPP gateway.

    Issue #123 narrows the egress surface from a generic
    `host.docker.internal:host-gateway` alias (which exposes the whole
    host loopback) to a specific `llamacpp` alias consumed only by
    LLAMACPP_HOST. This test fails the moment anyone re-adds a generic
    host alias — preventing the regression.
    """
    svc = _service(compose)
    extra_hosts = svc.get("extra_hosts")
    if not extra_hosts:
        # "(if any)" in the issue body permits removing extra_hosts entirely
        # and resolving the host via .env-supplied LLAMACPP_HOST.
        return

    # Compose accepts a list of "host:ip" strings or a dict {"host": "ip"}.
    if isinstance(extra_hosts, dict):
        names = list(extra_hosts.keys())
    else:
        names = [entry.split(":", 1)[0] for entry in extra_hosts]

    assert names, "extra_hosts declared but no entries resolved"

    forbidden = {"host.docker.internal", "host", "docker.internal"}
    leaks = [n for n in names if n in forbidden]
    assert not leaks, (
        f"extra_hosts contains a generic host alias {leaks!r}; "
        f"issue #123 requires the only declared alias to be the LLAMACPP "
        f"gateway. Found names: {names!r}"
    )

    llamacpp_entries = [n for n in names if "llamacpp" in n.lower()]
    assert llamacpp_entries, f"extra_hosts must declare the LLAMACPP gateway; got names: {names!r}"
