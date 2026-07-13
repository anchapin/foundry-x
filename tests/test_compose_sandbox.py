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
    assert "foundryx-runner" in services, "expected a 'foundryx-runner' service"
    return services["foundryx-runner"]


def test_src_and_harness_mounts_are_read_only(compose: dict) -> None:
    """Host source must be bind-mounted :ro (SECURITY.md threat #6)."""
    volumes = _service(compose)["volumes"]
    ro = {v.split(":")[1] for v in volumes if v.endswith(":ro")}
    assert (
        {"/app/src", "/app/harness"} <= ro
    ), "src/ and harness/ must be read-only so a buggy hook cannot write outside the workspace"


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
    """Acceptance #1: read_only: true on the foundryx-runner service.

    Closes the writable-surface gap that issue #6 left open: a buggy or
    malicious hook cannot persist to /root, /etc, or the overlay in
    general. The only writable surface is then the explicit tmpfs mounts
    and the logs/ bind mount.
    """
    svc = _service(compose)
    assert svc.get("read_only") is True, (
        "foundryx-runner service must declare read_only: true (issue #123); "
        "without it the container's root FS is writable and a buggy hook "
        "can persist past the run"
    )


def test_tmpfs_caps_present(compose: dict) -> None:
    """Acceptance #2: explicit tmpfs entries for /tmp, /var/tmp, /run.

    Each entry must carry a size cap so a misbehaving hook cannot exhaust
    host RAM. Mode 1777 matches the FHS defaults for /tmp-style dirs.
    """
    svc = _service(compose)
    assert "tmpfs" in svc, "foundryx-runner service must declare tmpfs: entries"

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
            assert re.search(
                _TMPFS_SIZE_CAP_PATTERN, opts
            ), f"tmpfs {path!r} must declare a size cap (e.g. size=256m); got opts={opts!r}"
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


# ---------------------------------------------------------------------------
# Issue #118: pids_limit + ulimits + tmpfs caps hardening.
# ---------------------------------------------------------------------------


def _tmpfs_entry(compose: dict, mount_path: str) -> tuple[int, int] | None:
    """Return ``(size_mb, mode_octal)`` for ``mount_path`` in tmpfs entries.

    Compose accepts either a list of strings ("/path:size=NNm,mode=NNNN") or
    a dict mapping mount paths to the same fragment. Returns ``None`` if the
    path isn't present.
    """
    svc = _service(compose)
    tmpfs = svc.get("tmpfs")
    if not tmpfs:
        return None
    if isinstance(tmpfs, dict):
        fragment = tmpfs.get(mount_path)
    else:
        match = [t for t in tmpfs if t.startswith(f"{mount_path}:")]
        fragment = match[0] if match else None
    if fragment is None:
        return None
    # Strip the leading "<path>:" so the remaining fragment is "size=NNm,mode=NNNN"
    # without the path segment attaching to a key on the first split.
    inner = fragment.split(":", 1)[1] if ":" in fragment else fragment
    parts: dict[str, str] = {}
    for piece in inner.split(","):
        if "=" not in piece:
            continue
        key, value = piece.split("=", 1)
        parts[key.strip()] = value.strip()
    size_token = parts.get("size", "").rstrip("m").rstrip("M")
    mode_token = parts.get("mode", "")
    if not size_token.isdigit():
        raise AssertionError(f"tmpfs {mount_path}: cannot parse size from {fragment!r}")
    if not mode_token.isdigit():
        raise AssertionError(f"tmpfs {mount_path}: cannot parse mode from {fragment!r}")
    return int(size_token), int(mode_token)


def test_pids_limit_is_set(compose: dict) -> None:
    """Issue #118: a top-level pids_limit on the foundryx-runner service bounds the
    total process count inside the container. Memory + CPU caps alone do not
    stop a fork-bomb (SECURITY.md threat #5)."""
    svc = _service(compose)
    assert (
        int(svc.get("pids_limit", 0)) >= 1
    ), f"pids_limit must be a positive integer; got {svc.get('pids_limit')!r}"


def test_pids_limit_matches_deploy_resources_pids(compose: dict) -> None:
    """Compose Spec requires top-level pids_limit to be consistent with the
    pids attribute under deploy.resources.limits. Issue #118 sets both to 256
    so swarm and local compose paths honour the same cap."""
    svc = _service(compose)
    top_level = int(svc.get("pids_limit", 0))
    deploy_pids = int(svc.get("deploy", {}).get("resources", {}).get("limits", {}).get("pids", 0))
    assert top_level > 0 and deploy_pids > 0, (
        f"both pids_limit and deploy.resources.limits.pids must be set; "
        f"got top_level={top_level}, deploy={deploy_pids}"
    )
    assert top_level == deploy_pids, (
        f"pids_limit ({top_level}) must match deploy.resources.limits.pids "
        f"({deploy_pids}) per Compose Spec"
    )


def test_ulimits_cap_process_and_fd_use(compose: dict) -> None:
    """Issue #118: ulimits cap per-process thread count (nproc) and open-file
    count (nofile) so a runaway hook cannot multiply its resource use inside
    one process."""
    svc = _service(compose)
    ulimits = svc.get("ulimits") or {}
    assert (
        "nproc" in ulimits and int(ulimits["nproc"]) > 0
    ), f"ulimits.nproc must be set; got {ulimits!r}"
    assert (
        "nofile" in ulimits and int(ulimits["nofile"]) > 0
    ), f"ulimits.nofile must be set; got {ulimits!r}"


def test_tmpfs_per_path_sizes(compose: dict) -> None:
    """Issue #118 sizes tmpfs per-path: /tmp busiest at 512m, /var/tmp at 256m,
    /run at 64m (PID files only)."""
    assert _tmpfs_entry(compose, "/tmp") == (
        512,
        1777,
    ), f"tmpfs /tmp must be 512m mode 1777; got {_tmpfs_entry(compose, '/tmp')!r}"
    assert _tmpfs_entry(compose, "/var/tmp") == (
        256,
        1777,
    ), f"tmpfs /var/tmp must be 256m mode 1777; got {_tmpfs_entry(compose, '/var/tmp')!r}"
    assert _tmpfs_entry(compose, "/run") == (
        64,
        1777,
    ), f"tmpfs /run must be 64m mode 1777 (PID files only); got {_tmpfs_entry(compose, '/run')!r}"
