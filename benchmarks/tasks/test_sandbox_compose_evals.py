"""Benchmark task: sandbox compose hardening is pinned (SECURITY.md §Sandbox).

Regression target for ``infra/docker/docker-compose.yml``. Issues #118,
#123, and #124 hardened the sandbox container with ``read_only: true``,
per-path tmpfs size caps, a narrowly-scoped ``extra_hosts`` entry, an
explicit ``deploy.resources.limits`` block, ``pids_limit``, ``cap_drop:
ALL``, and ``security_opt: no-new-privileges``. None of those knobs
were pinned by a benchmark: ``tests/test_compose_sandbox.py`` covers
part of the surface but is not tagged ``@pytest.mark.benchmark``, so a
regression that drops ``read_only: true`` or re-adds a generic host
alias would silently re-open SECURITY.md threats #5 (resource
exhaustion) and #6 (local privilege) and pass every benchmark in the
suite.

This task pins every hardening knob enumerated in issue #178. The
file is parsed with :func:`yaml.safe_load` only -- no Docker daemon
required, mirroring the acceptance criteria of issue #6 -- and the
parse + assertion cost stays well within the "<30 s on a developer
laptop" budget that ADR-0005 inherits from the existing suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from benchmarks.models import BenchmarkTask

#: Path to the Compose file under test, resolved relative to the repo root.
COMPOSE = Path(__file__).resolve().parents[2] / "infra" / "docker" / "docker-compose.yml"

#: Per-path tmpfs sizes (in MiB) that issue #118 sizes into the file. The
#: benchmark pins the exact values, not just "a size is set", because the
#: point of the hardening is the *bounded* writable surface: a regression
#: that silently inflates /var/tmp to 8g or drops the cap entirely must
#: turn the benchmark red.
_TMPFS_REQUIRED_SIZES_MB = {
    "/tmp": 512,
    "/var/tmp": 256,
    "/run": 64,
}

#: Compose Spec requires the top-level ``pids_limit`` to be consistent
#: with ``deploy.resources.limits.pids``. Issue #118 sets both to 256 so
#: swarm-mode and local-compose paths honour the same cap.
_EXPECTED_PIDS_LIMIT = 256

#: ``size=NNm`` token, where N is one-or-more digits. The presence of
#: this token is the contract; the actual size is checked against
#: ``_TMPFS_REQUIRED_SIZES_MB`` separately.
_SIZE_CAP_PATTERN = re.compile(r"(?:^|[,\s])size=(\d+)([kmg])")


def _service(compose: dict) -> dict:
    """Return the ``foundryx`` service entry from a parsed Compose file."""
    services = compose["services"]
    assert "foundryx" in services, "expected a 'foundryx' service in docker-compose.yml"
    return services["foundryx"]


def _tmpfs_entries(svc: dict) -> dict[str, str]:
    """Return ``{mount_path: opts}`` for the foundryx service tmpfs block.

    Compose accepts both the long syntax (a dict ``{"/path": "opts"}``)
    and the short syntax (a list of ``"/path:opts"`` strings). The
    existing ``infra/docker/docker-compose.yml`` uses the short syntax;
    this helper normalises both so the benchmark does not regress if
    someone re-shapes the file.
    """
    raw = svc.get("tmpfs")
    assert raw is not None, "foundryx service must declare tmpfs: entries (issue #123)"
    if isinstance(raw, dict):
        return {path: (opts or "") for path, opts in raw.items()}
    entries: dict[str, str] = {}
    for item in raw:
        if ":" in item:
            path, opts = item.split(":", 1)
        else:
            path, opts = item, ""
        entries[path] = opts
    return entries


def _extra_hosts_names(svc: dict) -> list[str]:
    """Return the ordered list of host aliases declared under ``extra_hosts``.

    Compose accepts both a list of ``"host:ip"`` strings and a dict
    ``{"host": "ip"}``. The benchmark only cares about the alias names,
    not their resolved IPs.
    """
    extra_hosts = svc.get("extra_hosts") or {}
    if isinstance(extra_hosts, dict):
        return list(extra_hosts.keys())
    return [entry.split(":", 1)[0] for entry in extra_hosts]


TASK = BenchmarkTask(
    name="sandbox_compose",
    description=(
        "infra/docker/docker-compose.yml pins the sandbox container to "
        "read_only root FS, per-path tmpfs size caps, a narrowly-scoped "
        "llamacpp-only extra_hosts entry, explicit deploy.resources.limits "
        "with pids=256, pids_limit=256, cap_drop=ALL, and "
        "security_opt=no-new-privileges."
    ),
    prompt=(
        "Inspect infra/docker/docker-compose.yml: confirm the foundryx "
        "service still declares read_only: true, tmpfs: entries for "
        "/tmp, /var/tmp, and /run each with an explicit size cap, "
        "extra_hosts containing exactly one entry whose key is "
        "'llamacpp', deploy.resources.limits with memory, cpus, and pids "
        "set, pids_limit equal to deploy.resources.limits.pids, "
        "cap_drop: ALL, and security_opt: no-new-privileges."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "Each hardening knob is present with the value issue #178 "
        "specifies; extra_hosts contains exactly one entry and its key "
        "is 'llamacpp'; every tmpfs entry carries an explicit size= cap "
        "matching the per-path sizes from issue #118."
    ),
    tags=["security", "sandbox"],
)


@pytest.fixture(scope="module")
def compose() -> dict:
    """Parse the Compose file once per module; every test reads the same dict."""
    assert COMPOSE.exists(), f"missing compose file: {COMPOSE}"
    with COMPOSE.open() as fh:
        return yaml.safe_load(fh)


@pytest.mark.benchmark
def test_read_only_root_filesystem(compose: dict) -> None:
    """``read_only: true`` closes the writable-surface gap (issue #123).

    A regression that drops the flag re-opens SECURITY.md threat #6: a
    buggy or malicious hook can persist to /root, /etc, or the overlay
    in general. The only writable surface is then the explicit tmpfs
    mounts and the logs/ bind mount; without ``read_only`` the cap is
    silently removed.
    """
    svc = _service(compose)
    assert svc.get("read_only") is True, (
        "foundryx service must declare read_only: true (issue #123); "
        "without it the container's root FS is writable and a buggy "
        "hook can persist past the run"
    )


@pytest.mark.benchmark
def test_tmpfs_has_required_paths(compose: dict) -> None:
    """Every required tmpfs path (``/tmp``, ``/var/tmp``, ``/run``) is declared.

    Each path is sized for its workload (uv cache + Python tempfile +
    sqlite WAL for ``/tmp``; rarely-written staging for ``/var/tmp``;
    PID files only for ``/run``). Removing any one path silently
    pushes its workload onto the overlay and re-opens threat #6.
    """
    entries = _tmpfs_entries(_service(compose))
    missing = [p for p in _TMPFS_REQUIRED_SIZES_MB if p not in entries]
    assert not missing, (
        f"tmpfs entries missing for required paths: {missing!r}. "
        f"Each must remain present per issue #118 / #123."
    )


@pytest.mark.benchmark
def test_every_tmpfs_entry_carries_explicit_size_cap(compose: dict) -> None:
    """Each tmpfs entry carries an explicit ``size=N{NM,K,M,G}`` token.

    Issue #123 acceptance #2: every tmpfs mount must declare a size cap
    so a misbehaving hook cannot exhaust host RAM before the memory
    limit fires. A future entry without ``size=`` would either be
    unbounded (host RAM exhaustion) or rejected by Compose at parse
    time; either failure mode turns this benchmark red.
    """
    entries = _tmpfs_entries(_service(compose))
    assert entries, "foundryx service must declare at least one tmpfs: entry"
    for path, opts in entries.items():
        match = _SIZE_CAP_PATTERN.search(opts)
        assert (
            match is not None
        ), f"tmpfs {path!r} must declare an explicit size= cap (issue #123); got opts={opts!r}"


@pytest.mark.benchmark
def test_tmpfs_per_path_sizes_match_issue_118(compose: dict) -> None:
    """Per-path tmpfs sizes match the values sized in issue #118.

    The exact sizes matter: ``/tmp`` at 512m absorbs uv + Python
    tempfile + sqlite WAL; ``/var/tmp`` at 256m covers the rare
    staging writes; ``/run`` at 64m holds PID files only. A
    regression that inflates ``/var/tmp`` to 8g or drops ``/run``
    entirely must surface as a literal value mismatch, not a vague
    "tmpfs is configured" check.
    """
    entries = _tmpfs_entries(_service(compose))
    for path, expected_mb in _TMPFS_REQUIRED_SIZES_MB.items():
        opts = entries.get(path, "")
        match = _SIZE_CAP_PATTERN.search(opts)
        assert match is not None, f"tmpfs {path!r} missing explicit size= cap; got opts={opts!r}"
        magnitude = int(match.group(1))
        unit = match.group(2).lower()
        # Convert the captured unit to MiB so the comparison is unit-aware.
        # MiB (m) is identity; KiB (k) and GiB (g) are scaled accordingly.
        unit_multiplier = {"k": 1 / 1024, "m": 1, "g": 1024}[unit]
        actual_mb = magnitude * unit_multiplier
        assert actual_mb == expected_mb, (
            f"tmpfs {path!r} must be sized to {expected_mb}m per issue #118; "
            f"got opts={opts!r} (parsed {actual_mb}m)"
        )


@pytest.mark.benchmark
def test_extra_hosts_is_exactly_one_llamacpp_entry(compose: dict) -> None:
    """``extra_hosts`` contains exactly one entry, and its key is ``llamacpp``.

    Issue #123 narrows the egress surface from a generic
    ``host.docker.internal:host-gateway`` alias (which exposes the whole
    host loopback) to a specific ``llamacpp`` alias consumed only by
    ``LLAMACPP_HOST``. Adding a second entry -- or widening it back to
    ``host.docker.internal`` -- silently re-opens the threat; this
    benchmark fails the moment either happens.
    """
    names = _extra_hosts_names(_service(compose))
    assert names, "foundryx service must declare at least one extra_hosts entry"
    assert len(names) == 1, (
        f"extra_hosts must contain EXACTLY one entry (the llamacpp "
        f"gateway); got {len(names)}: {names!r}. Adding a second alias "
        f"silently widens the egress surface."
    )
    (only,) = names
    assert only == "llamacpp", (
        f"extra_hosts key must be 'llamacpp' per issue #123; got {only!r}. "
        f"A generic alias like 'host.docker.internal' re-exposes the "
        f"entire host loopback."
    )


@pytest.mark.benchmark
def test_deploy_resources_limits_set(compose: dict) -> None:
    """``deploy.resources.limits`` carries memory, cpus, and pids caps.

    The memory + CPU caps contain a runaway evolution run (SECURITY.md
    threat #5); the ``pids`` field under ``deploy.resources.limits``
    must agree with the top-level ``pids_limit`` per Compose Spec. The
    benchmark asserts the keys are present and the values are positive,
    so a regression that drops any one of them surfaces immediately.
    """
    limits = _service(compose).get("deploy", {}).get("resources", {}).get("limits", {})
    assert limits, "deploy.resources.limits must be declared (issue #118)"
    assert limits.get("memory"), "deploy.resources.limits.memory must be set"
    assert limits.get("cpus"), "deploy.resources.limits.cpus must be set"
    assert (
        int(limits.get("pids", 0)) >= 1
    ), f"deploy.resources.limits.pids must be a positive integer; got {limits.get('pids')!r}"


@pytest.mark.benchmark
def test_pids_limit_matches_deploy_resources_pids(compose: dict) -> None:
    """``pids_limit`` equals ``deploy.resources.limits.pids`` (Compose Spec).

    The two fields must agree so swarm-mode and local-compose paths
    honour the same cap. Issue #118 sets both to 256; the benchmark
    pins that exact value and also asserts the consistency invariant
    so a future change that breaks the Spec surfaces here.
    """
    svc = _service(compose)
    top_level = int(svc.get("pids_limit", 0))
    deploy_pids = int(svc.get("deploy", {}).get("resources", {}).get("limits", {}).get("pids", 0))
    assert (
        top_level == _EXPECTED_PIDS_LIMIT
    ), f"pids_limit must be {_EXPECTED_PIDS_LIMIT} per issue #118; got {top_level}"
    assert deploy_pids == _EXPECTED_PIDS_LIMIT, (
        f"deploy.resources.limits.pids must be {_EXPECTED_PIDS_LIMIT} "
        f"per issue #118; got {deploy_pids}"
    )
    assert top_level == deploy_pids, (
        f"pids_limit ({top_level}) must match deploy.resources.limits.pids "
        f"({deploy_pids}) per Compose Spec"
    )


@pytest.mark.benchmark
def test_capabilities_dropped_and_no_new_privileges(compose: dict) -> None:
    """``cap_drop`` includes ALL and ``security_opt`` includes ``no-new-privileges``.

    Supplementary hardening beyond SECURITY.md:56: dropping every
    capability removes ambient privileges a buggy hook could abuse,
    and ``no-new-privileges`` forbids setuid binaries inside the
    container from escalating. Both are present in the file as a
    pair; a regression that drops one silently weakens the other.
    """
    svc = _service(compose)
    cap_drop = svc.get("cap_drop") or []
    assert "ALL" in cap_drop, f"cap_drop must include 'ALL' (issue #124); got {cap_drop!r}"

    security_opt = svc.get("security_opt") or []
    normalised = {str(opt).lower() for opt in security_opt}
    assert any(
        opt in {"no-new-privileges", "no-new-privileges:true"} for opt in normalised
    ), f"security_opt must include 'no-new-privileges:true' (issue #124); got {security_opt!r}"
