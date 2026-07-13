"""Static validation of ``.github/dependabot.yml`` (issue #283).

Mirrors the parse-the-file style of
``tests/infra/test_workflow_uv_pin.py``: loads the config directly with
PyYAML (a core dependency), no GitHub Actions run required. Guards the
three content-pinned artifact classes identified in issue #283:

  * ``github-actions`` — SHA-pinned ``uses:`` in .github/workflows/*
  * ``docker``          — digest-pinned base image in
                          infra/docker/Dockerfile (builder + runtime)
  * ``uv``              — pyproject.toml + uv.lock (ADR-0002, canonical)

If someone deletes an ecosystem, points it at the wrong directory, flips
the schedule off weekly, or widens the uv ecosystem to transitive deps,
this test fails before the supply-chain guardrail silently regresses
(docs/SECURITY.md "Dependencies"). The acceptance criteria in #283 are
encoded one assertion per criterion so a regression points at the exact
line that broke.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEPENDABOT_CONFIG = REPO_ROOT / ".github" / "dependabot.yml"

#: The three ecosystems mandated by issue #283's acceptance criteria.
EXPECTED_ECOSYSTEMS = ("github-actions", "docker", "uv")


@pytest.fixture(scope="module")
def config() -> dict:
    assert DEPENDABOT_CONFIG.exists(), (
        f"Dependabot config missing at {DEPENDABOT_CONFIG} (issue #283)."
    )
    data = yaml.safe_load(DEPENDABOT_CONFIG.read_text())
    assert isinstance(data, dict), "dependabot.yml must parse to a mapping."
    return data


def _entry(config: dict, ecosystem: str) -> dict:
    for entry in config["updates"]:
        if entry["package-ecosystem"] == ecosystem:
            return entry
    return pytest.fail(f"ecosystem {ecosystem!r} missing from dependabot.yml.")


# ---------------------------------------------------------------------------
# Top-level schema
# ---------------------------------------------------------------------------


def test_config_uses_version_2(config: dict) -> None:
    """dependabot.yml must declare the v2 schema (issue #283)."""
    assert config.get("version") == 2


def test_config_defines_exactly_the_three_expected_ecosystems(config: dict) -> None:
    """Acceptance criterion: github-actions, docker, and uv ecosystems."""
    ecosystems = {entry["package-ecosystem"] for entry in config["updates"]}
    assert ecosystems == set(EXPECTED_ECOSYSTEMS), (
        f"Expected {sorted(EXPECTED_ECOSYSTEMS)}; got {sorted(ecosystems)}."
    )


# ---------------------------------------------------------------------------
# Per-ecosystem directory + schedule
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ecosystem,directory",
    [
        ("github-actions", "/"),
        ("docker", "/infra/docker"),
        ("uv", "/"),
    ],
)
def test_ecosystem_targets_correct_directory(config: dict, ecosystem: str, directory: str) -> None:
    """Each ecosystem must watch the directory that holds its manifest.

    docker points at ``/infra/docker`` (the Dockerfile with the
    digest-pinned python:3.11-slim@sha256, issues #124 / #283); the
    other two watch the repo root where workflows and pyproject.toml
    live.
    """
    entry = _entry(config, ecosystem)
    assert entry.get("directory") == directory, (
        f"{ecosystem} must target {directory!r} (issue #283), got {entry.get('directory')!r}."
    )


@pytest.mark.parametrize("ecosystem", EXPECTED_ECOSYSTEMS)
def test_ecosystem_schedule_is_weekly(config: dict, ecosystem: str) -> None:
    """All ecosystems run weekly — one PR per stale dependency (issue #283)."""
    entry = _entry(config, ecosystem)
    assert entry["schedule"]["interval"] == "weekly", (
        f"{ecosystem} schedule must be weekly (issue #283), "
        f"got {entry['schedule'].get('interval')!r}."
    )


# ---------------------------------------------------------------------------
# Conventional Commits discipline (ADR-0008)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ecosystem", EXPECTED_ECOSYSTEMS)
def test_ecosystem_commit_message_is_chore_deps(config: dict, ecosystem: str) -> None:
    """Bump commits follow ``chore(deps): bump ...`` (ADR-0008).

    ``prefix: chore`` sets the Conventional Commit type; ``include:
    scope`` appends ``deps``/``deps-dev`` so the commit is
    ``chore(deps): ...``, which satisfies ADR-0008's ``type(scope)``
    rule.
    """
    entry = _entry(config, ecosystem)
    cm = entry.get("commit-message", {})
    assert cm.get("prefix") == "chore", f"{ecosystem} commit prefix must be 'chore' (ADR-0008)."
    assert cm.get("include") == "scope", (
        f"{ecosystem} must include scope → 'chore(deps):' (ADR-0008)."
    )


# ---------------------------------------------------------------------------
# uv ecosystem scoping — the subtle criterion (issue #283 / ADR-0002)
# ---------------------------------------------------------------------------


def test_uv_ecosystem_scoped_to_direct_deps(config: dict) -> None:
    """uv VERSION updates cover DIRECT deps only.

    Issue #283: "targets only pinned artifacts, not transitive python
    deps (uv.lock is canonical per ADR-0002)." Restricting
    ``dependency-type`` to ``direct`` means Dependabot proposes bumps for
    the five pyproject.toml dependencies and leaves the resolved
    transitive graph to ``uv sync --frozen``. Security updates still
    reach transitive deps (``allow`` only constrains version updates).
    """
    entry = _entry(config, "uv")
    allows = entry.get("allow", [])
    assert any(a.get("dependency-type") == "direct" for a in allows), (
        "uv ecosystem must restrict version updates to direct dependencies "
        "so transitive resolution stays canonical in uv.lock "
        "(issue #283, ADR-0002)."
    )
