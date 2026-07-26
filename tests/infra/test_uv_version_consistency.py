"""Guard against silent uv version drift between Dockerfile and CI.

Per ADR-0002 and ``docs/SECURITY.md`` threat #3, the uv version that
ships in the production sandbox image (declared by the ``ARG
UV_VERSION`` in ``infra/docker/Dockerfile``) MUST match the uv version
the CI composite action installs (the ``uv-version`` input default in
``.github/actions/install-uv/action.yml``). Without this guard the two
can drift silently: a bump to one location that forgets the other
passes every existing check — ``test_workflow_uv_pin.py`` only validates
that the action.yml default is semver-formatted, it never reads the
Dockerfile value, and ``test_python_version_consistency.py`` covers
Python minors but not uv (issue #982).

Mirrors ``tests/infra/test_python_version_consistency.py``: parses the
two files directly, no Docker daemon or GitHub Actions runner required.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "infra" / "docker" / "Dockerfile"
COMPOSITE_ACTION = REPO_ROOT / ".github" / "actions" / "install-uv" / "action.yml"

#: Matches the Dockerfile ``ARG UV_VERSION=x.y.z`` pin (issue #982).
#: Anchored on ``UV_VERSION`` so historical or unrelated ARGs do not
#: pollute the extracted value.
_DOCKERFILE_UV_VERSION_RE = re.compile(
    r"^\s*ARG\s+UV_VERSION=(?P<version>\S+)\s*$",
    re.MULTILINE,
)

#: Matches the ``uv-version`` input default in action.yml. Matches across
#: the description/required/default block so a multiline description does
#: not break the capture (same approach as
#: ``test_workflow_uv_pin.test_composite_action_pins_version``).
_ACTION_UV_VERSION_RE = re.compile(
    r'uv-version:\s*\n\s*description:.*?\n\s*required:\s*false\s*\n\s*default:\s*"([^"]+)"',
    re.DOTALL,
)


@pytest.fixture(scope="module")
def dockerfile_uv_version() -> str:
    """The uv version pinned by the Dockerfile ``ARG UV_VERSION``."""
    assert DOCKERFILE.exists(), f"missing Dockerfile: {DOCKERFILE}"
    text = DOCKERFILE.read_text()
    pins = list(_DOCKERFILE_UV_VERSION_RE.finditer(text))
    assert pins, "Dockerfile must declare `ARG UV_VERSION=x.y.z` (issue #25, #982)."
    versions = {m["version"] for m in pins}
    assert len(versions) == 1, (
        "Dockerfile must pin UV_VERSION exactly once; found conflicting pins: "
        f"{sorted(versions)} (issue #982)."
    )
    return next(iter(versions))


@pytest.fixture(scope="module")
def action_uv_version() -> str:
    """The uv version defaulted by the composite action ``uv-version`` input."""
    assert COMPOSITE_ACTION.exists(), f"composite action missing: {COMPOSITE_ACTION}"
    text = COMPOSITE_ACTION.read_text()
    m = _ACTION_UV_VERSION_RE.search(text)
    assert m, (
        "Composite action must declare an `uv-version` input with a default (issue #208, #982)."
    )
    return m.group(1)


def test_dockerfile_pins_concrete_uv_version(dockerfile_uv_version: str) -> None:
    """Sanity: the Dockerfile pins a concrete uv semver (issue #982)."""
    assert re.match(r"^\d+\.\d+\.\d+", dockerfile_uv_version), (
        f"Dockerfile UV_VERSION must look like semver x.y.z, got {dockerfile_uv_version!r}"
    )


def test_action_pins_concrete_uv_version(action_uv_version: str) -> None:
    """Sanity: the composite action defaults to a concrete uv semver (issue #982)."""
    assert re.match(r"^\d+\.\d+\.\d+", action_uv_version), (
        f"Composite action uv-version default must look like semver x.y.z, "
        f"got {action_uv_version!r}"
    )


def test_uv_version_matches_across_dockerfile_and_action(
    dockerfile_uv_version: str,
    action_uv_version: str,
) -> None:
    """The Dockerfile ``ARG UV_VERSION`` MUST match the action.yml default.

    A bump to one location without the other produces a silent drift:
    the Dockerfile ARG and the action.yml default disagree, yet all
    existing tests pass. This is exactly the gap issue #982 exists to
    close (mirroring the Python minor guard from issue #930).
    """
    assert dockerfile_uv_version == action_uv_version, (
        f"UV_VERSION drift detected (issue #982): Dockerfile pins "
        f"{dockerfile_uv_version!r} but the install-uv composite action "
        f"defaults to {action_uv_version!r}. Bump both locations together "
        f"— see the bump instructions in action.yml lines 10-13 and "
        f"Dockerfile lines 57-60."
    )
