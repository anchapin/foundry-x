"""Guard against silent Python version drift between Dockerfile and CI.

Per ADR-0002, the Python minor version that runs the production sandbox
container (declared by every ``FROM python:X.Y-slim`` in
``infra/docker/Dockerfile``) MUST match the Python minor version every
CI workflow installs via ``uv python install <X.Y>``. Without this
guard the two can drift silently: ``pyproject.toml`` only declares
``requires-python = ">=3.11"`` (a floor, not a pin), so a Dockerfile bump
to 3.14 while CI still runs 3.11 passes every check while a 3.14-only
regression (a stdlib deprecation, a typing change, an asyncio shift)
ships undetected (issue #930).

Mirrors ``tests/test_uv_pinning.py`` (Dockerfile) and
``tests/infra/test_workflow_uv_pin.py`` (CI workflows): parses the files
directly, no Docker daemon or GitHub Actions runner required.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = REPO_ROOT / "infra" / "docker" / "Dockerfile"
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

#: Matches a Dockerfile base-image pin, e.g. ``FROM python:3.14-slim@sha256:...``.
#: Anchored on ``FROM`` so historical ``# was: python:3.x-slim`` comments
#: do not pollute the extracted version (issue #930).
_DOCKERFILE_PYTHON_RE = re.compile(
    r"^\s*FROM\s+python:(?P<major>\d+)\.(?P<minor>\d+)-slim\b",
    re.MULTILINE,
)

#: Matches the CI Python install line ``uv python install 3.14``.
_WORKFLOW_PYTHON_RE = re.compile(r"uv python install\s+(?P<major>\d+)\.(?P<minor>\d+)\b")

#: Every workflow expected to declare a Python minor via
#: ``uv python install <X.Y>``. Kept in sync with
#: ``tests/infra/test_workflow_uv_pin.EXPECTED_WORKFLOWS`` (issue #868):
#: if a workflow is added that needs uv, add it here too so the
#: version-consistency guard enforces the pin on it as well.
EXPECTED_WORKFLOWS = [
    "ci.yml",
    "audit.yml",
    "critic.yml",
    "docker.yml",
    "lint.yml",
    "pre-commit.yml",
    "test.yml",
    "quantization-sweep.yml",
    "real-llm.yml",
]


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.exists(), f"missing Dockerfile: {DOCKERFILE}"
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def dockerfile_python_minor(dockerfile_text: str) -> str:
    """The canonical ``X.Y`` minor pinned by every ``FROM python:X.Y-slim``.

    The Dockerfile has two stages (builder + runtime); both MUST agree on
    the same minor, otherwise the production image is internally
    inconsistent (issue #930).
    """
    matches = list(_DOCKERFILE_PYTHON_RE.finditer(dockerfile_text))
    assert matches, (
        "Dockerfile must declare a `FROM python:X.Y-slim` base image (issue #124, #930)."
    )
    minors = {f"{m['major']}.{m['minor']}" for m in matches}
    assert len(minors) == 1, (
        "All `FROM python:X.Y-slim` stages in the Dockerfile must pin the "
        f"same minor version; found conflicting pins: {sorted(minors)} (issue #930)."
    )
    return next(iter(minors))


def test_dockerfile_pins_concrete_python_minor(dockerfile_python_minor: str) -> None:
    """Sanity: the Dockerfile pins a concrete ``X.Y`` Python minor (issue #930)."""
    assert re.match(r"^\d+\.\d+$", dockerfile_python_minor), (
        f"Dockerfile Python minor must look like `X.Y`, got {dockerfile_python_minor!r}"
    )


@pytest.mark.parametrize("name", EXPECTED_WORKFLOWS)
def test_workflow_python_matches_dockerfile(name: str, dockerfile_python_minor: str) -> None:
    """Every CI ``uv python install X.Y`` MUST match the Dockerfile minor.

    This FAILS (does not skip) on drift so a Dockerfile bump that leaves
    CI on the old minor cannot ship a version-specific regression
    undetected (issue #930, acceptance criterion #2).
    """
    path = WORKFLOWS_DIR / name
    assert path.exists(), f"workflow file missing: {path}"
    text = path.read_text()

    pins = list(_WORKFLOW_PYTHON_RE.finditer(text))
    assert pins, (
        f"{name} must declare the Python version via `uv python install <X.Y>` "
        f"so the consistency guard can enforce it (issue #930)."
    )
    workflow_minors = {f"{p['major']}.{p['minor']}" for p in pins}
    drifted = sorted(workflow_minors - {dockerfile_python_minor})
    assert not drifted, (
        f"{name} installs Python {sorted(workflow_minors)} but the Dockerfile "
        f"base image is python:{dockerfile_python_minor}-slim. CI and the "
        f"production container MUST run the same Python minor or a "
        f"version-specific regression ships undetected (issue #930). "
        f"Drifted: {drifted}."
    )


def test_no_workflow_left_out_of_consistency_guard() -> None:
    """No workflow outside ``EXPECTED_WORKFLOWS`` may pin a Python version.

    If a new workflow starts using ``uv python install`` it must be added
    to ``EXPECTED_WORKFLOWS`` (above) or it escapes the guard silently —
    the exact failure mode issue #930 exists to close.
    """
    pinned_workflows: list[str] = []
    for path in sorted(WORKFLOWS_DIR.glob("*.yml")):
        if _WORKFLOW_PYTHON_RE.search(path.read_text()):
            pinned_workflows.append(path.name)
    unguarded = sorted(set(pinned_workflows) - set(EXPECTED_WORKFLOWS))
    assert not unguarded, (
        "Workflows pin a Python version but are not in EXPECTED_WORKFLOWS, "
        f"so the consistency guard does not cover them: {unguarded}. Add them "
        f"to EXPECTED_WORKFLOWS in this file (issue #930)."
    )
