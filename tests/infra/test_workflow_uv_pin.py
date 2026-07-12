"""Static validation that CI workflows install uv via the hash-verified
composite action instead of ``pip install --upgrade uv``.

Mirrors ``tests/test_uv_pinning.py`` (which guards the Dockerfile) and
``tests/test_dockerignore.py``: parses YAML directly, no GitHub Actions
runner required.

Per ADR-0002 and ``docs/SECURITY.md`` threat #3, every workflow in
``.github/workflows/`` MUST install uv through the SHA256-pinned
composite action at ``.github/actions/install-uv`` (issue #208).  The
old ``pip install --upgrade uv`` path allowed a compromised PyPI
release or a MITM between the runner and PyPI to swap the binary that
gates every PR.

Guards the workflow migration so the supply-chain guardrail cannot
regress silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"
COMPOSITE_ACTION = REPO_ROOT / ".github" / "actions" / "install-uv" / "action.yml"

#: Workflows expected to install uv.  If a new workflow is added that
#: needs uv, add it here so the test enforces the pin on it too.
EXPECTED_WORKFLOWS = ["ci.yml", "audit.yml", "critic.yml", "docker.yml"]


def _load_workflow(name: str) -> str:
    path = WORKFLOWS_DIR / name
    assert path.exists(), f"workflow file missing: {path}"
    return path.read_text()


@pytest.fixture(scope="module")
def workflows() -> list[tuple[str, str]]:
    return [(name, _load_workflow(name)) for name in EXPECTED_WORKFLOWS]


@pytest.fixture(scope="module")
def composite_action_text() -> str:
    assert COMPOSITE_ACTION.exists(), f"composite action missing: {COMPOSITE_ACTION}"
    return COMPOSITE_ACTION.read_text()


# ---------------------------------------------------------------------------
# Workflow-level guards
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", EXPECTED_WORKFLOWS)
def test_no_pip_install_uv_in_workflow(name: str) -> None:
    """No workflow line may install uv via ``pip install`` (issue #208).

    ``pip install --upgrade uv`` is the supply-chain hole this issue
    closes: the version pin alone is not a content pin, and PyPI can
    serve a compromised wheel.
    """
    text = _load_workflow(name)
    for lineno, raw in enumerate(text.splitlines(), start=1):
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "pip install" not in stripped:
            continue
        if re.search(r"\buv\b", stripped):
            pytest.fail(
                f"`pip install` of uv is forbidden in {name}:{lineno}; use "
                f"the hash-verified composite action ./.github/actions/install-uv "
                f"(issue #208). Offending line: {stripped!r}"
            )


@pytest.mark.parametrize("name", EXPECTED_WORKFLOWS)
def test_workflow_uses_composite_action(name: str) -> None:
    """Every workflow MUST reference the hash-verified composite action.

    Ensures the workflow did not simply delete the ``pip install`` line
    without replacing it with the pinned installer.
    """
    text = _load_workflow(name)
    assert "./.github/actions/install-uv" in text, (
        f"{name} must use the composite action ./.github/actions/install-uv "
        f"to install uv (issue #208)."
    )


# ---------------------------------------------------------------------------
# Composite-action-level guards
# ---------------------------------------------------------------------------


def test_composite_action_downloads_from_releases(composite_action_text: str) -> None:
    """The composite action MUST fetch uv from the GitHub release tarball.

    Guards against reverting to ``pip install`` or a PyPI-based path
    inside the composite action itself.
    """
    assert re.search(
        r"https://github\.com/astral-sh/uv/releases/download/",
        composite_action_text,
    ), "Composite action must download from astral-sh/uv/releases/download/ (issue #208)."


def test_composite_action_verifies_sha256(composite_action_text: str) -> None:
    """The composite action MUST verify the tarball with ``sha256sum -c``."""
    assert "sha256sum -c" in composite_action_text, (
        "Composite action must verify the tarball with `sha256sum -c -` "
        "before extracting uv (issue #208, docs/SECURITY.md threat #3)."
    )


def test_composite_action_runs_uv_version(composite_action_text: str) -> None:
    """The composite action MUST run ``uv --version`` after install.

    AGENTS.md §2: *"Never silently swallow an exception."*  A
    mismatched or missing binary should fail the workflow immediately,
    not deep inside ``uv sync``.
    """
    assert "uv --version" in composite_action_text, (
        "Composite action must run `uv --version` after installing uv "
        "so a mismatched installer fails loudly (AGENTS.md §2)."
    )


def test_composite_action_pins_version(composite_action_text: str) -> None:
    """The composite action input default must be a concrete semver, not 'latest'."""
    m = re.search(
        r'uv-version:\s*\n\s*description:.*?\n\s*required:\s*false\s*\n\s*default:\s*"([^"]+)"',
        composite_action_text,
        re.DOTALL,
    )
    assert m, "Composite action must declare an `uv-version` input with a default."
    version = m.group(1)
    assert (
        version.lower() != "latest"
    ), f"uv-version must be a pinned release, not 'latest' (got {version!r})"
    assert re.match(
        r"^\d+\.\d+\.\d+", version
    ), f"uv-version must look like semver x.y.z (got {version!r})"


def test_composite_action_pins_sha256(composite_action_text: str) -> None:
    """The SHA256 default must be a 64-char hex string."""
    m = re.search(
        r'uv-sha256:\s*\n\s*description:.*?\n\s*required:\s*false\s*\n\s*default:\s*"([a-fA-F0-9]{64})"',
        composite_action_text,
        re.DOTALL,
    )
    assert m, "Composite action must declare `uv-sha256` with a 64-hex default. " "See issue #208."
