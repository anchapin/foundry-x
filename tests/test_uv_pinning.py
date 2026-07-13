"""Static validation of the sandbox Dockerfile uv installation.

Mirrors `tests/test_compose_sandbox.py` and `tests/test_dockerignore.py`:
parses the Dockerfile directly, no Docker daemon required.

Per ADR-0002 and `docs/SECURITY.md` threat #3, the Dockerfile MUST pin
both the base image and the uv installer to digests (issue #124):

 - The `FROM python:3.14-slim` line MUST carry a `@sha256:<64hex>` digest
  so a fresh Docker Hub build of the same tag cannot swap the base
  layer between rebuilds.
- The uv installer MUST be fetched as a GitHub release tarball AND
  verified against an embedded SHA256 before extraction, so a
  compromised release cannot auto-install.

Guards the pinned installer (issue #25) and the hash-verified install
(issue #124) so the supply-chain guardrails cannot regress silently.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = REPO_ROOT / "infra" / "docker" / "Dockerfile"


@pytest.fixture(scope="module")
def dockerfile_text() -> str:
    assert DOCKERFILE.exists(), f"missing Dockerfile: {DOCKERFILE}"
    return DOCKERFILE.read_text()


@pytest.fixture(scope="module")
def lines(dockerfile_text: str) -> list[str]:
    return dockerfile_text.splitlines()


def test_base_image_is_digest_pinned(lines: list[str]) -> None:
    """The base image MUST be pinned to a digest (`@sha256:<64hex>`).

    A tag-only `FROM python:3.14-slim` allows Docker Hub to swap the
    layer contents between rebuilds (docs/SECURITY.md threat #3).
    Closes the gap that issue #124 calls out on line 1.
    """
    from_lines = [ln for ln in lines if re.match(r"^\s*FROM\s+\S", ln)]
    assert from_lines, "Dockerfile must start with a FROM instruction"
    first = from_lines[0]
    assert "python:3.14" in first, f"FROM must pin a Python 3.14 image, got: {first!r}"
    digest = re.search(r"@sha256:([a-f0-9]{64})\b", first)
    assert digest, (
        f"FROM must pin a digest (@sha256:<64hex>), got: {first!r}. "
        "See issue #124 and docs/SECURITY.md threat #3."
    )


def test_uv_installer_is_hash_verified(dockerfile_text: str) -> None:
    """The uv installer tarball MUST be SHA256-verified before extraction.

    Guards against an attacker who controls either the release server
    or the network path between the build host and GitHub releases
    (docs/SECURITY.md threat #3). The version pin alone (issue #25)
    is not sufficient: a compromised `uv==${UV_VERSION}` PyPI release
    would still install. The hash check is what makes the pin
    trustworthy.
    """
    # 1) ARG UV_SHA256=<64hex> must be declared with a concrete value.
    sha_arg = re.search(
        r"^\s*ARG\s+UV_SHA256\s*=\s*([a-fA-F0-9]{64})\s*$",
        dockerfile_text,
        re.MULTILINE,
    )
    assert sha_arg, (
        "Dockerfile must declare `ARG UV_SHA256=<64hex>` so the uv "
        "tarball is verified before extraction. See issue #124."
    )

    # 2) The downloaded tarball must be checked with `sha256sum -c`.
    assert "sha256sum -c" in dockerfile_text, (
        "Dockerfile must verify the tarball with `sha256sum -c -` before "
        "extracting uv. See issue #124."
    )

    # 3) The download URL must reference the pinned UV_VERSION arg,
    #    not a hard-coded version (a hard-coded URL would defeat the ARG).
    download = re.search(
        r"https://github\.com/astral-sh/uv/releases/download/\$\{?UV_VERSION\}?/",
        dockerfile_text,
    )
    assert download, (
        "Dockerfile must download the uv tarball from "
        "`https://github.com/astral-sh/uv/releases/download/${UV_VERSION}/...`. "
        "Hard-coded versions in the URL defeat ARG pinning. See issue #124."
    )


def test_arg_uv_version_is_declared(lines: list[str]) -> None:
    """The pinned version MUST be declared via `ARG UV_VERSION=<x.y.z>` (issue #25)."""
    args = [ln for ln in lines if re.match(r"\s*ARG\s+UV_VERSION\b", ln)]
    assert args, (
        "Dockerfile must declare `ARG UV_VERSION=<version>` so the uv "
        "installer is reproducible. See issue #25 and ADR-0002."
    )
    arg = args[0]
    m = re.match(r"\s*ARG\s+UV_VERSION\s*=\s*([\w.\-]+)", arg)
    assert m, f"ARG UV_VERSION must have a concrete value, got: {arg!r}"
    version = m.group(1)
    assert (
        version.lower() != "latest"
    ), f"UV_VERSION must be a pinned version, not 'latest' (got {version!r})"
    assert re.match(
        r"^\d+\.\d+\.\d+", version
    ), f"UV_VERSION must look like semver `x.y.z` (got {version!r})"


def test_uv_version_sanity_check(lines: list[str]) -> None:
    """`uv --version` must run after the install so mismatches fail loudly.

    AGENTS.md §2: *"Never silently swallow an exception."* A pinned
    installer whose binary is missing or wrong-named would otherwise
    be caught only at `uv sync` time, deep inside the build.
    """
    install_idx = next(
        (i for i, ln in enumerate(lines) if "astral-sh/uv/releases/download" in ln),
        None,
    )
    assert (
        install_idx is not None
    ), "Dockerfile must fetch uv from astral-sh/uv/releases/download (issue #124)."
    post = lines[install_idx:]
    assert any("uv --version" in ln for ln in post), (
        "Dockerfile must run `uv --version` after installing uv so a "
        "mismatched installer fails the build loudly (AGENTS.md §2)."
    )


def test_no_pip_install_uv(dockerfile_text: str) -> None:
    """No line may install uv via `pip install` — the hash-verified
    tarball is the canonical install path (issue #124).

    Guards against a regression that reverts to `pip install uv==...`,
    which is the supply-chain hole this whole issue closes. The
    version pin alone is not a content pin.
    """
    for raw in dockerfile_text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "pip install" not in stripped:
            continue
        if re.search(r"\buv\b", stripped):
            pytest.fail(
                f"`pip install` of uv is forbidden; use the hash-verified "
                f"tarball from astral-sh/uv/releases/download (issue #124). "
                f"Offending line: {stripped!r}"
            )
