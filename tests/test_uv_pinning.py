"""Static validation of the sandbox Dockerfile uv installation.

Mirrors `tests/test_compose_sandbox.py` and `tests/test_dockerignore.py`:
parses the Dockerfile directly, no Docker daemon required. Guards the
pinned installer (issue #25) so the guardrail cannot regress silently.

Per ADR-0002 and `docs/SECURITY.md` threat #3, the uv installer MUST be
pinned to a specific version. An unpinned `pip install uv` reproduces
PyPI's "latest" on every build, which is exactly the reproducibility and
supply-chain weakness this guard exists to prevent.
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


def test_uv_install_is_pinned(lines: list[str]) -> None:
    """The uv install line MUST reference `${UV_VERSION}` so the version is sourced from ARG."""
    install_lines = [ln for ln in lines if "pip install" in ln and "uv" in ln]
    assert install_lines, "Dockerfile must install uv via pip (or another channel)"
    joined = " ".join(install_lines)
    assert "${UV_VERSION}" in joined, (
        "uv install must reference `${UV_VERSION}` so the version is "
        "sourced from the ARG. Unpinned `pip install uv` violates ADR-0002 "
        "reproducibility (see issue #25 and docs/SECURITY.md threat #3)."
    )


def test_uv_version_sanity_check(lines: list[str]) -> None:
    """`RUN uv --version` must follow the install so mismatches fail loudly.

    AGENTS.md §2: *"Never silently swallow an exception."* A pinned
    installer whose binary is missing or wrong-named would otherwise be
    caught only at `uv sync` time, deep inside the build.
    """
    install_idx = next(
        (i for i, ln in enumerate(lines) if "pip install" in ln and "uv" in ln),
        None,
    )
    assert install_idx is not None, "Dockerfile must install uv"
    post = lines[install_idx:]
    assert any("uv --version" in ln for ln in post), (
        "Dockerfile must run `uv --version` after installing uv so a "
        "mismatched installer fails the build loudly (AGENTS.md §2)."
    )


def test_no_unpinned_uv_install(dockerfile_text: str) -> None:
    """No line may contain a bare `pip install ... uv` without a version pin.

    Guards against a future regression that re-introduces the original
    unpinned `RUN pip install --no-cache-dir uv` while keeping a separate
    pinned line (e.g. for a different tool). Such a line would still be
    a supply-chain hole.
    """
    for raw in dockerfile_text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if "pip install" not in stripped:
            continue
        # If this line installs uv, it must carry a version pin.
        if (
            re.search(r"\buv\b", stripped)
            and "uv==" not in stripped
            and "${UV_VERSION}" not in stripped
        ):
            pytest.fail(
                f"unpinned uv install: {stripped!r}. Use `uv==${{UV_VERSION}}` "
                "or remove the line (see issue #25)."
            )
