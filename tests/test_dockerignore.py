"""Static validation of the repo-root `.dockerignore`.

Mirrors `tests/test_compose_sandbox.py`: parses the file directly, no Docker
daemon required. Guards the secret/trace exclusions so the guardrail cannot
regress silently (see issue #23 and docs/SECURITY.md threat #4).

A `.dockerignore` entry matches anything when its glob is a prefix of the
candidate path. We use the same rules Docker uses at the file-system level:
  - `name` matches `candidate` if `candidate == name` or
    `candidate.startswith(name + "/")` (and similarly for suffix globs).
We do not attempt to be a full Docker parser; the test asserts that the
*required exclusion patterns appear as lines*, not that every file the
Dockerfile might read is correctly excluded.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
DOCKERIGNORE = REPO_ROOT / ".dockerignore"


@pytest.fixture(scope="module")
def patterns() -> list[str]:
    """Return the non-comment, non-empty lines of `.dockerignore`."""
    assert DOCKERIGNORE.exists(), (
        f"missing .dockerignore at repo root: {DOCKERIGNORE}. "
        "Issue #23 requires one to keep secrets and the trace store out "
        "of the Docker build context."
    )
    lines = DOCKERIGNORE.read_text().splitlines()
    return [line.strip() for line in lines if line.strip() and not line.strip().startswith("#")]


# Patterns that MUST be excluded. Names mirror docs/SECURITY.md:61-69 and
# ADR-0003. Order is intentional: secret-adjacent patterns first. Each is a
# concrete path or file that any real trace store / cache will produce, so a
# matching glob in `.dockerignore` proves the exclusion is wired up.
REQUIRED_EXCLUSIONS: tuple[str, ...] = (
    ".git",
    ".env",
    "logs/trace.jsonl",
    "__pycache__/foo.cpython-311.pyc",
    ".venv/bin/python",
    ".ruff_cache/foo.txt",
    ".pytest_cache/foo.txt",
    ".idea/workspace.xml",
    ".vscode/settings.json",
    "foo.pyc",
    "foo.db",
)


# Patterns the Dockerfile COPYs and therefore MUST NOT be excluded. Excluding
# any of these would break `docker build -f infra/docker/Dockerfile .`.
FORBIDDEN_EXCLUSIONS: tuple[str, ...] = (
    "src",
    "harness",
    "tests",
    "pyproject.toml",
    "uv.lock",
    "infra",
)


@pytest.mark.parametrize("required", REQUIRED_EXCLUSIONS)
def test_required_pattern_is_excluded(patterns: list[str], required: str) -> None:
    """Each secret/trace/cache pattern from the issue must be matched by `.dockerignore`."""
    assert _is_ignored(patterns, required), (
        f".dockerignore must match '{required}' "
        f"(see issue #23 and docs/SECURITY.md). Lines checked: {patterns}"
    )


@pytest.mark.parametrize("forbidden", FORBIDDEN_EXCLUSIONS)
def test_dockerfile_inputs_are_not_excluded(patterns: list[str], forbidden: str) -> None:
    """Patterns the Dockerfile COPYs must NOT appear as standalone exclusions.

    We check exact-token match against the parsed lines so a benign substring
    inside a longer glob (e.g. a hypothetical `infra-tools/`) does not trip
    the guard.
    """
    exact = set(patterns)
    assert forbidden not in exact, (
        f".dockerignore excludes '{forbidden}' but infra/docker/Dockerfile "
        f"COPYs it; the build would fail. Drop the entry."
    )


def test_dockerignore_is_well_formed(patterns: list[str]) -> None:
    """Each line must be a non-empty, non-whitespace-only string."""
    assert patterns, ".dockerignore must not be empty"
    for line in patterns:
        assert line == line.strip(), f"trailing whitespace on line: {line!r}"


def test_dockerignore_has_a_lead_comment() -> None:
    """Operators should know why the file exists before they edit it."""
    head = DOCKERIGNORE.read_text().splitlines()[:3]
    assert any(line.lstrip().startswith("#") for line in head), (
        ".dockerignore should start with a comment explaining its purpose"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_ignored(patterns: list[str], candidate: str) -> bool:
    """Return True if any `.dockerignore` line excludes `candidate`.

    Models the two cases that cover every entry in this repo:
      1. A bare directory prefix (`logs`, `src`) matches anything under it.
      2. A glob pattern matches a concrete file via `fnmatch`.
    """
    for pat in patterns:
        # Directory-style exclusions: `logs` or `logs/` both ignore `logs/foo`.
        if pat.rstrip("/") == candidate.split("/", 1)[0]:
            return True
        # Glob-style exclusions: `*.pyc`, `*.py[cod]`, `foo.db`, etc.
        if any(c in pat for c in "*?[]") and fnmatch.fnmatch(candidate, pat):
            return True
        # Exact file exclusion.
        if pat == candidate:
            return True
    return False
