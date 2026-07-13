"""Tests for ``harness/manifest.json`` (issue #103).

The manifest is the producer for the harness version that
``TraceSession.harness_version`` stamps into every trace session
(``src/foundry_x/trace/logger.py:TraceSession.harness_version``).
Acceptance criteria for #103: the file parses as JSON, exposes the four
required keys, and every entry in ``hooks`` resolves under
``harness/hooks/`` and every entry in ``skills`` resolves under
``harness/skills/``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "harness" / "manifest.json"
HOOKS_DIR = REPO_ROOT / "harness" / "hooks"
SKILLS_DIR = REPO_ROOT / "harness" / "skills"

REQUIRED_KEYS = frozenset({"version", "model_target", "hooks", "skills"})
_SEMVER_RE = re.compile(r"^\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?$")


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"manifest missing at {MANIFEST_PATH}"
    text = MANIFEST_PATH.read_text(encoding="utf-8")
    return json.loads(text)


def test_manifest_parses_as_json() -> None:
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    parsed = json.loads(raw)
    assert isinstance(parsed, dict)


def test_manifest_has_required_keys(manifest: dict) -> None:
    missing = REQUIRED_KEYS - set(manifest.keys())
    assert not missing, f"manifest is missing required keys: {sorted(missing)}"


def test_manifest_version_is_semver_string(manifest: dict) -> None:
    version = manifest["version"]
    assert isinstance(version, str)
    assert _SEMVER_RE.match(version), f"version {version!r} is not a semver string"


def test_manifest_model_target_is_string(manifest: dict) -> None:
    assert isinstance(manifest["model_target"], str)
    assert manifest["model_target"].strip()


def test_manifest_hooks_is_list_of_strings(manifest: dict) -> None:
    hooks = manifest["hooks"]
    assert isinstance(hooks, list)
    assert hooks, "manifest.hooks must not be empty"
    for entry in hooks:
        assert isinstance(entry, str)
        assert entry, "hook entry must not be empty"


def test_manifest_skills_is_list_of_strings(manifest: dict) -> None:
    skills = manifest["skills"]
    assert isinstance(skills, list)
    assert skills, "manifest.skills must not be empty"
    for entry in skills:
        assert isinstance(entry, str)
        assert entry, "skill entry must not be empty"


def test_manifest_hooks_resolve_under_harness_hooks(manifest: dict) -> None:
    for entry in manifest["hooks"]:
        path = HOOKS_DIR / f"{entry}.py"
        assert path.exists(), f"hook module {entry!r} not found at {path}"


def test_manifest_skills_resolve_under_harness_skills(manifest: dict) -> None:
    for entry in manifest["skills"]:
        path = SKILLS_DIR / entry
        assert path.exists(), f"skill file {entry!r} not found at {path}"


def test_manifest_version_matches_harness_version_file(manifest: dict) -> None:
    """The manifest is the producer for ``harness_version`` (issue #103).

    The legacy single-line ``harness/VERSION`` file is what
    :func:`foundry_x.execution.runner.resolve_harness_version` reads today.
    While the runner is migrated to read the manifest (out of scope for
    #103), the two values must agree so trace-stamped versions stay
    continuous.
    """
    version_file = REPO_ROOT / "harness" / "VERSION"
    if not version_file.exists():
        pytest.skip("harness/VERSION not present")
    text = version_file.read_text(encoding="utf-8").strip()
    assert manifest["version"] == text, (
        f"manifest version {manifest['version']!r} disagrees with harness/VERSION ({text!r})"
    )
