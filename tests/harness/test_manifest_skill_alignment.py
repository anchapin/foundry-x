"""Round-trip alignment between ``harness/manifest.json`` and on-disk skills.

The manifest is the Runner's source of truth for which tool definitions it
advertises to the model (ADR-0010). When ``harness/skills/*.json`` files are
added (e.g. #104 bash, #105 list_dir + grep_search) but the manifest's
``skills`` array is not updated, those tools become invisible to every
manifest-constrained boot -- the model never learns they exist.

This module enforces the invariant both ways:

* every ``*.json`` file under ``harness/skills/`` appears in the manifest
* every entry in ``manifest["skills"]`` resolves to a file on disk

Acceptance criteria for #202.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "harness" / "manifest.json"
SKILLS_DIR = REPO_ROOT / "harness" / "skills"


def _load_manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"manifest missing at {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _on_disk_skill_basenames() -> list[str]:
    """Sorted, unique basenames of every ``harness/skills/*.json`` file."""
    return sorted(p.name for p in SKILLS_DIR.glob("*.json"))


def test_manifest_skills_match_on_disk_files() -> None:
    """Every ``harness/skills/*.json`` basename appears in the manifest and
    vice-versa -- the two sets are identical (sorted, unique)."""
    manifest = _load_manifest()
    manifest_skills = sorted(set(manifest["skills"]))
    on_disk = _on_disk_skill_basenames()
    assert manifest_skills == on_disk, (
        f"manifest.skills ({manifest_skills}) != on-disk skill files ({on_disk})"
    )


def test_no_phantom_manifest_entries() -> None:
    """No manifest ``skills`` entry may reference a non-existent file."""
    manifest = _load_manifest()
    for entry in manifest["skills"]:
        path = SKILLS_DIR / entry
        assert path.exists(), f"manifest lists skill {entry!r} but {path} does not exist"


def test_no_orphan_skill_files() -> None:
    """No on-disk ``harness/skills/*.json`` file may be absent from the manifest."""
    manifest = _load_manifest()
    manifest_set = set(manifest["skills"])
    for path in SKILLS_DIR.glob("*.json"):
        assert path.name in manifest_set, (
            f"{path.name} exists on disk but is missing from manifest.skills"
        )
