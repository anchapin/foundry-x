"""Gate the issue #204 skill-enumeration ProposedEdit.

``harness/system_prompt.txt`` is agent DNA: per AGENTS.md section 2 and
ADR-0004 it must NOT be hand-edited. Instead the skill enumeration rides in
as a ``ProposedEdit`` artifact
(``harness/proposed_edits/issue-204-skills-enumeration.json``) that the
Evolver -> Critic gate evaluates before it is applied.

These tests prove the artifact is well-formed AND that, once applied, the
proposed system prompt satisfies the issue #204 acceptance criterion: every
name in ``manifest.skills`` appears in the prompt (one bullet, sorted
ascending, no ``.json`` suffix). If a human later applies the edit through
the Critic gate and ``system_prompt.txt`` itself carries the bullets, the
``test_system_prompt_already_enumerates_skills`` case flips on automatically
and the artifact-application cases stay green as belt-and-suspenders.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_HARNESS = _REPO_ROOT / "harness"
_ARTIFACT = _HARNESS / "proposed_edits" / "issue-204-skills-enumeration.json"
_MANIFEST = _HARNESS / "manifest.json"
_SYSTEM_PROMPT = _HARNESS / "system_prompt.txt"


def _manifest_skill_names() -> list[str]:
    """Return the manifest skill names sorted ascending, sans ``.json``."""
    manifest = json.loads(_MANIFEST.read_text(encoding="utf-8"))
    return sorted(entry[: -len(".json")] for entry in manifest["skills"])


def _load_artifact() -> dict:
    return json.loads(_ARTIFACT.read_text(encoding="utf-8"))


def _proposed_prompt() -> str:
    """Apply the artifact's old_text -> new_text replacement to the live prompt.

    Falls back to the diff's ``+`` lines if old_text/new_text are absent, so the
    test stays useful if a future artifact drops the explicit text fields.
    """
    current = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    artifact = _load_artifact()
    old_text = artifact.get("old_text")
    new_text = artifact.get("new_text")
    if old_text and new_text and old_text in current:
        return current.replace(old_text, new_text)
    # Last-resort fallback: reconstruct from the unified diff additions.
    added = [
        line[1:]
        for line in artifact["unified_diff"].splitlines()
        if line.startswith("+") and not line.startswith("+++")
    ]
    return current + "\n" + "".join(added)


# --- artifact is well-formed -------------------------------------------------


def test_artifact_targets_system_prompt():
    artifact = _load_artifact()
    assert artifact["target_file"] == "harness/system_prompt.txt"


def test_artifact_old_text_exists_in_live_prompt():
    # The proposed replacement must still apply cleanly to the current DNA.
    current = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    assert _load_artifact()["old_text"] in current


def test_artifact_references_issue_204():
    assert "204" in _load_artifact()["issue_reference"]


# --- the acceptance criterion: every manifest skill enumerated ---------------


@pytest.mark.parametrize("name", _manifest_skill_names())
def test_proposed_prompt_enumerates_each_skill(name):
    """Each manifest skill name appears as a bullet in the proposed prompt."""
    proposed = _proposed_prompt()
    assert f"- {name}" in proposed, f"skill {name!r} missing from proposed system_prompt.txt"


def test_proposed_skill_bullets_sorted_ascending():
    """Bullets must be sorted ascending to match manifest order."""
    proposed = _proposed_prompt()
    bullets = [
        line[2:]
        for line in proposed.splitlines()
        if line.startswith("- ") and line[2:] in _manifest_skill_names()
    ]
    assert bullets == sorted(bullets)


def test_proposed_skill_bullets_carry_no_json_suffix():
    proposed = _proposed_prompt()
    offending = [
        line for line in proposed.splitlines() if line.startswith("- ") and line.endswith(".json")
    ]
    assert offending == [], f"bullets must not carry .json suffix: {offending}"


# --- forward compatibility: if the edit ships, the live prompt passes too ----


def test_system_prompt_already_enumerates_skills():
    """Once the Critic gate applies the ProposedEdit, the live prompt itself
    must satisfy the invariant. Skipped (not failed) until the edit ships."""
    current = _SYSTEM_PROMPT.read_text(encoding="utf-8")
    if _load_artifact()["old_text"] in current:
        pytest.skip("ProposedEdit not yet applied to harness DNA (ADR-0004 gate)")
    missing = [n for n in _manifest_skill_names() if f"- {n}" not in current]
    assert missing == [], f"live prompt missing skill bullets: {missing}"
