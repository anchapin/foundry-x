from __future__ import annotations

import json
from pathlib import Path


from foundry_x.execution.runner import _inject_skill_list


def test_inject_skill_list_replaces_placeholder(tmp_path: Path):
    """Issue #582: ``{{ SKILL_LIST }}`` is replaced with bulleted skill names."""
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "system_prompt.txt").write_text(
        "Skills:\n\n{{ SKILL_LIST }}\n\nEnd.",
        encoding="utf-8",
    )
    manifest = {
        "version": "0.1.0",
        "model_target": "test/model",
        "hooks": [],
        "skills": ["alpha.json", "beta.json"],
        "skill_inventory": [
            {"name": "alpha", "description": "Alpha skill"},
            {"name": "beta", "description": "Beta skill"},
        ],
    }
    (harness / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prompt = (harness / "system_prompt.txt").read_text(encoding="utf-8")
    result = _inject_skill_list(harness, prompt)

    assert result == "Skills:\n\n- alpha\n- beta\n\nEnd."
    assert "{{ SKILL_LIST }}" not in result


def test_inject_skill_list_falls_back_to_skills_field(tmp_path: Path):
    """When ``skill_inventory`` is absent, names are derived from ``skills`` filenames."""
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "system_prompt.txt").write_text(
        "Tools: {{ SKILL_LIST }}",
        encoding="utf-8",
    )
    manifest = {
        "version": "0.1.0",
        "model_target": "test/model",
        "hooks": [],
        "skills": ["gamma.json", "delta.json"],
    }
    (harness / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prompt = (harness / "system_prompt.txt").read_text(encoding="utf-8")
    result = _inject_skill_list(harness, prompt)

    assert result == "Tools: - delta\n- gamma"
    assert "{{ SKILL_LIST }}" not in result


def test_inject_skill_list_no_manifest_leaves_prompt_unchanged(tmp_path: Path):
    """When no manifest.json exists, the prompt is returned unchanged."""
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "system_prompt.txt").write_text(
        "Hello {{ SKILL_LIST }} world",
        encoding="utf-8",
    )

    prompt = (harness / "system_prompt.txt").read_text(encoding="utf-8")
    result = _inject_skill_list(harness, prompt)

    assert result == "Hello {{ SKILL_LIST }} world"


def test_inject_skill_list_empty_skills_removes_placeholder(tmp_path: Path):
    """When both ``skill_inventory`` and ``skills`` are empty, placeholder is removed."""
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "system_prompt.txt").write_text(
        "Tools:{{ SKILL_LIST }}done",
        encoding="utf-8",
    )
    manifest = {
        "version": "0.1.0",
        "model_target": "test/model",
        "hooks": [],
        "skills": [],
    }
    (harness / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prompt = (harness / "system_prompt.txt").read_text(encoding="utf-8")
    result = _inject_skill_list(harness, prompt)

    assert result == "Tools:done"
    assert "{{ SKILL_LIST }}" not in result


def test_inject_skill_list_invalid_json_leaves_prompt_unchanged(tmp_path: Path):
    """When manifest.json is malformed JSON, prompt is returned unchanged."""
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "system_prompt.txt").write_text(
        "Hello {{ SKILL_LIST }} world",
        encoding="utf-8",
    )
    (harness / "manifest.json").write_text("{ not json", encoding="utf-8")

    prompt = (harness / "system_prompt.txt").read_text(encoding="utf-8")
    result = _inject_skill_list(harness, prompt)

    assert result == "Hello {{ SKILL_LIST }} world"


def test_inject_skill_list_sorted_alphabetically(tmp_path: Path):
    """Skill names are sorted alphabetically for deterministic output."""
    harness = tmp_path / "harness"
    harness.mkdir()
    (harness / "system_prompt.txt").write_text(
        "{{ SKILL_LIST }}",
        encoding="utf-8",
    )
    manifest = {
        "version": "0.1.0",
        "model_target": "test/model",
        "hooks": [],
        "skills": ["zulu.json", "alpha.json", "mike.json"],
        "skill_inventory": [
            {"name": "zulu", "description": "Zulu"},
            {"name": "alpha", "description": "Alpha"},
            {"name": "mike", "description": "Mike"},
        ],
    }
    (harness / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    prompt = (harness / "system_prompt.txt").read_text(encoding="utf-8")
    result = _inject_skill_list(harness, prompt)

    assert result == "- alpha\n- mike\n- zulu"
