"""Schema validation for the harness skill surface.

Loads every ``harness/skills/*.json`` and validates it against the shape
established by ``example_skill.json`` (ADR-0004, issue #19). A skill must:

* have a non-empty ``name`` and ``version`` string,
* declare an ``input_schema`` that is a JSON-schema *object* with
  ``additionalProperties: false`` (closed schema, no invented args),
* declare an ``output_schema`` that is a JSON-schema *object*.

The active skill set must include at least one write-capable skill so the
agent can produce or modify code (PRD improvement-rate KPI).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

SKILLS_DIR = Path(__file__).resolve().parents[1] / "harness" / "skills"


def _load_skills() -> list[dict]:
    files = sorted(SKILLS_DIR.glob("*.json"))
    assert files, "no skill files found in harness/skills/"
    return [json.loads(f.read_text()) for f in files]


SKILLS = _load_skills()
SKILL_NAMES = {s["name"] for s in SKILLS}


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s["name"])
def test_skill_has_required_top_level_fields(skill: dict) -> None:
    for field in ("name", "version", "description", "input_schema", "output_schema"):
        assert field in skill, f"skill {skill.get('name')!r} missing top-level field {field!r}"


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s["name"])
def test_skill_name_and_version_are_non_empty_strings(skill: dict) -> None:
    assert isinstance(skill["name"], str) and skill["name"].strip()
    assert isinstance(skill["version"], str) and skill["version"].strip()


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s["name"])
def test_input_schema_is_closed_object(skill: dict) -> None:
    schema = skill["input_schema"]
    assert schema["type"] == "object"
    assert isinstance(schema["properties"], dict) and schema["properties"]
    assert "required" in schema and isinstance(schema["required"], list)
    # Closed schema: the agent must not invent new arguments.
    assert schema.get("additionalProperties") is False


@pytest.mark.parametrize("skill", SKILLS, ids=lambda s: s["name"])
def test_output_schema_is_object(skill: dict) -> None:
    schema = skill["output_schema"]
    assert schema["type"] == "object"
    assert isinstance(schema["properties"], dict) and schema["properties"]
    assert isinstance(schema["required"], list)


def test_read_file_skill_present() -> None:
    assert "read_file" in SKILL_NAMES


def test_at_least_one_write_capable_skill() -> None:
    """The tool surface must allow the agent to produce code (issue #19)."""
    write_capable = SKILL_NAMES & {"write_file", "edit_file"}
    assert write_capable, f"no write-capable skill found; expected one of write_file/edit_file, got {sorted(SKILL_NAMES)}"
