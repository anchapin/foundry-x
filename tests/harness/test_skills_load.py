"""Schema conformance test for every ``harness/skills/*.json`` document.

Mirrors ``harness/scripts/load_check.py`` (issue #107) at the unit-test
level: iterate the JSON files in the harness skills directory, parse each,
and assert the five required keys (``name``, ``version``, ``description``,
``input_schema``, ``output_schema``) are present and well-typed.

The ``load_check`` script is the Critic-gate equivalent (ADR-0004); this
test is the developer-facing regression net that runs on every push via
``uv run pytest tests/harness/``. Keeping both paths aligned is
intentional -- a broken skill must surface red both in CI and at the
Critic gate, never just one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SKILLS_DIR = REPO_ROOT / "harness" / "skills"

REQUIRED_TOP_KEYS: tuple[str, ...] = (
    "name",
    "version",
    "description",
    "input_schema",
    "output_schema",
)


def _skill_files() -> list[Path]:
    files = sorted(SKILLS_DIR.glob("*.json"))
    if not files:
        pytest.skip(f"no skill files found under {SKILLS_DIR}")
    return files


def test_skills_directory_exists() -> None:
    assert SKILLS_DIR.is_dir(), f"harness skills directory missing: {SKILLS_DIR}"


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.name)
def test_skill_file_parses_as_json(skill_path: Path) -> None:
    raw = skill_path.read_text(encoding="utf-8")
    doc = json.loads(raw)
    assert isinstance(
        doc, dict
    ), f"{skill_path.name}: top-level must be a JSON object, got {type(doc).__name__}"


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.name)
def test_skill_has_required_top_level_keys(skill_path: Path) -> None:
    doc = json.loads(skill_path.read_text(encoding="utf-8"))
    missing = [k for k in REQUIRED_TOP_KEYS if k not in doc]
    assert not missing, f"{skill_path.name}: missing required keys {missing!r}"


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.name)
def test_skill_metadata_fields_are_well_typed(skill_path: Path) -> None:
    doc = json.loads(skill_path.read_text(encoding="utf-8"))
    name = doc.get("name")
    version = doc.get("version")
    description = doc.get("description")
    assert isinstance(name, str) and name, f"{skill_path.name}: name must be a non-empty string"
    assert (
        isinstance(version, str) and version
    ), f"{skill_path.name}: version must be a non-empty semver string"
    assert (
        isinstance(description, str) and description
    ), f"{skill_path.name}: description must be a non-empty string"


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.name)
def test_skill_schemas_are_json_schemas(skill_path: Path) -> None:
    doc = json.loads(skill_path.read_text(encoding="utf-8"))
    for key in ("input_schema", "output_schema"):
        schema = doc.get(key)
        assert isinstance(
            schema, dict
        ), f"{skill_path.name}: {key} must be a JSON object, got {type(schema).__name__}"
        assert (
            schema.get("type") == "object"
        ), f"{skill_path.name}: {key} must declare type=object at the top level"


def test_bash_skill_exists_and_has_documented_contract() -> None:
    """Issue #104: the bash skill ships with the documented I/O contract.

    Lives in addition to the per-file parametrized tests above because the
    bash skill is the first executable surface seeded into the harness
    (issue #104, SECURITY.md \u00a71 threat #3). A regression that deletes
    bash.json or weakens its contract must surface here with a precise
    message so the Critic gate (ADR-0004) flags it.
    """
    path = SKILLS_DIR / "bash.json"
    assert path.is_file(), f"bash skill missing: {path}"
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["name"] == "bash"
    assert isinstance(doc["version"], str) and doc["version"]

    command_prop = doc["input_schema"]["properties"]["command"]
    assert (
        command_prop["type"] == "string"
    ), "bash input_schema.command must be a string per the issue #104 acceptance"
    assert (
        "command" in doc["input_schema"]["required"]
    ), "bash input_schema must require 'command' per the issue #104 acceptance"
    cwd_prop = doc["input_schema"]["properties"].get("cwd")
    assert (
        cwd_prop is not None and cwd_prop["type"] == "string"
    ), "bash input_schema.properties.cwd must be an optional string"

    for field in ("stdout", "stderr", "exit_code", "truncated"):
        assert (
            field in doc["output_schema"]["properties"]
        ), f"bash output_schema must declare {field!r} per the issue #104 acceptance"
    assert set(doc["output_schema"]["required"]) >= {
        "stdout",
        "stderr",
        "exit_code",
        "truncated",
    }, "bash output_schema.required must list all four documented fields"

    assert (
        "subprocess" in doc["description"]
    ), "bash description must defer to subprocess per docs/SECURITY.md \u00a71 threat #3"
