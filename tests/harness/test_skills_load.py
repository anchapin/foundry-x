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
    assert isinstance(doc, dict), (
        f"{skill_path.name}: top-level must be a JSON object, got {type(doc).__name__}"
    )


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
    assert isinstance(version, str) and version, (
        f"{skill_path.name}: version must be a non-empty semver string"
    )
    assert isinstance(description, str) and description, (
        f"{skill_path.name}: description must be a non-empty string"
    )


@pytest.mark.parametrize("skill_path", _skill_files(), ids=lambda p: p.name)
def test_skill_schemas_are_json_schemas(skill_path: Path) -> None:
    doc = json.loads(skill_path.read_text(encoding="utf-8"))
    for key in ("input_schema", "output_schema"):
        schema = doc.get(key)
        assert isinstance(schema, dict), (
            f"{skill_path.name}: {key} must be a JSON object, got {type(schema).__name__}"
        )
        assert schema.get("type") == "object", (
            f"{skill_path.name}: {key} must declare type=object at the top level"
        )


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
    assert command_prop["type"] == "string", (
        "bash input_schema.command must be a string per the issue #104 acceptance"
    )
    assert "command" in doc["input_schema"]["required"], (
        "bash input_schema must require 'command' per the issue #104 acceptance"
    )
    cwd_prop = doc["input_schema"]["properties"].get("cwd")
    assert cwd_prop is not None and cwd_prop["type"] == "string", (
        "bash input_schema.properties.cwd must be an optional string"
    )

    for field in ("stdout", "stderr", "exit_code", "truncated"):
        assert field in doc["output_schema"]["properties"], (
            f"bash output_schema must declare {field!r} per the issue #104 acceptance"
        )
    assert set(doc["output_schema"]["required"]) >= {
        "stdout",
        "stderr",
        "exit_code",
        "truncated",
    }, "bash output_schema.required must list all four documented fields"

    assert "subprocess" in doc["description"], (
        "bash description must defer to subprocess per docs/SECURITY.md \u00a71 threat #3"
    )


def _load_skill(name: str) -> dict:
    """Load ``harness/skills/<name>.json`` and return its parsed document.

    Centralises the file->dict hop so the per-skill contract tests below
    share one well-known entry point. Raises ``FileNotFoundError`` loudly
    when the JSON is missing -- the agent harness must ship every seeded
    skill on disk (SECURITY.md threat #5: a contract without an
    implementation is a silent capability regression).
    """
    path = SKILLS_DIR / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"harness skill missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def test_list_dir_skill_exists_and_has_documented_contract() -> None:
    """Issue #105: the ``list_dir`` skill ships with the documented I/O contract.

    Mirrors the bash-skill contract test above. The agent uses
    ``list_dir`` to *discover* files before reading them; without it,
    locating a benchmark target such as ``benchmarks/fixtures/
    write_unit_test/target.py`` reduces to guessing paths (issue #105
    acceptance: a new test asserts list_dir on the workspace returns
    ``target.py`` among its entries).
    """
    doc = _load_skill("list_dir")
    assert doc["name"] == "list_dir"
    assert isinstance(doc["version"], str) and doc["version"]

    properties = doc["input_schema"]["properties"]
    assert "path" in properties, (
        "list_dir input_schema must declare 'path' per issue #105 acceptance"
    )
    assert properties["path"]["type"] == "string", (
        "list_dir input_schema.path must be a string per issue #105 acceptance"
    )
    assert "path" in doc["input_schema"]["required"], (
        "list_dir input_schema.required must list 'path' per issue #105 acceptance"
    )

    out_properties = doc["output_schema"]["properties"]
    assert "entries" in out_properties, (
        "list_dir output_schema must declare 'entries' per issue #105 acceptance"
    )
    entries_schema = out_properties["entries"]
    assert entries_schema["type"] == "array", (
        "list_dir output_schema.entries must be an array per issue #105 acceptance"
    )
    entry_props = entries_schema["items"]["properties"]
    for field in ("name", "kind", "size"):
        assert field in entry_props, (
            f"list_dir output_schema.entries.items must declare {field!r} per issue #105 acceptance"
        )
    assert set(doc["output_schema"]["required"]) >= {"entries", "truncated"}, (
        "list_dir output_schema.required must list 'entries' and 'truncated' per issue #105 acceptance"
    )

    assert "truncation_policy" in doc, (
        "list_dir must publish a truncation_policy so the Phase 3 historical-log "
        "pruner has a per-call bound to fall back to (SECURITY.md threat #5)"
    )
    assert "scandir" in doc["description"] or "stdlib" in doc["description"], (
        "list_dir description must defer to Python stdlib (os.scandir) per docs/SECURITY.md \u00a71 threat #3"
    )


def test_grep_search_skill_exists_and_has_documented_contract() -> None:
    """Issue #105: the ``grep_search`` skill ships with the documented I/O contract.

    Companion to :func:`test_list_dir_skill_exists_and_has_documented_contract`.
    The agent uses ``grep_search`` to *locate* a symbol by content before
    reading it; the contract is engine-agnostic but the audit surface
    MUST stay stdlib-only (SECURITY.md \u00a71 threat #3).
    """
    doc = _load_skill("grep_search")
    assert doc["name"] == "grep_search"
    assert isinstance(doc["version"], str) and doc["version"]

    properties = doc["input_schema"]["properties"]
    for field in ("pattern", "path"):
        assert field in properties, (
            f"grep_search input_schema must declare {field!r} per issue #105 acceptance"
        )
        assert properties[field]["type"] == "string", (
            f"grep_search input_schema.{field} must be a string per issue #105 acceptance"
        )
    for field in ("pattern", "path"):
        assert field in doc["input_schema"]["required"], (
            f"grep_search input_schema.required must list {field!r} per issue #105 acceptance"
        )

    glob_prop = properties.get("glob")
    assert glob_prop is not None and glob_prop["type"] == "string", (
        "grep_search input_schema.properties.glob must be an optional string "
        "per issue #105 acceptance"
    )

    out_properties = doc["output_schema"]["properties"]
    matches_schema = out_properties["matches"]
    assert matches_schema["type"] == "array", (
        "grep_search output_schema.matches must be an array per issue #105 acceptance"
    )
    match_props = matches_schema["items"]["properties"]
    for field in ("file", "line", "text"):
        assert field in match_props, (
            f"grep_search output_schema.matches.items must declare {field!r} per issue #105 acceptance"
        )
    assert set(doc["output_schema"]["required"]) >= {"matches", "truncated"}, (
        "grep_search output_schema.required must list 'matches' and 'truncated' "
        "per issue #105 acceptance"
    )

    assert "truncation_policy" in doc, (
        "grep_search must publish a truncation_policy so the Phase 3 historical-log "
        "pruner has a per-call bound to fall back to (SECURITY.md threat #5)"
    )
    assert "stdlib" in doc["description"], (
        "grep_search description must defer to Python stdlib per docs/SECURITY.md \u00a71 threat #3"
    )
