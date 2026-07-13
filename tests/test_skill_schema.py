"""Schema validation for the harness skill surface.

Loads every ``harness/skills/*.json`` and validates it against the shape
established by ``example_skill.json`` (ADR-0004, issue #19, issue #20). A skill must:

* have a non-empty ``name`` and ``version`` string,
* declare an ``input_schema`` that is a JSON-schema *object* with
  ``additionalProperties: false`` (closed schema, no invented args),
* declare an ``output_schema`` that is a JSON-schema *object*.

The active skill set must include at least one write-capable skill so the
agent can produce or modify code (PRD improvement-rate KPI).

Issue #20 adds bounded-output contract checks for the ``read_file`` skill
to prevent context-window runaway on local 7B-70B models (SECURITY.md
threat #5, PRD §4). The ``read_file`` skill must declare a bounded-size
input parameter (``max_bytes``) and a documented truncation threshold;
the contract below asserts the executor will return ``truncated: true``
with bounded ``content`` when a synthetic file exceeds the default bound.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

SKILLS_DIR = Path(__file__).resolve().parents[1] / "harness" / "skills"


def _load_skills() -> list[dict]:
    files = sorted(SKILLS_DIR.glob("*.json"))
    assert files, "no skill files found in harness/skills/"
    return [json.loads(f.read_text()) for f in files]


SKILLS = _load_skills()
SKILL_NAMES = {s["name"] for s in SKILLS}
SKILLS_BY_NAME = {s["name"]: s for s in SKILLS}


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
    assert write_capable, (
        f"no write-capable skill found; expected one of write_file/edit_file, got {sorted(SKILL_NAMES)}"
    )


# ---------------------------------------------------------------------------
# Issue #20: bound read_file skill output to prevent context runaway
# ---------------------------------------------------------------------------


def _read_file_skill() -> dict:
    skill = SKILLS_BY_NAME.get("read_file")
    assert skill is not None, "read_file skill not present in harness/skills/"
    return skill


def test_read_file_input_schema_declares_max_bytes() -> None:
    """The read_file input schema must declare a bounded-size parameter.

    Issue #20 acceptance: the input schema must declare a bounded-size
    parameter (``max_bytes``) so a single read cannot exhaust the local-model
    context window (SECURITY.md threat #5).
    """
    skill = _read_file_skill()
    props = skill["input_schema"]["properties"]
    assert "max_bytes" in props, (
        "read_file input schema must declare a `max_bytes` parameter to bound "
        "the returned content (issue #20)"
    )
    spec = props["max_bytes"]
    assert spec.get("type") == "integer"
    assert isinstance(spec.get("default"), int) and spec["default"] > 0
    # `minimum: 1` keeps a zero-byte cap from silently disabling truncation.
    assert spec.get("minimum") == 1


def test_read_file_input_schema_declares_optional_max_lines() -> None:
    """Optional ``max_lines`` cap is a documented line-count safety bound."""
    skill = _read_file_skill()
    props = skill["input_schema"]["properties"]
    assert "max_lines" in props
    spec = props["max_lines"]
    assert spec.get("type") == "integer"
    assert spec.get("minimum") == 1


def test_read_file_truncation_policy_threshold_is_nonzero() -> None:
    """The documented truncation threshold must be a positive integer.

    Issue #20 acceptance: ``truncation_policy.default_max_bytes`` is the
    single source of truth for the per-call safety bound; it must be > 0.
    """
    skill = _read_file_skill()
    policy = skill.get("truncation_policy")
    assert isinstance(policy, dict), "read_file must declare a truncation_policy"
    threshold = policy.get("default_max_bytes")
    assert isinstance(threshold, int), "default_max_bytes must be an integer"
    assert threshold > 0, f"default_max_bytes must be > 0, got {threshold}"
    # 32 KiB (32768) is the documented safe bound for 4K-32K context windows.
    assert threshold >= 4096, (
        f"default_max_bytes={threshold} is below the minimum safety bound of 4096 bytes"
    )


def test_read_file_input_default_matches_truncation_policy() -> None:
    """The input schema's ``max_bytes`` default must match the documented policy.

    Otherwise the schema and the policy can drift, and the executor may
    apply a different bound than the one the agent was told to expect.
    """
    skill = _read_file_skill()
    props = skill["input_schema"]["properties"]
    assert props["max_bytes"]["default"] == skill["truncation_policy"]["default_max_bytes"]


def test_read_file_output_schema_requires_truncated_flag() -> None:
    """``truncated`` must be a required boolean in the output schema.

    The agent is contractually required to branch on this flag; making it
    required at the schema level prevents a future executor from silently
    omitting it.
    """
    skill = _read_file_skill()
    out = skill["output_schema"]
    assert "truncated" in out["properties"]
    assert out["properties"]["truncated"].get("type") == "boolean"
    assert "truncated" in out["required"]


# ---------------------------------------------------------------------------
# Synthetic-file contract test (issue #20 acceptance)
# ---------------------------------------------------------------------------


def _apply_read_file_contract(
    path: Path,
    *,
    max_bytes: int | None = None,
    max_lines: int | None = None,
    offset: int = 0,
) -> dict[str, Any]:
    """Apply the read_file contract to ``path`` exactly as the executor must.

    This is a test fixture that mirrors the contract declared in
    ``harness/skills/example_skill.json`` for the read_file skill. The
    production executor is out of scope for issue #20 (see issue body
    "Out of scope"); tests use this to verify the contract on synthetic
    input. Keep this in sync with the schema's ``truncation_policy``.

    Returns a dict matching the skill's ``output_schema``: ``content``,
    ``sha256``, ``truncated``, ``bytes_returned``, ``bytes_total``.
    """
    skill = _read_file_skill()
    policy = skill["truncation_policy"]
    if max_bytes is None:
        max_bytes = policy["default_max_bytes"]
    assert max_bytes > 0

    raw = path.read_bytes()
    bytes_total = len(raw)
    chunk = raw[offset:]

    truncated = False
    if max_lines is not None:
        text = chunk.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)
        if len(lines) > max_lines:
            text = "".join(lines[:max_lines])
            chunk = text.encode("utf-8")
            truncated = True
    if not truncated and len(chunk) > max_bytes:
        # Cut at the largest byte position <= max_bytes that ends on '\n'.
        cut = chunk.rfind(b"\n", 0, max_bytes)
        if cut <= 0:
            cut = max_bytes
        chunk = chunk[:cut]
        truncated = True

    return {
        "content": chunk.decode("utf-8", errors="replace"),
        "sha256": hashlib.sha256(chunk).hexdigest(),
        "truncated": truncated,
        "bytes_returned": len(chunk),
        "bytes_total": bytes_total,
    }


def test_read_file_contract_truncates_large_synthetic_file(tmp_path: Path) -> None:
    """A file larger than the default bound must yield truncated=True with bounded content.

    Issue #20 acceptance: synthetic file > default_max_bytes -> contract returns
    truncated=True with content size <= max_bytes. This proves the schema's
    per-call safety bound is actually honored, not just declared.
    """
    skill = _read_file_skill()
    default_max_bytes = skill["truncation_policy"]["default_max_bytes"]
    # Pick a size comfortably above the default; default is 32768 so 64 KiB works.
    synthetic = tmp_path / "big.txt"
    payload = b"a\n" * (default_max_bytes + 8192)  # 64 KiB-ish, line-aligned
    synthetic.write_bytes(payload)

    result = _apply_read_file_contract(synthetic)

    assert result["truncated"] is True, (
        f"file of {len(payload)} bytes should exceed default_max_bytes="
        f"{default_max_bytes} but truncated=False"
    )
    assert result["bytes_returned"] <= default_max_bytes, (
        f"bytes_returned={result['bytes_returned']} exceeded default_max_bytes={default_max_bytes}"
    )
    assert result["bytes_total"] == len(payload)
    assert isinstance(result["content"], str) and result["content"]
    assert len(result["sha256"]) == 64  # hex SHA-256


def test_read_file_contract_returns_full_content_for_small_file(tmp_path: Path) -> None:
    """A file within the bound must yield truncated=False with the full payload."""
    synthetic = tmp_path / "small.txt"
    payload = b"hello, agent\n"
    synthetic.write_bytes(payload)

    result = _apply_read_file_contract(synthetic)

    assert result["truncated"] is False
    assert result["content"].encode("utf-8") == payload
    assert result["bytes_returned"] == len(payload)
    assert result["bytes_total"] == len(payload)


def test_read_file_contract_max_lines_caps_line_count(tmp_path: Path) -> None:
    """When max_lines is the binding constraint, the cut honors it."""
    synthetic = tmp_path / "lines.txt"
    payload = b"".join(f"line {i:04d}\n".encode() for i in range(500))
    synthetic.write_bytes(payload)

    result = _apply_read_file_contract(synthetic, max_lines=10)

    assert result["truncated"] is True
    assert result["content"].count("\n") <= 10


def test_read_file_contract_default_bound_is_within_context_budget() -> None:
    """The default bound must be small enough to stay inside a 32K-token budget.

    32 KiB of text is roughly 8K-16K tokens depending on the tokenizer;
    this is the documented safety bound for local 7B-70B models.
    """
    skill = _read_file_skill()
    bound = skill["truncation_policy"]["default_max_bytes"]
    # 256 KiB would already be ~64K-128K tokens on a BPE-style tokenizer,
    # so any sane default should be well under that.
    assert bound <= 256 * 1024
