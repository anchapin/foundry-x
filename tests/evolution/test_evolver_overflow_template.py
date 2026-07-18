"""Regression tests for issue #892.

The ``context-overflow`` template edit targets ``harness/manifest.json``.
Before the fix the template appended a raw JSON fragment to the file's
existing content, producing syntactically invalid JSON (a trailing
``"context_pruning": {...}`` after the closing ``}``). The Critic's
``load_check`` gate then rejected the proposal as a false-negative even
though the edit was semantically correct.

These tests pin the JSON-aware patching contract:

1. ``Evolver.propose()`` for a ``context-overflow`` failure emits a diff
   whose post-``git apply`` result is valid JSON.
2. The patched manifest carries the lowered ``token_threshold`` while
   preserving every pre-existing key.
3. The patched harness passes ``harness/scripts/load_check.py`` (the
   exact gate the Critic runs in :func:`Critic.evaluate`).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    Evolver,
    _apply_json_merge_patch,
    _deep_merge,
)
from tests._harness_fixture import install_load_check_prerequisites


def _context_overflow_failure() -> FailureReport:
    return FailureReport(
        session_id="sess-issue-892",
        summary="Agent loop reached max_steps before producing a final answer.",
        failed_steps=[
            {
                "kind": "outcome",
                "event_id": "e-co-892",
                "payload": {"status": "truncated", "reason": "max_steps", "steps": 10},
            }
        ],
        suspected_causes=[
            "Agent loop reached max_steps (steps=10) before producing a final answer."
        ],
        proposed_class="context-overflow",
    )


def _build_load_check_harness(tmp_path: Path) -> Path:
    """Create a load-check-compliant harness whose manifest carries context_pruning.

    ``install_load_check_prerequisites`` seeds a minimal manifest; we then
    overwrite it with a richer one that mirrors the production layout (a
    ``context_pruning`` block the template is expected to lower).
    """
    harness_dir = tmp_path / "harness"
    tests_dir = harness_dir / "tests"
    tests_dir.mkdir(parents=True)
    (harness_dir / "system_prompt.txt").write_text("You are a helpful agent.\n", encoding="utf-8")
    (tests_dir / "test_gate.py").write_text(
        "def test_placeholder():\n    assert True\n",
        encoding="utf-8",
    )
    install_load_check_prerequisites(harness_dir)
    # Overwrite the minimal manifest with the production-shaped one AFTER
    # install_load_check_prerequisites so the richer shape survives.
    manifest = {
        "version": "0.1.0",
        "model_target": "minimax-coding-plan/MiniMax-M3",
        "hooks": [],
        "context_pruning": {
            "token_threshold": 8192,
            "event_threshold": 200,
        },
        "skills": [],
        "skill_inventory": [],
    }
    (harness_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return harness_dir


def _apply_diff(parent_dir: Path, diff: str) -> None:
    """Apply a unified diff in ``parent_dir`` (mirrors the Critic's gate)."""
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=diff,
        cwd=parent_dir,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"git apply failed: stderr={result.stderr!r} stdout={result.stdout!r}"
    )


def _copy_tree(src: Path, dst: Path) -> None:
    """Recursively copy a directory tree preserving files and empty dirs."""
    for root, dirs, files in src.walk():
        rel = root.relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            (dst / rel / name).write_bytes((root / name).read_bytes())


def _materialize_sandbox(tmp_path: Path, name: str, harness_dir: Path) -> Path:
    """Build ``tmp_path/name/harness`` mirroring the Critic's sandbox layout.

    The Critic copies ``harness_dir`` into ``TemporaryDirectory()/harness``
    and runs ``git apply`` with ``cwd=sandbox_root.parent``; the diff paths
    (``a/harness/...`` / ``b/harness/...``) resolve against that parent.
    """
    sandbox_parent = tmp_path / name
    sandbox_harness = sandbox_parent / "harness"
    sandbox_parent.mkdir()
    sandbox_harness.mkdir()
    _copy_tree(harness_dir, sandbox_harness)
    return sandbox_parent


# --- Unit tests for the JSON merge-patch helpers ---------------------------


def test_deep_merge_recurses_into_nested_dicts() -> None:
    target = {"a": {"x": 1, "y": 2}, "b": 3}
    _deep_merge(target, {"a": {"y": 99, "z": 100}, "c": 4})
    assert target == {"a": {"x": 1, "y": 99, "z": 100}, "b": 3, "c": 4}


def test_deep_merge_overwrites_non_dict_values() -> None:
    target = {"a": [1, 2], "b": "old"}
    _deep_merge(target, {"a": [3], "b": "new"})
    assert target == {"a": [3], "b": "new"}


def test_apply_json_merge_patch_preserves_unrelated_keys() -> None:
    original = '{"version": "0.1.0", "hooks": ["base"], "skills": []}\n'
    patched = _apply_json_merge_patch(
        original, {"context_pruning": {"token_threshold": 6144, "event_threshold": 200}}
    )
    doc = json.loads(patched)
    assert doc["version"] == "0.1.0"
    assert doc["hooks"] == ["base"]
    assert doc["skills"] == []
    assert doc["context_pruning"]["token_threshold"] == 6144
    # Output must be valid JSON and end with a newline (canonical layout).
    assert patched.endswith("\n")


def test_apply_json_merge_patch_lowers_existing_threshold() -> None:
    original = '{"context_pruning": {"token_threshold": 8192, "event_threshold": 200}}\n'
    patched = _apply_json_merge_patch(
        original, {"context_pruning": {"token_threshold": 6144, "event_threshold": 200}}
    )
    doc = json.loads(patched)
    assert doc["context_pruning"]["token_threshold"] == 6144
    assert doc["context_pruning"]["event_threshold"] == 200


def test_apply_json_merge_patch_rejects_non_object_top_level() -> None:
    with pytest.raises(ValueError, match="top-level object"):
        _apply_json_merge_patch("[1, 2, 3]\n", {"a": 1})


# --- End-to-end template -> diff -> manifest tests -------------------------


def test_context_overflow_template_diff_yields_valid_json(tmp_path: Path) -> None:
    """Acceptance criterion #1 + #3 (issue #892).

    The diff produced by ``Evolver.propose()`` for a ``context-overflow``
    failure must, after ``git apply``, leave ``manifest.json`` as valid JSON
    that still parses with ``json.loads`` and carries the lowered threshold.
    """
    harness_dir = _build_load_check_harness(tmp_path)
    original_manifest = json.loads((harness_dir / "manifest.json").read_text(encoding="utf-8"))
    assert original_manifest["context_pruning"]["token_threshold"] == 8192

    evolver = Evolver(model_adapter=None)
    edits = evolver.propose(harness_dir, _context_overflow_failure())

    assert len(edits) == 1, "context-overflow template must produce exactly one edit"
    edit = edits[0]
    assert edit.target_file == "harness/manifest.json"

    # Apply the proposal to a throwaway sandbox copy so the source harness
    # is untouched. The layout (sandbox/harness) mirrors the Critic.
    sandbox_parent = _materialize_sandbox(tmp_path, "json-sandbox", harness_dir)
    dest = sandbox_parent / "harness"

    _apply_diff(sandbox_parent, edit.unified_diff)

    patched_text = (dest / "manifest.json").read_text(encoding="utf-8")
    patched = json.loads(patched_text)  # acceptance criterion #1: must parse

    # The token_threshold is lowered; surrounding keys survive.
    assert patched["context_pruning"]["token_threshold"] == 6144
    assert patched["context_pruning"]["event_threshold"] == 200
    assert patched["version"] == original_manifest["version"]
    assert patched["hooks"] == original_manifest["hooks"]
    assert patched["skills"] == original_manifest["skills"]


def test_context_overflow_template_passes_load_check(tmp_path: Path) -> None:
    """Acceptance criterion #2 (issue #892).

    After applying the ``context-overflow`` template diff, the sandboxed
    harness must pass ``harness/scripts/load_check.py`` — the exact gate
    the Critic runs in :func:`Critic.evaluate` (gate 3, ADR-0004).
    """
    harness_dir = _build_load_check_harness(tmp_path)

    evolver = Evolver(model_adapter=None)
    edits = evolver.propose(harness_dir, _context_overflow_failure())
    assert len(edits) == 1
    edit = edits[0]

    sandbox_parent = _materialize_sandbox(tmp_path, "load-check-sandbox", harness_dir)
    sandbox_harness = sandbox_parent / "harness"
    _apply_diff(sandbox_parent, edit.unified_diff)

    load_check_script = sandbox_harness / "scripts" / "load_check.py"
    assert load_check_script.is_file(), "load_check.py missing from sandbox"

    result = subprocess.run(
        [
            sys.executable,
            str(load_check_script),
            "--harness-dir",
            str(sandbox_harness),
        ],
        cwd=sandbox_harness,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"load_check failed after context-overflow edit:\n"
        f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
    )


def test_text_template_still_appends_lines(tmp_path: Path) -> None:
    """Guardrail: JSON-aware path must not regress plain-text templates.

    The ``wrong-tool`` template appends a bullet to ``system_prompt.txt``.
    That contract is unchanged by the JSON fix.
    """
    harness_dir = _build_load_check_harness(tmp_path)
    original_prompt = (harness_dir / "system_prompt.txt").read_text(encoding="utf-8")

    failure = FailureReport(
        session_id="sess-wrong-tool",
        summary="Agent invoked a tool outside its declared schema.",
        failed_steps=[
            {
                "kind": "tool_call",
                "event_id": "e-wt",
                "payload": {"name": "unknown_tool"},
            }
        ],
        suspected_causes=["Agent invoked unknown_tool which is not in the schema."],
        proposed_class="wrong-tool",
    )

    edits = Evolver(model_adapter=None).propose(harness_dir, failure)
    assert len(edits) == 1
    assert edits[0].target_file == "harness/system_prompt.txt"
    # Sanity: appending still works for text targets.
    assert "available-tool schema" in edits[0].unified_diff
    # Manifest is untouched by this template.
    assert "manifest.json" not in edits[0].target_file
    assert (harness_dir / "system_prompt.txt").read_text(encoding="utf-8") == original_prompt
