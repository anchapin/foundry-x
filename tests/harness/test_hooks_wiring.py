"""Verify the ProposedEdit for issue #203 wires context_pruning correctly.

Issue #203 asks to wire ``context_pruning`` into ``harness/hooks/__init__.py``
so that ``import harness.hooks`` exposes the context_pruning symbols the
manifest promises. Because ``harness/hooks/__init__.py`` is harness DNA
(AGENTS.md section 2), we do NOT edit it directly. Instead the change is
captured as a ``ProposedEdit`` JSON artifact at
``harness/proposed_edits/issue-203-context-pruning-wiring.json`` and must
be routed through the Evolver->Critic pipeline (ADR-0004).

These tests verify the proposal is correct **without** modifying the live
DNA. They:

1. Load the JSON artifact and validate it as a ``ProposedEdit`` (confirms
   the ``target_file`` confinement guardrail from ADR-0004).
2. Apply the unified diff to a sandbox copy of the harness hooks tree and
   import the patched module, verifying ``context_pruning`` symbols appear
   in ``harness.hooks.__all__``.
3. Assert the live (un-patched) ``harness.hooks.__all__`` does NOT yet
   contain context_pruning — proving we have not secretly hand-edited the
   DNA.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from foundry_x.evolution.evolver import ProposedEdit


REPO_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_PATH = REPO_ROOT / "harness" / "proposed_edits" / "issue-203-context-pruning-wiring.json"
LIVE_INIT = REPO_ROOT / "harness" / "hooks" / "__init__.py"


# Symbols the manifest promises and the proposal must expose.
_EXPECTED_CONTEXT_PRUNING_EXPORTS = frozenset(
    {"ContextPruningHook", "DEFAULT_THRESHOLD", "Pruner", "Tracer", "register_into"}
)


def _load_proposal() -> dict:
    """Read the ProposedEdit JSON artifact from disk."""
    return json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))


# --- 1. The artifact is a well-formed ProposedEdit --------------------------


def test_proposal_artifact_exists() -> None:
    """The ProposedEdit JSON must exist on disk for the Evolver->Critic loop
    to consume it."""
    assert PROPOSAL_PATH.exists(), (
        f"Expected ProposedEdit artifact at {PROPOSAL_PATH}; "
        "harness/hooks/__init__.py is DNA and cannot be hand-edited (AGENTS.md section 2)."
    )


def test_proposal_validates_as_proposed_edit() -> None:
    """The JSON must round-trip into a ``ProposedEdit`` pydantic model,
    which enforces the ADR-0004 ``target_file`` confinement guardrail."""
    raw = _load_proposal()
    edit = ProposedEdit(
        target_file=raw["target_file"],
        rationale=raw["rationale"],
        unified_diff=raw["unified_diff"],
    )
    assert edit.target_file == "harness/hooks/__init__.py"


def test_proposal_targets_correct_file() -> None:
    """The proposal must target ``harness/hooks/__init__.py`` — not a
    sibling file, not the directory itself."""
    raw = _load_proposal()
    assert raw["target_file"] == "harness/hooks/__init__.py"


def test_proposal_rationale_cites_guardrail() -> None:
    """The rationale must acknowledge that this file is harness DNA and
    explain why a ProposedEdit (not a direct edit) is used."""
    raw = _load_proposal()
    rationale = raw["rationale"].lower()
    assert "dna" in rationale or "agents.md" in rationale
    assert "adr-0004" in rationale or "critic" in rationale


# --- 2. The diff, once applied, exposes context_pruning ---------------------


def _apply_and_import(tmp_path: Path) -> set[str]:
    """Copy the live hooks tree to ``tmp_path``, apply the ProposedEdit
    diff, import the patched module in a subprocess, and return the
    patched ``__all__``.

    A subprocess is used (mirroring ``tests/harness/test_load_check.py``)
    so the already-imported live ``harness.hooks`` in the test process is
    not disturbed and no mutation leaks across tests.
    """
    sandbox_hooks = tmp_path / "harness" / "hooks"
    sandbox_hooks.parent.mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "harness" / "hooks", sandbox_hooks)

    raw = _load_proposal()
    diff_text = raw["unified_diff"]

    # Apply via git apply (same mechanism the Critic uses, critic.py:97).
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=diff_text,
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )

    # Import the patched module in a clean subprocess and dump __all__.
    runner = textwrap.dedent(
        f"""
        import json, sys
        sys.path.insert(0, {str(tmp_path)!r})
        import harness.hooks
        print(json.dumps(sorted(harness.hooks.__all__)))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PYTHONPATH": str(tmp_path), "PATH": ""},
    )
    assert (
        proc.returncode == 0
    ), f"Patched harness.hooks failed to import: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    return set(json.loads(proc.stdout.strip()))


def test_diff_applies_and_imports(tmp_path: Path) -> None:
    """The ProposedEdit diff must apply cleanly to the live hooks tree and
    the patched ``harness.hooks`` must import without error."""
    all_names = _apply_and_import(tmp_path)
    # If we got here, the import succeeded.
    assert len(all_names) > 0


def test_diff_exposes_context_pruning_exports(tmp_path: Path) -> None:
    """After applying the proposal, ``harness.hooks.__all__`` must contain
    every symbol ``context_pruning.py`` exports (issue #203 acceptance)."""
    all_names = _apply_and_import(tmp_path)
    missing = _EXPECTED_CONTEXT_PRUNING_EXPORTS - all_names
    assert (
        not missing
    ), f"context_pruning exports missing from patched harness.hooks.__all__: {missing}"


def test_diff_exposes_context_pruning_module(tmp_path: Path) -> None:
    """After applying the proposal, the ``context_pruning`` submodule must
    be accessible as ``harness.hooks.context_pruning``."""
    sandbox_hooks = tmp_path / "harness" / "hooks"
    sandbox_hooks.parent.mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "harness" / "hooks", sandbox_hooks)

    raw = _load_proposal()
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=raw["unified_diff"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )

    runner = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import harness.hooks
        from harness.hooks import context_pruning
        print(context_pruning.ContextPruningHook.__name__)
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PYTHONPATH": str(tmp_path), "PATH": ""},
    )
    assert proc.returncode == 0, (
        f"Cannot access harness.hooks.context_pruning after patch: "
        f"stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    assert "ContextPruningHook" in proc.stdout


def test_diff_does_not_self_register(tmp_path: Path) -> None:
    """context_pruning must NOT self-register on import (unlike
    InjectionFirewallHook). The hook needs a session_id and closures that
    only the runner can supply; the runner calls register_into(). This is
    explicitly listed as out-of-scope in issue #203."""
    sandbox_hooks = tmp_path / "harness" / "hooks"
    sandbox_hooks.parent.mkdir(parents=True)
    shutil.copytree(REPO_ROOT / "harness" / "hooks", sandbox_hooks)

    raw = _load_proposal()
    subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=raw["unified_diff"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
        check=True,
    )

    runner = textwrap.dedent(
        f"""
        import sys
        sys.path.insert(0, {str(tmp_path)!r})
        import harness.hooks
        reg = harness.hooks.get_registry()
        hook_names = [type(h).__name__ for h in getattr(reg, '_hooks', [])]
        print(','.join(hook_names))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", runner],
        capture_output=True,
        text=True,
        timeout=30,
        env={"PYTHONPATH": str(tmp_path), "PATH": ""},
    )
    assert proc.returncode == 0, f"stderr={proc.stderr!r}"
    registered = [n for n in proc.stdout.strip().split(",") if n]
    assert "ContextPruningHook" not in registered, (
        "context_pruning self-registered on import; it should NOT (issue #203 "
        "out-of-scope: the runner calls register_into() with a session_id)"
    )


# --- 3. The live DNA is unmodified -----------------------------------------


def test_live_dna_does_not_expose_context_pruning() -> None:
    """The LIVE ``harness/hooks/__init__.py`` (un-patched) must NOT yet
    contain context_pruning exports. This proves we have not hand-edited
    the DNA — the change is captured only in the ProposedEdit artifact and
    must go through the Critic gate (ADR-0004) before it ships."""
    text = LIVE_INIT.read_text(encoding="utf-8")
    # The import line for context_pruning must not exist in the live file.
    assert "from .context_pruning import" not in text, (
        "harness/hooks/__init__.py appears to have been hand-edited to import "
        "context_pruning. This violates AGENTS.md section 2 — harness DNA must "
        "be routed through the Evolver->Critic pipeline, not edited directly."
    )


def test_live_dna_all_unchanged() -> None:
    """The live ``__all__`` must not contain context_pruning symbols yet."""
    text = LIVE_INIT.read_text(encoding="utf-8")
    for sym in _EXPECTED_CONTEXT_PRUNING_EXPORTS:
        assert f'"{sym}"' not in text, (
            f"harness/hooks/__init__.py __all__ already contains {sym!r}; "
            "the DNA should be unmodified."
        )
