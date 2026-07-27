"""Regression tests for issue #1051: injection-attempt edit template.

The ``injection-attempt`` failure class (ADR-0011) is security-critical.
When the Digester classifies a session as ``injection-attempt``, the
Evolver's template path must produce a non-empty ``ProposedEdit``
targeting ``system_prompt.txt`` with guidance specific to injection
prevention — not the generic ``unknown`` fallback.

These tests pin the contract:

1. ``Evolver.propose()`` for an ``injection-attempt`` failure emits a
   diff whose post-``git apply`` result adds injection-specific guidance
   to ``system_prompt.txt``.
2. The diff is a valid ``git apply`` unified diff under 200 lines
   (ADR-0004 §diff size cap).
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import Evolver


def _injection_attempt_failure() -> FailureReport:
    return FailureReport(
        session_id="sess-issue-1051",
        summary="InjectionFirewallHook suppressed tool results with prompt-injection markers.",
        failed_steps=[
            {
                "kind": "injection_blocked",
                "event_id": "e-inj-1051",
                "payload": {
                    "markers": ["ignore previous instructions"],
                    "tool": "read_file",
                },
            }
        ],
        suspected_causes=[
            (
                "InjectionFirewallHook suppressed 1 tool result(s) for prompt-injection "
                "markers (first block matched: ignore previous instructions)."
            )
        ],
        proposed_class="injection-attempt",
    )


def _build_harness(tmp_path: Path) -> Path:
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("You are a helpful agent.\n", encoding="utf-8")
    return harness_dir


def _copy_tree(src: Path, dst: Path) -> None:
    for root, _dirs, files in os.walk(src):
        root_path = Path(root)
        rel = root_path.relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for name in files:
            (dst / rel / name).write_bytes((root_path / name).read_bytes())


def _materialize_sandbox(tmp_path: Path, name: str, harness_dir: Path) -> Path:
    sandbox_parent = tmp_path / name
    sandbox_harness = sandbox_parent / "harness"
    sandbox_parent.mkdir()
    sandbox_harness.mkdir()
    _copy_tree(harness_dir, sandbox_harness)
    return sandbox_parent


def _apply_diff(parent_dir: Path, diff: str) -> None:
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=diff,
        cwd=parent_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"git apply failed: stderr={result.stderr!r} stdout={result.stdout!r}"
    )


def test_injection_attempt_template_targets_system_prompt(tmp_path: Path) -> None:
    """Acceptance criterion #1 (issue #1051).

    The edit produced by ``Evolver.propose()`` for an ``injection-attempt``
    failure must target ``system_prompt.txt`` and, after ``git apply``,
    append injection-specific guidance.
    """
    harness_dir = _build_harness(tmp_path)
    original_prompt = (harness_dir / "system_prompt.txt").read_text(encoding="utf-8")

    evolver = Evolver(model_adapter=None)
    edits = evolver.propose(harness_dir, _injection_attempt_failure())

    assert len(edits) == 1, "injection-attempt template must produce exactly one edit"
    edit = edits[0]
    assert edit.target_file == "harness/system_prompt.txt"

    sandbox_parent = _materialize_sandbox(tmp_path, "injection-sandbox", harness_dir)
    dest = sandbox_parent / "harness"

    _apply_diff(sandbox_parent, edit.unified_diff)

    patched_prompt = (dest / "system_prompt.txt").read_text(encoding="utf-8")

    assert original_prompt.rstrip("\n") in patched_prompt
    assert "injection" in patched_prompt.lower(), (
        "system_prompt must mention injection (issue #1051 AC #1)"
    )
    assert "firewall" in patched_prompt.lower(), (
        "system_prompt must reference the firewall hook (issue #1051 AC #1)"
    )


def test_injection_attempt_template_diff_under_200_lines(tmp_path: Path) -> None:
    """Acceptance criterion #3 (issue #1051).

    The template diff must be a valid git-apply unified diff under 200
    lines (ADR-0004 diff size cap).
    """
    harness_dir = _build_harness(tmp_path)

    evolver = Evolver(model_adapter=None)
    edits = evolver.propose(harness_dir, _injection_attempt_failure())
    assert len(edits) == 1
    edit = edits[0]

    diff_lines = edit.unified_diff.splitlines()
    assert len(diff_lines) < 200, f"template diff is {len(diff_lines)} lines; must be under 200"
    assert edit.unified_diff.startswith("--- a/"), "diff must have --- a/ header"
    assert "+++ b/" in edit.unified_diff, "diff must have +++ b/ header"


def test_injection_attempt_template_does_not_use_unknown_fallback(tmp_path: Path) -> None:
    """Guardrail: the injection-attempt template must not fall through to unknown.

    The ``unknown`` template provides generic clarification guidance with
    no injection-specific content. If ``injection-attempt`` fell through
    to ``unknown``, the patched prompt would lack injection/firewall
    keywords entirely.
    """
    harness_dir = _build_harness(tmp_path)

    evolver = Evolver(model_adapter=None)
    edits = evolver.propose(harness_dir, _injection_attempt_failure())
    assert len(edits) == 1

    diff_text = edits[0].unified_diff
    assert "clarification" not in diff_text.lower(), (
        "injection-attempt must not fall through to the unknown-class template"
    )
