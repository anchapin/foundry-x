from __future__ import annotations

import difflib
from pathlib import Path

import pytest
from pydantic import ValidationError

from foundry_x.evolution.critic import Critic
from foundry_x.evolution.evolver import ProposedEdit
from tests._harness_fixture import install_load_check_prerequisites


def _write_harness(tmp_path: Path, test_source: str) -> Path:
    harness_dir = tmp_path / "harness"
    tests_dir = harness_dir / "tests"
    tests_dir.mkdir(parents=True)
    (harness_dir / "system_prompt.txt").write_text("original\n")
    (harness_dir / "marker.txt").write_text("safe\n")
    (tests_dir / "test_gate.py").write_text(test_source)
    # load_check prerequisites (issue #187): the Critic gates on
    # harness/scripts/load_check.py before pytest, so the fixture must be
    # load-check-compliant.
    install_load_check_prerequisites(harness_dir)
    return harness_dir


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def _diff(relative_path: str, old: str, new: str) -> str:
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"a/{relative_path}",
            tofile=f"b/{relative_path}",
        )
    )


def test_evaluate_runs_pytest_against_sandbox_copy(tmp_path: Path):
    harness_dir = _write_harness(
        tmp_path,
        """
from pathlib import Path


def test_patch_is_visible_to_pytest():
    assert Path("system_prompt.txt").read_text() == "patched\\n"
    Path("pytest_was_here.txt").write_text("sandbox\\n")
""".lstrip(),
    )
    before = _snapshot(harness_dir)

    verdict = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_gate.py"],
    ).evaluate(_diff("system_prompt.txt", "original\n", "patched\n"))

    assert verdict.verdict is True
    assert "pytest" in verdict.passed_checks
    assert _snapshot(harness_dir) == before


def test_pytest_failure_rejects_with_failed_check(tmp_path: Path):
    harness_dir = _write_harness(
        tmp_path,
        """
from pathlib import Path


def test_marker_stays_safe():
    assert Path("marker.txt").read_text() == "safe\\n"
""".lstrip(),
    )

    verdict = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_gate.py"],
    ).evaluate(_diff("marker.txt", "safe\n", "broken\n"))

    assert verdict.verdict is False
    assert "git apply" in verdict.passed_checks
    assert "pytest" in verdict.failed_checks


def test_clean_diff_approves_with_pytest_check(
    tmp_path: Path,
    proposed_edit: ProposedEdit,
):
    harness_dir = _write_harness(
        tmp_path,
        """
def test_clean_gate_passes():
    assert True
""".lstrip(),
    )
    relative_target = Path(proposed_edit.target_file).relative_to("harness").as_posix()
    edit = ProposedEdit(
        target_file=proposed_edit.target_file,
        rationale=proposed_edit.rationale,
        unified_diff=_diff(relative_target, "original\n", "clean\n"),
    )

    verdict = Critic(
        harness_dir=harness_dir,
        pytest_args=["-q", "tests/test_gate.py"],
    ).evaluate(edit.unified_diff)

    assert verdict.verdict is True
    assert "pytest" in verdict.passed_checks
    assert verdict.failed_checks == []


def test_proposed_edit_rejects_bare_hunk_diff():
    """ProposedEdit rejects a unified diff missing git-apply headers (issue #257)."""
    bare_hunk = "@@ -1 +1 @@\n-old line\n+new line\n"
    with pytest.raises(ValidationError) as exc_info:
        ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Test rationale",
            unified_diff=bare_hunk,
        )
    assert "missing '--- a/' header" in str(exc_info.value)


def test_proposed_edit_rejects_diff_missing_old_header():
    """ProposedEdit rejects a unified diff missing the '--- a/' header."""
    diff_missing_old = "+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old line\n+new line\n"
    with pytest.raises(ValidationError) as exc_info:
        ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Test rationale",
            unified_diff=diff_missing_old,
        )
    assert "missing '--- a/' header" in str(exc_info.value)


def test_proposed_edit_rejects_diff_missing_new_header():
    """ProposedEdit rejects a unified diff missing the '+++ b/' header."""
    diff_missing_new = "--- a/harness/system_prompt.txt\n@@ -1 +1 @@\n-old line\n+new line\n"
    with pytest.raises(ValidationError) as exc_info:
        ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Test rationale",
            unified_diff=diff_missing_new,
        )
    assert "missing '+++ b/' header" in str(exc_info.value)


def test_proposed_edit_accepts_well_formed_diff():
    """ProposedEdit accepts a well-formed unified diff with git-apply headers."""
    well_formed = (
        "--- a/harness/system_prompt.txt\n"
        "+++ b/harness/system_prompt.txt\n"
        "@@ -1 +1 @@\n"
        "-old line\n"
        "+new line\n"
    )
    edit = ProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="Test rationale",
        unified_diff=well_formed,
    )
    assert edit.unified_diff == well_formed


def test_proposed_edit_accepts_multi_hunk_diff():
    """ProposedEdit accepts a multi-hunk unified diff with headers."""
    multi_hunk = (
        "--- a/harness/system_prompt.txt\n"
        "+++ b/harness/system_prompt.txt\n"
        "@@ -1,3 +1,3 @@\n"
        "-old line 1\n"
        "+new line 1\n"
        " unchanged\n"
        "@@ -10,2 +10,2 @@\n"
        "-old line 10\n"
        "+new line 10\n"
    )
    edit = ProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="Test rationale",
        unified_diff=multi_hunk,
    )
    assert edit.unified_diff == multi_hunk
