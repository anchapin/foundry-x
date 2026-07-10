from __future__ import annotations

import pytest
from pydantic import ValidationError

from foundry_x.evolution.evolver import ProposedEdit, _confine_to_harness_tree

# A minimal valid diff body; these tests exercise ``target_file`` confinement
# (ADR-0004), not diff parsing, so a simple non-blank string suffices.
_DIFF = (
    "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n"
)


def _make(target_file: str) -> ProposedEdit:
    return ProposedEdit(
        target_file=target_file,
        rationale="tighten tool guidance",
        unified_diff=_DIFF,
    )


# --- acceptance cases from issue #18 ----------------------------------------


def test_harness_system_prompt_accepted():
    edit = _make("harness/system_prompt.txt")
    assert edit.target_file == "harness/system_prompt.txt"


def test_source_file_rejected():
    with pytest.raises(ValidationError, match="harness"):
        _make("src/foundry_x/trace/logger.py")


def test_directory_traversal_rejected():
    with pytest.raises(ValidationError, match=r"\.\.|harness"):
        _make("../../../etc/passwd")


# --- the three allowed subtrees --------------------------------------------


def test_hooks_subtree_accepted():
    edit = _make("harness/hooks/post_step.py")
    assert edit.target_file == "harness/hooks/post_step.py"


def test_skills_subtree_accepted_deep():
    edit = _make("harness/skills/retrieval/SKILL.md")
    assert edit.target_file == "harness/skills/retrieval/SKILL.md"


# --- escapes and non-editable entries --------------------------------------


@pytest.mark.parametrize(
    "bad",
    [
        "/etc/passwd",  # absolute path
        "harness/../src/foundry_x/evolution/evolver.py",  # traversal back out
        "harness",  # the root itself is not a file
        "harness/VERSION",  # harness file that is NOT editable
        "harness/hooks",  # directory, not a file
        "harness/skills",  # directory, not a file
        "harness/system_prompt.txt/extra",  # file treated as a directory
        "harness/README.md",  # arbitrary non-allowed harness entry
    ],
)
def test_non_allowed_targets_rejected(bad: str):
    with pytest.raises(ValidationError):
        _make(bad)


# --- canonicalization -------------------------------------------------------


def test_canonical_form_collapses_dot_and_dotdot():
    # ./ and intra-harness .. segments must collapse to one canonical path.
    edit = _make("harness/./hooks/../hooks/post_step.py")
    assert edit.target_file == "harness/hooks/post_step.py"


def test_backslash_component_rejected():
    # Path components must not smuggle separators or NULs.
    with pytest.raises(ValidationError):
        _make("harness/hooks/subdir\\x.py")


def test_blank_target_still_rejected():
    with pytest.raises(ValidationError):
        _make("")


# --- the pure helper is independently testable ------------------------------


def test_helper_returns_canonical_for_valid_path():
    assert _confine_to_harness_tree("harness/system_prompt.txt") == ("harness/system_prompt.txt")


def test_helper_raises_value_error_for_source_path():
    with pytest.raises(ValueError, match="harness"):
        _confine_to_harness_tree("src/foundry_x/trace/logger.py")


def test_helper_raises_value_error_for_traversal():
    with pytest.raises(ValueError):
        _confine_to_harness_tree("../../etc/passwd")
