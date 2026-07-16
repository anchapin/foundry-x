from __future__ import annotations

import pytest
from pydantic import ValidationError

from foundry_x.evolution.evolver import (
    InvalidStateTransition,
    ProposedEdit,
    ReviewState,
    ReviewStateMachine,
    _confine_to_harness_tree,
)

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


def test_harness_manifest_json_accepted():
    edit = _make("harness/manifest.json")
    assert edit.target_file == "harness/manifest.json"


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
        "harness/manifest.json/extra",  # file treated as a directory
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


def test_helper_returns_canonical_for_prompt():
    assert _confine_to_harness_tree("harness/system_prompt.txt") == "harness/system_prompt.txt"


def test_helper_returns_canonical_for_manifest():
    assert _confine_to_harness_tree("harness/manifest.json") == "harness/manifest.json"


def test_helper_raises_value_error_for_source_path():
    with pytest.raises(ValueError, match="harness"):
        _confine_to_harness_tree("src/foundry_x/trace/logger.py")


def test_helper_raises_value_error_for_traversal():
    with pytest.raises(ValueError):
        _confine_to_harness_tree("../../etc/passwd")


def test_helper_rejects_manifest_beneath_directory():
    with pytest.raises(ValueError):
        _confine_to_harness_tree("harness/manifest.json/subpath")


# --- review state machine (issue #497) ----------------------------------------


class TestReviewState:
    def test_review_state_values(self):
        assert ReviewState.PROPOSED.value == "PROPOSED"
        assert ReviewState.PENDING_REVIEW.value == "PENDING_REVIEW"
        assert ReviewState.APPROVED.value == "APPROVED"
        assert ReviewState.REJECTED.value == "REJECTED"


class TestReviewStateMachine:
    def test_can_apply_approved(self):
        sm = ReviewStateMachine()
        assert sm.can_apply(ReviewState.APPROVED) is True

    def test_cannot_apply_proposed(self):
        sm = ReviewStateMachine()
        assert sm.can_apply(ReviewState.PROPOSED) is False

    def test_cannot_apply_pending_review(self):
        sm = ReviewStateMachine()
        assert sm.can_apply(ReviewState.PENDING_REVIEW) is False

    def test_cannot_apply_rejected(self):
        sm = ReviewStateMachine()
        assert sm.can_apply(ReviewState.REJECTED) is False

    def test_valid_transition_proposed_to_pending_review(self):
        sm = ReviewStateMachine()
        sm.validate_transition(ReviewState.PROPOSED, ReviewState.PENDING_REVIEW)

    def test_valid_transition_pending_review_to_approved(self):
        sm = ReviewStateMachine()
        sm.validate_transition(ReviewState.PENDING_REVIEW, ReviewState.APPROVED)

    def test_valid_transition_pending_review_to_rejected(self):
        sm = ReviewStateMachine()
        sm.validate_transition(ReviewState.PENDING_REVIEW, ReviewState.REJECTED)

    def test_invalid_transition_proposed_to_approved(self):
        sm = ReviewStateMachine()
        with pytest.raises(InvalidStateTransition, match="PROPOSED.*APPROVED"):
            sm.validate_transition(ReviewState.PROPOSED, ReviewState.APPROVED)

    def test_invalid_transition_rejected_to_approved(self):
        sm = ReviewStateMachine()
        with pytest.raises(InvalidStateTransition, match="REJECTED.*APPROVED"):
            sm.validate_transition(ReviewState.REJECTED, ReviewState.APPROVED)

    def test_invalid_transition_approved_to_rejected(self):
        sm = ReviewStateMachine()
        with pytest.raises(InvalidStateTransition, match="APPROVED.*REJECTED"):
            sm.validate_transition(ReviewState.APPROVED, ReviewState.REJECTED)

    def test_terminal_approved_no_outbound(self):
        sm = ReviewStateMachine()
        with pytest.raises(InvalidStateTransition, match="APPROVED.*PROPOSED"):
            sm.validate_transition(ReviewState.APPROVED, ReviewState.PROPOSED)

    def test_terminal_rejected_no_outbound(self):
        sm = ReviewStateMachine()
        with pytest.raises(InvalidStateTransition, match="REJECTED.*PENDING_REVIEW"):
            sm.validate_transition(ReviewState.REJECTED, ReviewState.PENDING_REVIEW)

    def test_transition_calls_record_when_logger_configured(self):
        from unittest.mock import MagicMock

        mock_logger = MagicMock()
        sm = ReviewStateMachine(trace_logger=mock_logger, session_id="test-session")
        sm.transition("edit-1", ReviewState.PROPOSED, ReviewState.PENDING_REVIEW)

        mock_logger.record.assert_called_once_with(
            "test-session",
            "review_state_transition",
            {
                "edit_id": "edit-1",
                "from_state": "PROPOSED",
                "to_state": "PENDING_REVIEW",
            },
        )

    def test_transition_does_not_raise_on_valid_transition(self):
        sm = ReviewStateMachine()
        sm.transition("edit-1", ReviewState.PROPOSED, ReviewState.PENDING_REVIEW)


class TestProposedEditReviewState:
    def test_proposed_edit_default_review_state(self):
        edit = _make("harness/system_prompt.txt")
        assert edit.review_state == ReviewState.PROPOSED

    def test_proposed_edit_can_set_review_state(self):
        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="test",
            unified_diff=_DIFF,
            review_state=ReviewState.PENDING_REVIEW,
        )
        assert edit.review_state == ReviewState.PENDING_REVIEW
