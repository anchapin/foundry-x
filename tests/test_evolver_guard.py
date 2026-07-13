from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    PROPOSED_EDIT_KIND,
    Evolver,
    EvolverGuardError,
    ProposedEdit,
)
from foundry_x.trace.logger import TraceLogger


def _make_diff(*lines: str) -> str:
    header = "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n"
    hunk = "@@ -0,0 +1 @@\n"
    return header + hunk + "".join(f"+{line}\n" for line in lines)


def _edit(diff: str) -> ProposedEdit:
    return ProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="tighten tool guidance",
        unified_diff=diff,
    )


def test_record_proposal_emits_trace_event(tmp_path):
    logger = TraceLogger(tmp_path / "trace.db")
    edit = _edit(_make_diff("be precise"))
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_proposals(edit=edit)

    events = list(logger.iter_events(session_id, kind=PROPOSED_EDIT_KIND))
    assert len(events) == 1
    assert events[0].payload == edit.model_dump(mode="json")


def test_defaults_match_security_doc():
    e = Evolver()
    assert e.max_proposals_per_hour == 10
    assert e.max_diff_lines == 200


def test_invalid_limits_rejected():
    with pytest.raises(EvolverGuardError, match="max_proposals_per_hour"):
        Evolver(max_proposals_per_hour=0)
    with pytest.raises(EvolverGuardError, match="max_diff_lines"):
        Evolver(max_diff_lines=0)


def test_oversized_diff_rejected():
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=5)
    big_diff = _make_diff(*[f"line {i}" for i in range(10)])
    with pytest.raises(EvolverGuardError, match="diff too large"):
        e._validate_edit(_edit(big_diff))


def test_diff_at_exact_cap_passes():
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=5)
    e._validate_edit(_edit(_make_diff("l0", "l1")))


def test_rate_limit_triggers_after_cap():
    e = Evolver(max_proposals_per_hour=3, max_diff_lines=200)
    e._record_proposals(3)
    with pytest.raises(EvolverGuardError, match="rate limit exceeded"):
        e._check_rate_limit()


def test_rate_limit_below_cap_passes():
    e = Evolver(max_proposals_per_hour=3, max_diff_lines=200)
    e._record_proposals(2)
    e._check_rate_limit()


def test_old_proposals_purged_after_window():
    e = Evolver(max_proposals_per_hour=2, max_diff_lines=200)
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    e._proposal_times.append(stale)
    e._proposal_times.append(stale)
    e._check_rate_limit()


def test_partial_window_keeps_recent_only():
    e = Evolver(max_proposals_per_hour=2, max_diff_lines=200)
    stale = datetime.now(timezone.utc) - timedelta(hours=2)
    e._proposal_times.append(stale)
    e._record_proposals(2)
    with pytest.raises(EvolverGuardError, match="rate limit exceeded"):
        e._check_rate_limit()


def test_propose_calls_guard_before_body(tmp_path):
    e = Evolver(max_proposals_per_hour=1, max_diff_lines=200)
    e._record_proposals(1)
    failure = FailureReport(session_id="s", summary="x", proposed_class="clean")
    with pytest.raises(EvolverGuardError, match="rate limit"):
        e.propose(tmp_path / "harness", failure=failure)


def test_propose_clean_class_returns_empty_list(tmp_path):
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    failure = FailureReport(session_id="s", summary="no failures", proposed_class="clean")
    result = e.propose(tmp_path / "harness", failure=failure)
    assert result == []


@pytest.mark.parametrize(
    "proposed_class",
    [
        "wrong-tool",
        "bad-prompt",
        "state-leak",
        "tool-error",
        "injection-attempt",
    ],
)
def test_propose_class_returns_proposed_edit(tmp_path, proposed_class: str):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    prompt_file = harness_dir / "system_prompt.txt"
    prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    failure = FailureReport(session_id="s", summary="test failure", proposed_class=proposed_class)
    result = e.propose(harness_dir, failure=failure)
    assert len(result) == 1
    edit = result[0]
    assert isinstance(edit, ProposedEdit)
    assert edit.target_file == "harness/system_prompt.txt"
    assert edit.rationale is not None
    assert "--- a/" in edit.unified_diff
    assert "+++ b/" in edit.unified_diff


def test_propose_unknown_class_returns_empty_list(tmp_path):
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    failure = FailureReport(session_id="s", summary="x", proposed_class="nonexistent-class")
    result = e.propose(tmp_path / "harness", failure=failure)
    assert result == []


def test_propose_edit_passes_validate_edit(tmp_path):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    prompt_file = harness_dir / "system_prompt.txt"
    prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    failure = FailureReport(session_id="s", summary="x", proposed_class="wrong-tool")
    result = e.propose(harness_dir, failure=failure)
    assert len(result) == 1
    e._validate_edit(result[0])


def test_propose_records_proposal(tmp_path):
    logger = TraceLogger(tmp_path / "trace.db")
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    prompt_file = harness_dir / "system_prompt.txt"
    prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
    with logger.session("test-session") as session_id:
        e = Evolver(
            max_proposals_per_hour=10,
            max_diff_lines=200,
            trace_logger=logger,
            session_id=session_id,
        )
        failure = FailureReport(session_id="s", summary="x", proposed_class="wrong-tool")
        result = e.propose(harness_dir, failure=failure)
        assert len(result) == 1
    events = list(logger.iter_events(session_id, kind=PROPOSED_EDIT_KIND))
    assert len(events) == 1


class TestMutationClassDerivation:
    def test_system_prompt_file_derives_system_prompt_mutation_class(self):
        from foundry_x.evolution.evolver import _derive_mutation_class

        result = _derive_mutation_class("harness/system_prompt.txt")
        assert result == "system-prompt"

    def test_hooks_file_derives_hook_mutation_class(self):
        from foundry_x.evolution.evolver import _derive_mutation_class

        result = _derive_mutation_class("harness/hooks/my_hook.py")
        assert result == "hook"

    def test_skills_file_derives_skill_mutation_class(self):
        from foundry_x.evolution.evolver import _derive_mutation_class

        result = _derive_mutation_class("harness/skills/my_skill.py")
        assert result == "skill"

    def test_propose_wires_mutation_class(self, tmp_path):
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        failure = FailureReport(session_id="s", summary="x", proposed_class="wrong-tool")
        result = e.propose(harness_dir, failure=failure)
        assert len(result) == 1
        assert result[0].mutation_class == "system-prompt"


class TestProposedEditMutationFields:
    def test_default_mutation_fields(self):
        diff = _make_diff("be precise")
        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="tighten tool guidance",
            unified_diff=diff,
        )
        assert edit.mutation_class == "system-prompt"
        assert edit.risk_level == "low"
        assert edit.is_corrective is False

    def test_explicit_mutation_fields(self):
        diff = _make_diff("be precise")
        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="tighten tool guidance",
            unified_diff=diff,
            mutation_class="hook",
            risk_level="medium",
            is_corrective=True,
        )
        assert edit.mutation_class == "hook"
        assert edit.risk_level == "medium"
        assert edit.is_corrective is True


class TestHighRiskValidator:
    def test_high_risk_with_short_rationale_rejected(self):
        diff = _make_diff("be precise")
        with pytest.raises(
            Exception, match="high-risk edits require a rationale of at least 20 characters"
        ):
            ProposedEdit(
                target_file="harness/system_prompt.txt",
                rationale="short",
                unified_diff=diff,
                risk_level="high",
            )

    def test_high_risk_with_long_rationale_accepted(self):
        diff = _make_diff("be precise")
        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="this is a very thorough rationale explaining the change",
            unified_diff=diff,
            risk_level="high",
        )
        assert edit.risk_level == "high"

    def test_medium_risk_with_short_rationale_accepted(self):
        diff = _make_diff("be precise")
        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="short",
            unified_diff=diff,
            risk_level="medium",
        )
        assert edit.risk_level == "medium"

    def test_low_risk_with_short_rationale_accepted(self):
        diff = _make_diff("be precise")
        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="short",
            unified_diff=diff,
            risk_level="low",
        )
        assert edit.risk_level == "low"


def test_evolver_uses_meta_agent_term_in_comment():
    """Verify evolver.py comment uses 'meta-agent' (canonical term from CONTEXT.md §Concepts).

    Issue #358: CONTEXT.md §Concepts defines 'meta-agent' as the canonical term
    for the Evolver's role. The comment in evolver.py:22 should use this term
    consistently to avoid breaking contributor onboarding (Observe step of the
    workflow).
    """
    import re
    from pathlib import Path

    evolver_path = (
        Path(__file__).resolve().parents[1] / "src" / "foundry_x" / "evolution" / "evolver.py"
    )
    source = evolver_path.read_text(encoding="utf-8")
    assert re.search(r"meta-agent", source), (
        "evolver.py must contain 'meta-agent' in a comment to align with "
        "CONTEXT.md §Concepts terminology"
    )
