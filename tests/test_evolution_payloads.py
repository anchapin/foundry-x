from __future__ import annotations

import pytest
from pydantic import ValidationError

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import ProposedEdit


def test_failure_report_defaults_preserved():
    report = FailureReport(session_id="sess-1", summary="boom")
    assert report.failed_steps == []
    assert report.suspected_causes == []
    assert report.proposed_class == "unknown"


def test_failure_report_missing_required_field_raises():
    with pytest.raises(ValidationError):
        FailureReport(summary="missing session id")  # type: ignore[call-arg]


def test_failure_report_wrong_type_raises():
    with pytest.raises(ValidationError):
        FailureReport(session_id=123, summary="non-string id")  # type: ignore[arg-type]


def test_failure_report_failed_steps_accepts_typed_dicts():
    report = FailureReport(
        session_id="sess-1",
        summary="boom",
        failed_steps=[{"step": 1, "kind": "tool_call", "detail": "wrong tool"}],
    )
    assert report.failed_steps[0]["kind"] == "tool_call"


def test_proposed_edit_requires_non_blank_fields():
    with pytest.raises(ValidationError):
        ProposedEdit(target_file="", rationale="r", unified_diff="diff")


def test_proposed_edit_blank_unified_diff_raises():
    with pytest.raises(ValidationError):
        ProposedEdit(target_file="harness/system_prompt.txt", rationale="r", unified_diff="")


def test_proposed_edit_valid_payload_constructs():
    edit = ProposedEdit(
        target_file="harness/system_prompt.txt",
        rationale="add tool guidance",
        unified_diff="@@ -1 +1 @@\n-old\n+new\n",
    )
    assert edit.target_file == "harness/system_prompt.txt"


def test_critic_verdict_defaults_preserved():
    verdict = CriticVerdict(approved=True)
    assert verdict.passed_checks == []
    assert verdict.failed_checks == []
    assert verdict.notes == ""


def test_critic_verdict_missing_approved_raises():
    with pytest.raises(ValidationError):
        CriticVerdict()  # type: ignore[call-arg]


def test_critic_verdict_non_bool_approved_raises():
    with pytest.raises(ValidationError):
        CriticVerdict(approved=["not", "a", "bool"])  # type: ignore[arg-type]
