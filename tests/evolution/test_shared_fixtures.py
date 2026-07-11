from __future__ import annotations

from foundry_x.evolution.digester import Digester, FailureReport
from foundry_x.evolution.evolver import ProposedEdit
from foundry_x.trace.logger import TraceEvent


def test_failing_trace_is_classified_as_failure(failing_trace: list[TraceEvent]):
    report = Digester().digest("sess-fixture", failing_trace)
    assert report.proposed_class != "clean"
    assert report.failed_steps


def test_failure_report_fixture_is_valid_instance(failure_report: FailureReport):
    assert isinstance(failure_report, FailureReport)
    assert failure_report.session_id == "sess-fixture"
    assert failure_report.proposed_class == "bad-prompt"
    assert len(failure_report.failed_steps) == 2
    assert len(failure_report.suspected_causes) == 2


def test_proposed_edit_fixture_is_valid_instance(proposed_edit: ProposedEdit):
    assert isinstance(proposed_edit, ProposedEdit)
    assert proposed_edit.target_file == "harness/system_prompt.txt"
    assert proposed_edit.rationale
    assert proposed_edit.unified_diff
