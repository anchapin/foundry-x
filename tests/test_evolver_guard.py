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


def test_record_proposal_with_failure_class(tmp_path):
    logger = TraceLogger(tmp_path / "trace.db")
    edit = _edit(_make_diff("be precise"))
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_proposals(edit=edit, failure_class="wrong-tool")

    events = list(logger.iter_events(session_id, kind=PROPOSED_EDIT_KIND))
    assert len(events) == 1
    assert events[0].payload["failure_class"] == "wrong-tool"
    assert events[0].payload["review_state"] == "PROPOSED"


def test_record_approved_edit_emits_event(tmp_path):
    logger = TraceLogger(tmp_path / "trace.db")
    edit = _edit(_make_diff("be precise"))
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_approved_edit(edit, failure_class="wrong-tool")

    from foundry_x.evolution.evolver import APPROVED_EDIT_KIND

    events = list(logger.iter_events(session_id, kind=APPROVED_EDIT_KIND))
    assert len(events) == 1
    assert events[0].payload["failure_class"] == "wrong-tool"
    assert events[0].payload["review_state"] == "PROPOSED"


def test_get_past_successful_edits_returns_approved(tmp_path):
    logger = TraceLogger(tmp_path / "trace.db")
    edit1 = _edit(_make_diff("be precise"))
    edit2 = ProposedEdit(
        target_file="harness/hooks/check_tool.py",
        rationale="add tool validation",
        unified_diff="--- a/harness/hooks/check_tool.py\n+++ b/harness/hooks/check_tool.py\n@@ -0,0 +1 @@\n+def check():\n",
    )
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_approved_edit(edit1, failure_class="wrong-tool")
        evolver._record_approved_edit(edit2, failure_class="wrong-tool")

    past_edits = evolver._get_past_successful_edits("wrong-tool")
    assert len(past_edits) == 2
    assert all(e.review_state.value == "PROPOSED" for e in past_edits)


def test_get_past_successful_edits_filters_by_class(tmp_path):
    logger = TraceLogger(tmp_path / "trace.db")
    edit1 = _edit(_make_diff("be precise"))
    edit2 = _edit(_make_diff("check twice"))
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_approved_edit(edit1, failure_class="wrong-tool")
        evolver._record_approved_edit(edit2, failure_class="bad-prompt")

    past_edits = evolver._get_past_successful_edits("wrong-tool")
    assert len(past_edits) == 1
    assert past_edits[0].target_file == "harness/system_prompt.txt"


def test_build_llm_prompt_with_few_shot_examples():
    failure = FailureReport(
        session_id="test",
        summary="tool used incorrectly",
        proposed_class="wrong-tool",
        failed_steps=[{"step": "invoke", "tool": "bash"}],
    )
    edit = _edit(_make_diff("check tool list"))
    e = Evolver()
    prompt = e._build_llm_prompt(failure, few_shot_edits=[edit])
    assert "PREVIOUSLY SUCCESSFUL EDITS" in prompt
    assert "wrong-tool" in prompt
    assert "check tool list" in prompt


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
    result = e.propose(tmp_path / "harness", failure=failure)
    assert result == []


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
        "context-overflow",
    ],
)
def test_propose_class_returns_proposed_edit(tmp_path, proposed_class: str):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    if proposed_class == "context-overflow":
        manifest_file = harness_dir / "manifest.json"
        manifest_file.write_text("{}\n", encoding="utf-8")
    else:
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    failure = FailureReport(session_id="s", summary="test failure", proposed_class=proposed_class)
    result = e.propose(harness_dir, failure=failure)
    assert len(result) == 1
    edit = result[0]
    assert isinstance(edit, ProposedEdit)
    if proposed_class == "context-overflow":
        assert edit.target_file == "harness/manifest.json"
    else:
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


@pytest.mark.asyncio
async def test_propose_async_returns_same_result_as_sync(tmp_path):
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    prompt_file = harness_dir / "system_prompt.txt"
    prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    failure = FailureReport(session_id="s", summary="test failure", proposed_class="wrong-tool")
    sync_result = e.propose(harness_dir, failure=failure)
    async_result = await e.propose_async(harness_dir, failure=failure)
    assert sync_result == async_result
    assert len(async_result) == 1
    edit = async_result[0]
    assert isinstance(edit, ProposedEdit)
    assert edit.target_file == "harness/system_prompt.txt"


@pytest.mark.asyncio
async def test_propose_async_rate_limit_enforced(tmp_path):
    e = Evolver(max_proposals_per_hour=1, max_diff_lines=200)
    e._record_proposals(1)
    failure = FailureReport(session_id="s", summary="x", proposed_class="clean")
    result = await e.propose_async(tmp_path / "harness", failure=failure)
    assert result == []


@pytest.mark.asyncio
async def test_propose_async_clean_class_returns_empty(tmp_path):
    e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
    failure = FailureReport(session_id="s", summary="no failures", proposed_class="clean")
    result = await e.propose_async(tmp_path / "harness", failure=failure)
    assert result == []
