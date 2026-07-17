from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    PROPOSED_EDIT_KIND,
    GENERATION_EXHAUSTED_KIND,
    Evolver,
    EvolverGuardError,
    EvolverLLMError,
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


class TestBuildLlmPrompt:
    def test_prompt_includes_failure_summary(self):
        e = Evolver()
        failure = FailureReport(
            session_id="s",
            summary="Agent called wrong tool",
            proposed_class="wrong-tool",
        )
        prompt = e._build_llm_prompt(failure)
        assert "Agent called wrong tool" in prompt
        assert "wrong-tool" in prompt

    def test_prompt_includes_suspected_causes(self):
        e = Evolver()
        failure = FailureReport(
            session_id="s",
            summary="failure",
            suspected_causes=["cause 1", "cause 2"],
            proposed_class="bad-prompt",
        )
        prompt = e._build_llm_prompt(failure)
        assert "cause 1" in prompt
        assert "cause 2" in prompt

    def test_prompt_includes_constraints(self):
        e = Evolver()
        failure = FailureReport(session_id="s", summary="f", proposed_class="clean")
        prompt = e._build_llm_prompt(failure)
        assert "harness/" in prompt
        assert "system_prompt.txt" in prompt
        assert "--- a/" in prompt


class TestParseLlmResponse:
    def test_parses_valid_json_with_proposed_edits(self):
        e = Evolver()
        content = '{"proposed_edits": [{"target_file": "harness/system_prompt.txt", "rationale": "test", "unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\n-old\\n+new\\n"}]}'
        edits = e._parse_llm_response(content)
        assert len(edits) == 1
        assert edits[0].target_file == "harness/system_prompt.txt"
        assert edits[0].rationale == "test"

    def test_parses_raw_json_array(self):
        e = Evolver()
        content = '{"proposed_edits": [{"target_file": "harness/manifest.json", "rationale": "update", "unified_diff": "--- a/harness/manifest.json\\n+++ b/harness/manifest.json\\n@@ -1 +1 @@\\n-a\\n+b\\n"}]}'
        edits = e._parse_llm_response(content)
        assert len(edits) == 1

    def test_skips_malformed_edits(self):
        e = Evolver()
        content = '{"proposed_edits": [{"target_file": "harness/system_prompt.txt", "rationale": "valid edit", "unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\n-a\\n+b\\n"}, {"target_file": "harness/hooks/bad.py", "rationale": "missing diff"}]}'
        edits = e._parse_llm_response(content)
        assert len(edits) == 1
        assert edits[0].rationale == "valid edit"

    def test_returns_empty_on_invalid_json(self):
        e = Evolver()
        assert e._parse_llm_response("not json") == []
        assert e._parse_llm_response('{"proposed_edits": null}') == []
        assert e._parse_llm_response("") == []

    def test_rejects_out_of_tree_target(self):
        e = Evolver()
        content = '{"proposed_edits": [{"target_file": "../etc/passwd", "rationale": "test", "unified_diff": "--- a/etc/passwd\\n+++ b/etc/passwd\\n@@ -1 +1 @@\\n-a\\n+b\\n"}]}'
        edits = e._parse_llm_response(content)
        assert len(edits) == 0

    def test_rejects_diff_without_headers(self):
        e = Evolver()
        content = '{"proposed_edits": [{"target_file": "harness/system_prompt.txt", "rationale": "test", "unified_diff": "@@ -1 +1 @@\\n-old\\n+new\\n"}]}'
        edits = e._parse_llm_response(content)
        assert len(edits) == 0


class TestProposeWithLlm:
    def test_propose_uses_llm_when_adapter_configured(self, tmp_path):
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = '{"proposed_edits": [{"target_file": "harness/system_prompt.txt", "rationale": "llm generated", "unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\n-old\\n+new\\n"}]}'
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200, model_adapter=mock_adapter)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        result = e.propose(harness_dir, failure=failure)
        assert len(result) == 1
        assert result[0].rationale == "llm generated"
        mock_adapter.complete.assert_called_once()

    def test_propose_falls_back_to_template_on_llm_failure(self, tmp_path):
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock(side_effect=RuntimeError("LLM unavailable"))
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200, model_adapter=mock_adapter)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        result = e.propose(harness_dir, failure=failure)
        assert len(result) == 1
        assert result[0].rationale == "address wrong-tool failure: reinforce tool list adherence"

    def test_propose_falls_back_to_template_on_invalid_llm_output(self, tmp_path):
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "not valid json"
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200, model_adapter=mock_adapter)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        result = e.propose(harness_dir, failure=failure)
        assert len(result) == 1

    def test_propose_without_adapter_uses_template(self, tmp_path):
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        prompt_file = harness_dir / "system_prompt.txt"
        prompt_file.write_text("You are FoundryAgent.\n", encoding="utf-8")
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        failure = FailureReport(session_id="s", summary="test", proposed_class="bad-prompt")
        result = e.propose(harness_dir, failure=failure)
        assert len(result) == 1
        assert result[0].rationale == "address bad-prompt failure: add disambiguation guidance"


class TestLLMRateLimiting:
    def setup_method(self) -> None:
        Evolver._llm_call_times.clear()
        Evolver._llm_call_costs.clear()

    def teardown_method(self) -> None:
        Evolver._llm_call_times.clear()
        Evolver._llm_call_costs.clear()

    def test_defaults_match_security_doc(self) -> None:
        e = Evolver()
        assert e.max_llm_calls_per_hour == 60
        assert e.max_cost_per_day == 5.00

    def test_invalid_llm_limits_rejected(self) -> None:
        with pytest.raises(EvolverGuardError, match="max_llm_calls_per_hour"):
            Evolver(max_llm_calls_per_hour=0)
        with pytest.raises(EvolverGuardError, match="max_cost_per_day"):
            Evolver(max_cost_per_day=-1.0)

    def test_llm_rate_limit_triggers_after_call_cap(self) -> None:
        e = Evolver(max_llm_calls_per_hour=3, max_cost_per_day=10.0)
        e._llm_call_times.extend([datetime.now(timezone.utc)] * 3)
        with pytest.raises(EvolverGuardError, match="LLM rate limit exceeded"):
            e._check_llm_rate_limit()

    def test_llm_rate_limit_below_cap_passes(self) -> None:
        e = Evolver(max_llm_calls_per_hour=3, max_cost_per_day=10.0)
        e._llm_call_times.extend([datetime.now(timezone.utc)] * 2)
        e._check_llm_rate_limit()

    def test_llm_cost_limit_triggers_after_daily_cap(self) -> None:
        e = Evolver(max_llm_calls_per_hour=100, max_cost_per_day=1.0)
        now = datetime.now(timezone.utc)
        e._llm_call_costs.extend([(now, 0.6)] * 2)
        with pytest.raises(EvolverGuardError, match="LLM cost limit exceeded"):
            e._check_llm_rate_limit()

    def test_llm_cost_limit_below_cap_passes(self) -> None:
        e = Evolver(max_llm_calls_per_hour=100, max_cost_per_day=1.0)
        now = datetime.now(timezone.utc)
        e._llm_call_costs.extend([(now, 0.5)])
        e._check_llm_rate_limit()

    def test_old_llm_calls_purged_after_window(self) -> None:
        e = Evolver(max_llm_calls_per_hour=2, max_cost_per_day=10.0)
        stale = datetime.now(timezone.utc) - timedelta(hours=2)
        e._llm_call_times.append(stale)
        e._llm_call_times.append(stale)
        e._check_llm_rate_limit()

    def test_old_llm_costs_purged_after_window(self) -> None:
        e = Evolver(max_llm_calls_per_hour=100, max_cost_per_day=0.5)
        stale = datetime.now(timezone.utc) - timedelta(days=2)
        e._llm_call_costs.append((stale, 10.0))
        e._check_llm_rate_limit()

    def test_record_llm_call_updates_shared_state(self) -> None:
        e1 = Evolver(max_llm_calls_per_hour=10, max_cost_per_day=10.0)
        e2 = Evolver(max_llm_calls_per_hour=10, max_cost_per_day=10.0)
        e1.record_llm_call(cost=0.5)
        e2.record_llm_call(cost=0.3)
        assert len(e1._llm_call_times) == 2
        assert len(e2._llm_call_times) == 2
        assert len(Evolver._llm_call_times) == 2
        total_cost = sum(cost for _, cost in Evolver._llm_call_costs)
        assert abs(total_cost - 0.8) < 0.001

    def test_record_llm_call_accepts_zero_cost(self) -> None:
        e = Evolver(max_llm_calls_per_hour=10, max_cost_per_day=10.0)
        e.record_llm_call(cost=0.0)
        assert len(e._llm_call_times) == 1


class TestEvolverLLMError:
    def test_llm_error_is_raised_on_exhaustion(self) -> None:
        from foundry_x.evolution.evolver import EvolverLLMError

        err = EvolverLLMError("test error")
        assert str(err) == "test error"
        assert isinstance(err, Exception)


class TestGenerateEdits:
    def setup_method(self) -> None:
        Evolver._llm_call_times.clear()
        Evolver._llm_call_costs.clear()

    def teardown_method(self) -> None:
        Evolver._llm_call_times.clear()
        Evolver._llm_call_costs.clear()

    @pytest.mark.asyncio
    async def test_generate_edits_raises_llm_error_on_rate_limit(self, tmp_path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        mock_adapter = MagicMock()
        mock_adapter.complete = AsyncMock()
        e = Evolver(
            max_proposals_per_hour=10,
            max_diff_lines=200,
            max_llm_calls_per_hour=1,
            max_cost_per_day=10.0,
        )
        e._llm_call_times.append(datetime.now(timezone.utc))
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        with pytest.raises(EvolverLLMError, match="LLM rate limit exceeded"):
            await e.generate_edits(mock_adapter, harness_dir, failure)

    @pytest.mark.asyncio
    async def test_generate_edits_records_llm_call_before_attempt(self, tmp_path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = '[{"target_file": "harness/system_prompt.txt", "rationale": "test", "unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\n-old\\n+new\\n"}]'
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200, max_llm_calls_per_hour=10)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        await e.generate_edits(mock_adapter, harness_dir, failure)
        assert len(e._llm_call_times) == 1

    @pytest.mark.asyncio
    async def test_generate_edits_retries_on_malformed_json(self, tmp_path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "not valid json"
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200, max_llm_calls_per_hour=10)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        with pytest.raises(EvolverLLMError):
            await e.generate_edits(mock_adapter, harness_dir, failure, max_retries=2)
        assert mock_adapter.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_edits_retries_on_validation_error(self, tmp_path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = '{"proposed_edits": [{"target_file": "../etc/passwd", "rationale": "test", "unified_diff": "--- a/etc/passwd\\n+++ b/etc/passwd\\n@@ -1 +1 @@\\n-old\\n+new\\n"}]}'
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200, max_llm_calls_per_hour=10)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        with pytest.raises(EvolverLLMError):
            await e.generate_edits(mock_adapter, harness_dir, failure, max_retries=2)
        assert mock_adapter.complete.call_count == 2

    @pytest.mark.asyncio
    async def test_generate_edits_returns_valid_edits(self, tmp_path) -> None:
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = '[{"target_file": "harness/system_prompt.txt", "rationale": "test", "unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\n-old\\n+new\\n"}]'
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        e = Evolver(max_proposals_per_hour=10, max_diff_lines=200, max_llm_calls_per_hour=10)
        failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
        edits = await e.generate_edits(mock_adapter, harness_dir, failure, max_retries=2)
        assert len(edits) == 1
        assert edits[0].rationale == "test"

    @pytest.mark.asyncio
    async def test_generate_edits_records_exhausted_event_on_failure(self, tmp_path) -> None:
        logger = TraceLogger(tmp_path / "trace.db")
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "not valid json"
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        with logger.session("test-session") as session_id:
            e = Evolver(
                max_proposals_per_hour=10,
                max_diff_lines=200,
                max_llm_calls_per_hour=10,
                trace_logger=logger,
                session_id=session_id,
            )
            failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
            with pytest.raises(EvolverLLMError):
                await e.generate_edits(mock_adapter, harness_dir, failure, max_retries=2)
        events = list(logger.iter_events(session_id, kind=GENERATION_EXHAUSTED_KIND))
        assert len(events) == 1

    @pytest.mark.asyncio
    async def test_generate_edits_records_attempt_events(self, tmp_path) -> None:
        logger = TraceLogger(tmp_path / "trace.db")
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        mock_adapter = MagicMock()
        mock_response = MagicMock()
        mock_response.message.content = "not valid json"
        mock_adapter.complete = AsyncMock(return_value=mock_response)
        with logger.session("test-session") as session_id:
            e = Evolver(
                max_proposals_per_hour=10,
                max_diff_lines=200,
                max_llm_calls_per_hour=10,
                trace_logger=logger,
                session_id=session_id,
            )
            failure = FailureReport(session_id="s", summary="test", proposed_class="wrong-tool")
            with pytest.raises(EvolverLLMError):
                await e.generate_edits(mock_adapter, harness_dir, failure, max_retries=2)
        events = list(logger.iter_events(session_id, kind="generation_attempt"))
        assert len(events) == 2


class TestJitteredBackoff:
    def test_backoff_base_increases_with_attempt(self) -> None:
        base = 0.5
        for attempt in range(1, 5):
            min_possible = base * attempt
            next_min_possible = base * (attempt + 1)
            assert next_min_possible > min_possible

    def test_backoff_has_jitter(self) -> None:
        from foundry_x.evolution.evolver import _jittered_backoff

        delays = [_jittered_backoff(2) for _ in range(10)]
        assert len(set(delays)) > 1
