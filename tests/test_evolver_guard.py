from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

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
