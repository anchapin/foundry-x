from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    GENERATION_ATTEMPT_KIND,
    Evolver,
    EvolverGenerationError,
    EvolverLLMError,
    _build_generation_prompt,
    _parse_edits_from_response,
)
from foundry_x.trace.logger import TraceLogger


class TestBuildGenerationPrompt:
    def test_includes_failure_class_and_summary(
        self, failure_report: FailureReport, tmp_path: Path
    ) -> None:
        messages = _build_generation_prompt(failure_report, tmp_path)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert "harness" in messages[0]["content"]
        assert messages[1]["role"] == "user"
        assert "bad-prompt" in messages[1]["content"]
        assert "wrong tool" in messages[1]["content"]

    def test_includes_suspected_causes(self, failure_report: FailureReport, tmp_path: Path) -> None:
        messages = _build_generation_prompt(failure_report, tmp_path)
        user_content = messages[1]["content"]
        assert "Suspected causes" in user_content
        assert "System prompt does not list" in user_content

    def test_includes_failed_steps(self, failure_report: FailureReport, tmp_path: Path) -> None:
        messages = _build_generation_prompt(failure_report, tmp_path)
        user_content = messages[1]["content"]
        assert "Failed step 1" in user_content
        assert "step" in user_content


class TestParseEditsFromResponse:
    def test_parses_valid_json_array(self, tmp_path: Path) -> None:
        raw = json.dumps(
            [
                {
                    "target_file": "harness/system_prompt.txt",
                    "rationale": "test rationale",
                    "unified_diff": "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
                }
            ]
        )
        edits = _parse_edits_from_response(raw)
        assert len(edits) == 1
        assert edits[0].target_file == "harness/system_prompt.txt"
        assert edits[0].rationale == "test rationale"

    def test_parses_json_in_code_fence(self, tmp_path: Path) -> None:
        raw = '```json\n[{"target_file": "harness/manifest.json", "rationale": "fix", "unified_diff": "--- a/harness/manifest.json\\n+++ b/harness/manifest.json\\n@@ -1 +1 @@\\n{}\\n+{}\\n"}]\n```'
        edits = _parse_edits_from_response(raw)
        assert len(edits) == 1
        assert edits[0].target_file == "harness/manifest.json"

    def test_rejects_non_array(self) -> None:
        raw = '{"target_file": "harness/system_prompt.txt"}'
        with pytest.raises(EvolverGenerationError, match="must be a JSON array"):
            _parse_edits_from_response(raw)

    def test_rejects_invalid_json(self) -> None:
        raw = "not json at all"
        with pytest.raises(EvolverGenerationError, match="not valid JSON"):
            _parse_edits_from_response(raw)

    def test_skips_invalid_items_keeps_valid(self) -> None:
        raw = json.dumps(
            [
                {
                    "target_file": "harness/system_prompt.txt",
                    "rationale": "valid",
                    "unified_diff": "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\na\nb\n",
                },
                {"not": "a valid edit"},
                {
                    "target_file": "harness/manifest.json",
                    "rationale": "also valid",
                    "unified_diff": "--- a/harness/manifest.json\n+++ b/harness/manifest.json\n@@ -1 +1 @@\n{}\n{}\n",
                },
            ]
        )
        edits = _parse_edits_from_response(raw)
        assert len(edits) == 2

    def test_raises_when_all_items_invalid(self) -> None:
        raw = json.dumps([{"not": "a valid ProposedEdit"}])
        with pytest.raises(EvolverGenerationError, match="no valid ProposedEdit"):
            _parse_edits_from_response(raw)

    def test_raises_on_empty_array(self) -> None:
        """A bare empty JSON array ``[]`` is a generation failure, not a silent no-op.

        Regression test for issue #973: previously ``[]`` returned ``[]``
        silently, bypassing the retry/template-fallback path in
        ``generate_edits`` and emitting no ``generation_attempt`` event.
        """
        with pytest.raises(EvolverGenerationError, match="zero ProposedEdit objects"):
            _parse_edits_from_response("[]")

    def test_confines_target_file_to_harness(self) -> None:
        raw = json.dumps(
            [
                {
                    "target_file": "/etc/passwd",
                    "rationale": "malicious",
                    "unified_diff": "--- a/x\n+++ b/x\n",
                }
            ]
        )
        with pytest.raises(EvolverGenerationError):
            _parse_edits_from_response(raw)


class TestGenerateEdits:
    @pytest.fixture
    def mock_adapter(self) -> MagicMock:
        adapter = MagicMock()
        adapter.complete = AsyncMock()
        return adapter

    @pytest.mark.asyncio
    async def test_returns_parsed_edits(
        self, mock_adapter: MagicMock, failure_report: FailureReport, tmp_path: Path
    ) -> None:
        mock_adapter.complete.return_value = MagicMock(
            message=MagicMock(
                content='[{"target_file": "harness/system_prompt.txt", "rationale": "test", "unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\na\\nb\\n"}]'
            )
        )
        evolver = Evolver()
        edits = await evolver.generate_edits(mock_adapter, tmp_path, failure_report)
        assert len(edits) == 1
        assert edits[0].target_file == "harness/system_prompt.txt"
        mock_adapter.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_on_validation_failure(
        self, mock_adapter: MagicMock, failure_report: FailureReport, tmp_path: Path
    ) -> None:
        call_count = 0

        async def mock_complete(messages):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return MagicMock(message=MagicMock(content="not json"))
            return MagicMock(
                message=MagicMock(
                    content=json.dumps(
                        [
                            {
                                "target_file": "harness/system_prompt.txt",
                                "rationale": "test",
                                "unified_diff": "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\na\nb\n",
                            }
                        ]
                    )
                )
            )

        mock_adapter.complete.side_effect = mock_complete
        evolver = Evolver()
        edits = await evolver.generate_edits(mock_adapter, tmp_path, failure_report, max_retries=2)
        assert len(edits) == 1
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_raises_after_max_retries(
        self, mock_adapter: MagicMock, failure_report: FailureReport, tmp_path: Path
    ) -> None:
        mock_adapter.complete.side_effect = Exception("model error")
        evolver = Evolver()
        with pytest.raises(EvolverLLMError, match="generation failed after"):
            await evolver.generate_edits(mock_adapter, tmp_path, failure_report, max_retries=2)

    @pytest.mark.asyncio
    async def test_records_proposals_on_success(
        self, mock_adapter: MagicMock, failure_report: FailureReport, tmp_path: Path
    ) -> None:
        evolver = Evolver(trace_logger=MagicMock(), session_id="test-session")
        mock_adapter.complete.return_value = MagicMock(
            message=MagicMock(
                content=json.dumps(
                    [
                        {
                            "target_file": "harness/system_prompt.txt",
                            "rationale": "test",
                            "unified_diff": "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\na\nb\n",
                        }
                    ]
                )
            )
        )
        edits = await evolver.generate_edits(mock_adapter, tmp_path, failure_report)
        assert len(edits) == 1
        assert len(evolver._proposal_times) == 1

    @pytest.mark.asyncio
    async def test_empty_array_raises_and_records_generation_attempt(
        self, mock_adapter: MagicMock, failure_report: FailureReport, tmp_path: Path
    ) -> None:
        """A bare ``[]`` LLM response exhausts retries and raises ``EvolverLLMError``.

        Each attempt records a ``generation_attempt`` trace event so the
        failure class is observable (issue #973 acceptance criterion 2).
        """
        mock_adapter.complete.return_value = MagicMock(message=MagicMock(content="[]"))
        trace_logger = MagicMock()
        evolver = Evolver(trace_logger=trace_logger, session_id="sess-empty-array")
        with pytest.raises(EvolverLLMError, match="no valid ProposedEdit"):
            await evolver.generate_edits(mock_adapter, tmp_path, failure_report, max_retries=2)
        attempt_calls = [
            call
            for call in trace_logger.record.call_args_list
            if call.args[1] == GENERATION_ATTEMPT_KIND
        ]
        assert len(attempt_calls) == 2, (
            f"Expected one generation_attempt event per retry (2), got {len(attempt_calls)}"
        )


class TestParseLlmResponseParseFailure:
    """Tests for issue #976: JSON-parse failure must be observable, not silent.

    ``_parse_llm_response`` previously returned ``[]`` with no trace event
    when both the regex extraction and the bare ``json.loads`` failed,
    causing silent edit loss and ``improvement-rate`` under-reporting.
    """

    def test_unparseable_content_emits_generation_attempt(self, tmp_path: Path) -> None:
        """Garbage content emits a ``generation_attempt`` trace event."""
        logger = TraceLogger(tmp_path / "trace.db")
        with logger.session("sess-parse-fail") as session_id:
            evolver = Evolver(
                trace_logger=logger,
                session_id=session_id,
            )
            edits = evolver._parse_llm_response("this is not json at all")

        assert edits == []
        events = list(logger.iter_events(session_id, kind=GENERATION_ATTEMPT_KIND))
        assert len(events) == 1, "Expected exactly one generation_attempt event"
        error = events[0].payload["error"]
        assert "parse_failure" in error
        assert "content_length=" in error

    def test_regex_match_but_both_json_loads_fail_emits_event(self, tmp_path: Path) -> None:
        """Content matching ``_EDIT_JSON_RE`` but with broken inner JSON emits an event.

        Exercises the double-failure path: the regex wrapper matches but
        ``json.loads`` on both the matched group and the full content fail.
        """
        logger = TraceLogger(tmp_path / "trace.db")
        with logger.session("sess-double-fail") as session_id:
            evolver = Evolver(
                trace_logger=logger,
                session_id=session_id,
            )
            # The regex can match a ``{...proposed_edits...}`` shell, but
            # the inner JSON is intentionally malformed (trailing comma).
            content = '{"proposed_edits": [BROKEN,]}'
            edits = evolver._parse_llm_response(content)

        assert edits == []
        events = list(logger.iter_events(session_id, kind=GENERATION_ATTEMPT_KIND))
        assert len(events) == 1
        assert "parse_failure" in events[0].payload["error"]

    def test_event_includes_content_snippet(self, tmp_path: Path) -> None:
        """The trace event carries a snippet of the unparseable content."""
        logger = TraceLogger(tmp_path / "trace.db")
        content = "<<<garbage>>>"
        with logger.session("sess-snippet") as session_id:
            evolver = Evolver(
                trace_logger=logger,
                session_id=session_id,
            )
            evolver._parse_llm_response(content)

        events = list(logger.iter_events(session_id, kind=GENERATION_ATTEMPT_KIND))
        assert len(events) == 1
        # The model_response_excerpt field carries the raw content.
        assert events[0].payload["model_response_excerpt"] == content

    def test_valid_json_no_event_emitted(self, tmp_path: Path) -> None:
        """Valid JSON with no parseable edits does NOT emit a parse_failure event.

        A well-formed JSON response that simply has no valid edits is not
        a parse failure — only unparseable content triggers the event.
        """
        logger = TraceLogger(tmp_path / "trace.db")
        with logger.session("sess-valid") as session_id:
            evolver = Evolver(
                trace_logger=logger,
                session_id=session_id,
            )
            edits = evolver._parse_llm_response('{"proposed_edits": []}')

        assert edits == []
        events = list(logger.iter_events(session_id, kind=GENERATION_ATTEMPT_KIND))
        assert len(events) == 0, "Valid JSON should not emit a parse_failure event"

    def test_no_trace_logger_no_crash(self) -> None:
        """Without a TraceLogger, parse failure still returns [] gracefully."""
        evolver = Evolver()
        edits = evolver._parse_llm_response("not json")
        assert edits == []
