from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    Evolver,
    EvolverGenerationError,
    EvolverLLMError,
    _build_generation_prompt,
    _parse_edits_from_response,
)


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
