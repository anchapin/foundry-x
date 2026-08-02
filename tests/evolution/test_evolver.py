"""Unit tests for Evolver few-shot learning from approved_edit events (issue #957).

Tests verify that:
- _get_past_successful_edits returns at most 5 edits
- _get_past_successful_edits deduplicates by target_file
- _get_past_successful_edits sorts by target_file
- generate_edits incorporates few-shot examples into the LLM prompt
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

from foundry_x.evolution.digester import INFRA_FAILURE_CLASS, FailureReport
from foundry_x.evolution.evolver import (
    Evolver,
    ProposedEdit,
)
from foundry_x.trace.logger import TraceLogger


def _make_diff(target_file: str = "harness/system_prompt.txt", *lines: str) -> str:
    header = f"--- a/{target_file}\n+++ b/{target_file}\n"
    hunk = "@@ -0,0 +1 @@\n"
    return header + hunk + "".join(f"+{line}\n" for line in lines)


def _edit(target_file: str, diff: str, rationale: str = "tighten tool guidance") -> ProposedEdit:
    return ProposedEdit(
        target_file=target_file,
        rationale=rationale,
        unified_diff=diff,
    )


class _MockModelAdapter:
    """Mock ModelAdapter that captures calls for inspection."""

    def __init__(self, response_content: str) -> None:
        self.response_content = response_content
        self.call_count = 0
        self.calls: list[list[Any]] = []

    async def complete(self, messages: list[Any], **kwargs: Any) -> MagicMock:
        self.call_count += 1
        self.calls.append(messages)
        mock_response = MagicMock()
        mock_response.message = MagicMock(content=self.response_content)
        return mock_response


def test_get_past_successful_edits_caps_at_five(tmp_path: Path) -> None:
    """_get_past_successful_edits returns at most 5 edits even when more exist."""
    logger = TraceLogger(tmp_path / "trace.db")
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        for i in range(10):
            edit = _edit(
                f"harness/hooks/hook_{i}.py",
                f"--- a/harness/hooks/hook_{i}.py\n+++ b/harness/hooks/hook_{i}.py\n@@ -0,0 +1 @@\n+def h{i}():\n",
                f"hook {i}",
            )
            evolver._record_approved_edit(edit, failure_class="wrong-tool")

    past_edits = evolver._get_past_successful_edits("wrong-tool")
    assert len(past_edits) == 5


def test_get_past_successful_edits_deduplicates_by_target_file(tmp_path: Path) -> None:
    """_get_past_successful_edits returns only one edit per target_file."""
    logger = TraceLogger(tmp_path / "trace.db")
    diff1 = _make_diff("first change")
    diff2 = _make_diff("second change")
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_approved_edit(
            _edit("harness/system_prompt.txt", diff1), failure_class="wrong-tool"
        )
        evolver._record_approved_edit(
            _edit("harness/system_prompt.txt", diff2), failure_class="wrong-tool"
        )
        evolver._record_approved_edit(
            _edit("harness/hooks/check.py", diff1), failure_class="wrong-tool"
        )

    past_edits = evolver._get_past_successful_edits("wrong-tool")
    assert len(past_edits) == 2
    target_files = {e.target_file for e in past_edits}
    assert target_files == {"harness/system_prompt.txt", "harness/hooks/check.py"}


def test_get_past_successful_edits_sorts_by_target_file(tmp_path: Path) -> None:
    """_get_past_successful_edits returns edits sorted alphabetically by target_file."""
    logger = TraceLogger(tmp_path / "trace.db")
    with logger.session("harness-v1") as session_id:
        evolver = Evolver(trace_logger=logger, session_id=session_id)
        evolver._record_approved_edit(
            _edit("harness/hooks/zzz_hook.py", _make_diff("harness/hooks/zzz_hook.py", "z")),
            failure_class="wrong-tool",
        )
        evolver._record_approved_edit(
            _edit("harness/hooks/aaa_hook.py", _make_diff("harness/hooks/aaa_hook.py", "a")),
            failure_class="wrong-tool",
        )
        evolver._record_approved_edit(
            _edit("harness/hooks/mmm_hook.py", _make_diff("harness/hooks/mmm_hook.py", "m")),
            failure_class="wrong-tool",
        )

    past_edits = evolver._get_past_successful_edits("wrong-tool")
    assert [e.target_file for e in past_edits] == [
        "harness/hooks/aaa_hook.py",
        "harness/hooks/mmm_hook.py",
        "harness/hooks/zzz_hook.py",
    ]


def test_generate_edits_injects_few_shot_examples_into_prompt(tmp_path: Path) -> None:
    """generate_edits calls _get_past_successful_edits and includes examples in the LLM prompt."""
    logger = TraceLogger(tmp_path / "trace.db")
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

    past_diff = _make_diff("previously successful change")
    past_edit = _edit("harness/system_prompt.txt", past_diff, "used before")

    with logger.session("harness-v1") as session_id:
        evolver = Evolver(
            trace_logger=logger,
            session_id=session_id,
            max_proposals_per_hour=10,
            max_diff_lines=200,
        )
        evolver._record_approved_edit(past_edit, failure_class="wrong-tool")

        adapter = _MockModelAdapter(
            '[{"target_file": "harness/system_prompt.txt", '
            '"rationale": "test", "unified_diff": "--- a/harness/system_prompt.txt\\n+++ b/harness/system_prompt.txt\\n@@ -1 +1 @@\\noriginal\\nupdated\\n"}]'
        )

        failure = FailureReport(
            session_id="s",
            summary="tool used incorrectly",
            proposed_class="wrong-tool",
            failed_steps=[{"step": "invoke", "tool": "bash"}],
        )

        import asyncio

        result = asyncio.run(evolver.generate_edits(adapter, harness_dir, failure))

    assert len(result) == 1
    assert adapter.call_count == 1
    call_messages = adapter.calls[0]
    prompt_content = next(
        getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else None)
        for m in call_messages
        if (getattr(m, "role", None) == "user") or (isinstance(m, dict) and m.get("role") == "user")
    )
    assert "PREVIOUSLY SUCCESSFUL EDITS" in prompt_content
    assert "previously successful change" in prompt_content
    assert "wrong-tool" in prompt_content


# ---------------------------------------------------------------------------
# Issue #1033: batch propose for processing multiple failures at once
# ---------------------------------------------------------------------------


class TestProposeBatch:
    """Tests for the batch propose functionality (issue #1033)."""

    def test_propose_batch_empty_reports_returns_empty(self, tmp_path: Path) -> None:
        """An empty batch returns no edits."""
        from foundry_x.evolution.digester import BatchFailureReport

        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        batch = BatchFailureReport(session_id="s", failure_reports=[], total_failures=0)
        edits = evolver.propose_batch(harness_dir, batch)
        assert edits == []

    def test_propose_batch_clean_report_returns_empty(self, tmp_path: Path) -> None:
        """A batch with only a clean report returns no edits."""
        from foundry_x.evolution.digester import BatchFailureReport, FailureReport

        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[
                FailureReport(session_id="s", summary="No failures", proposed_class="clean")
            ],
            total_failures=0,
        )
        edits = evolver.propose_batch(harness_dir, batch)
        assert edits == []

    def test_propose_batch_single_failure_returns_edit(self, tmp_path: Path) -> None:
        """A batch with one failure returns one edit."""
        from foundry_x.evolution.digester import BatchFailureReport, FailureReport

        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[
                FailureReport(
                    session_id="s",
                    summary="no such tool: frobnicate",
                    proposed_class="wrong-tool",
                    failed_steps=[{"kind": "tool_error"}],
                )
            ],
            total_failures=1,
        )
        edits = evolver.propose_batch(harness_dir, batch)
        assert len(edits) == 1
        assert edits[0].target_file == "harness/system_prompt.txt"

    def test_propose_batch_multiple_failures_same_class_deduplicates(self, tmp_path: Path) -> None:
        """Multiple failures of the same class produce only one edit (dedup by target)."""
        from foundry_x.evolution.digester import BatchFailureReport, FailureReport

        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[
                FailureReport(
                    session_id="s",
                    summary="first wrong-tool",
                    proposed_class="wrong-tool",
                    failed_steps=[{"kind": "tool_error"}],
                ),
                FailureReport(
                    session_id="s",
                    summary="second wrong-tool",
                    proposed_class="wrong-tool",
                    failed_steps=[{"kind": "tool_error"}],
                ),
            ],
            total_failures=2,
        )
        edits = evolver.propose_batch(harness_dir, batch)
        assert len(edits) == 1

    def test_propose_batch_multiple_failures_different_classes(self, tmp_path: Path) -> None:
        """Multiple failures of different classes produce separate edits."""
        from foundry_x.evolution.digester import BatchFailureReport, FailureReport

        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[
                FailureReport(
                    session_id="s",
                    summary="no such tool: frobnicate",
                    proposed_class="wrong-tool",
                    failed_steps=[{"kind": "tool_error"}],
                ),
                FailureReport(
                    session_id="s",
                    summary="some traceback happened",
                    proposed_class="tool-error",
                    failed_steps=[{"kind": "tool_error"}],
                ),
            ],
            total_failures=2,
        )
        edits = evolver.propose_batch(harness_dir, batch)
        assert len(edits) == 1

    def test_propose_batch_all_clean_reports_returns_empty(self, tmp_path: Path) -> None:
        """When all reports in the batch are clean, no edits are proposed."""
        from foundry_x.evolution.digester import BatchFailureReport, FailureReport

        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[
                FailureReport(session_id="s", summary="clean", proposed_class="clean"),
                FailureReport(session_id="s", summary="also clean", proposed_class="clean"),
            ],
            total_failures=0,
        )
        edits = evolver.propose_batch(harness_dir, batch)
        assert edits == []


# ---------------------------------------------------------------------------
# Issue #1462: infra-failure must not trigger a system_prompt.txt edit.
# The Evolver skips harness-edit remediation for infra/model-server
# failures because no prompt edit can remediate them.
# ---------------------------------------------------------------------------


class TestInfraFailureSkip:
    """The Evolver must not propose edits for ``infra-failure`` class."""

    def test_propose_returns_empty_for_infra_failure(self, tmp_path: Path) -> None:
        """propose() returns [] for an infra-failure report."""
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        failure = FailureReport(
            session_id="s",
            summary="server unavailable",
            proposed_class=INFRA_FAILURE_CLASS,
            failed_steps=[{"kind": "server_unavailable"}],
        )
        edits = evolver.propose(harness_dir, failure)
        assert edits == []

    def test_propose_does_not_target_system_prompt_for_infra_failure(self, tmp_path: Path) -> None:
        """No ProposedEdit targets system_prompt.txt for infra-failure."""
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        failure = FailureReport(
            session_id="s",
            summary="hook registry error",
            proposed_class=INFRA_FAILURE_CLASS,
            failed_steps=[{"kind": "hook_registry_error"}],
        )
        edits = evolver.propose(harness_dir, failure)
        for edit in edits:
            assert "system_prompt.txt" not in edit.target_file

    def test_propose_batch_skips_infra_failure(self, tmp_path: Path) -> None:
        """propose_batch() skips infra-failure reports."""
        from foundry_x.evolution.digester import BatchFailureReport

        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")

        evolver = Evolver(max_proposals_per_hour=10, max_diff_lines=200)
        batch = BatchFailureReport(
            session_id="s",
            failure_reports=[
                FailureReport(
                    session_id="s",
                    summary="server down",
                    proposed_class=INFRA_FAILURE_CLASS,
                    failed_steps=[{"kind": "server_unavailable"}],
                ),
                FailureReport(
                    session_id="s",
                    summary="no such tool: frobnicate",
                    proposed_class="wrong-tool",
                    failed_steps=[{"kind": "tool_error"}],
                ),
            ],
            total_failures=2,
        )
        edits = evolver.propose_batch(harness_dir, batch)
        # Only the wrong-tool edit should be proposed; infra-failure is skipped.
        assert len(edits) == 1
        assert edits[0].target_file == "harness/system_prompt.txt"
