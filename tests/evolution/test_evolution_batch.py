"""Tests for run_evolution_batch — specifically the zero-edit failure-class path (issue #1117)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.evolution.loop import run_evolution_batch
from foundry_x.trace.logger import TraceEvent
from tests._harness_fixture import install_load_check_prerequisites

_BASE_TS = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)


def _event(kind: str, offset: float, payload: dict, *, event_id: str) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        session_id="sess-batch-zero-edit",
        timestamp=(_BASE_TS + timedelta(seconds=offset)).isoformat(),
        kind=kind,
        payload=payload,
    )


def _write_harness(tmp_path: Path) -> Path:
    harness_dir = tmp_path / "harness"
    tests_dir = harness_dir / "tests"
    tests_dir.mkdir(parents=True)
    (harness_dir / "system_prompt.txt").write_text("original\n", encoding="utf-8")
    (tests_dir / "test_gate.py").write_text(
        """
def test_original_content():
    assert open("system_prompt.txt").read() == "original\\n"
""".lstrip(),
        encoding="utf-8",
    )
    install_load_check_prerequisites(harness_dir)
    return harness_dir


class TestZeroEditFailureClass:
    """Issue #1117: failure classes that always return [] must still appear in results."""

    def test_zero_edit_failure_class_appears_in_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure class returning [] is recorded with verdict=None, not dropped."""
        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return []

        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-zero-edit", events, harness_dir)

        assert result.total_failures == 1
        assert len(result.results) == 1
        assert result.results[0].proposed_edits == []
        assert result.results[0].verdict is None

    def test_zero_edit_and_producing_failure_classes_both_in_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch with one class producing edits and another returning [] shows both."""
        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event(
                "tool_error",
                1.0,
                {"error": "no such tool: frobnicate"},
                event_id="e-wrong-tool",
            ),
            _event("tool_error", 2.0, {"error": "traceback occurred"}, event_id="e-tool-err"),
        ]

        proposed_edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="fix",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
        )

        call_count = 0

        def mock_propose(self, harness_dir, failure, current_diff=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [proposed_edit]
            return []

        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-mixed-edit", events, harness_dir)

        assert result.total_failures >= 2, "batch should have found multiple failure classes"
        assert len(result.results) == result.total_failures, (
            "len(results) must equal total_failures (issue #1117 invariant)"
        )
        results_with_edits = [r for r in result.results if r.proposed_edits]
        results_without_edits = [r for r in result.results if not r.proposed_edits]
        assert len(results_with_edits) >= 1, "at least one result should have proposed edits"
        assert len(results_without_edits) >= 1, "at least one result should have zero edits"
        for r in results_without_edits:
            assert r.verdict is None, "zero-edit result must have verdict=None"

    def test_total_failures_equals_len_results_invariant(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BatchEvolutionResult.total_failures == len(results) always holds."""
        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        monkeypatch.setattr(
            Evolver, "propose", lambda self, harness_dir, failure, current_diff=None: []
        )
        result = run_evolution_batch("sess-invariant", events, harness_dir)

        assert result.total_failures == len(result.results), (
            f"total_failures ({result.total_failures}) != len(results) ({len(result.results)})"
        )

    def test_zero_edit_result_has_correct_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A zero-edit EvolutionResult carries the expected fields from the failure report."""
        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        monkeypatch.setattr(
            Evolver, "propose", lambda self, harness_dir, failure, current_diff=None: []
        )
        result = run_evolution_batch("sess-fields", events, harness_dir)

        assert len(result.results) == 1
        zero_edit_result = result.results[0]
        assert zero_edit_result.failure_report is not None
        assert zero_edit_result.failure_class is not None
        assert zero_edit_result.failure_class != "clean"
        assert zero_edit_result.proposed_edits == []
        assert zero_edit_result.verdict is None
        assert zero_edit_result.evolver_duration_ms is not None
        assert zero_edit_result.harness_version is not None
        assert zero_edit_result.started_at is not None
        assert zero_edit_result.completed_at is not None
