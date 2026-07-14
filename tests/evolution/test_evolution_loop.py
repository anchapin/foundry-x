"""Integration tests for the full Digester → Evolver → Critic pipeline (issue #255)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.evolution.loop import EvolutionResult, run_evolution_step
from foundry_x.trace.logger import TraceEvent
from tests._harness_fixture import install_load_check_prerequisites


_BASE_TS = datetime(2026, 7, 10, 12, 0, 0, tzinfo=timezone.utc)


def _event(kind: str, offset: float, payload: dict, *, event_id: str) -> TraceEvent:
    return TraceEvent(
        event_id=event_id,
        session_id="sess-loop-test",
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


class TestEditsToDiff:
    """Unit tests for the _edits_to_diff helper (tested via run_evolution_step)."""

    def test_single_edit(self):
        from foundry_x.evolution.loop import _edits_to_diff

        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="test",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
        )
        result = _edits_to_diff([edit])
        assert "--- a/harness/system_prompt.txt" in result
        assert "+new" in result

    def test_multiple_edits(self):
        from foundry_x.evolution.loop import _edits_to_diff

        edit1 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="test1",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new1\n",
        )
        edit2 = ProposedEdit(
            target_file="harness/manifest.json",
            rationale="test2",
            unified_diff="--- a/harness/manifest.json\n+++ b/harness/manifest.json\n@@ -1 +1 @@\n-old\n+new2\n",
        )
        result = _edits_to_diff([edit1, edit2])
        assert "--- a/harness/system_prompt.txt" in result
        assert "+new1" in result
        assert "--- a/harness/manifest.json" in result
        assert "+new2" in result

    def test_empty_list(self):
        from foundry_x.evolution.loop import _edits_to_diff

        result = _edits_to_diff([])
        assert result == ""


class TestRunEvolutionStep:
    def test_clean_report_short_circuits(self, tmp_path: Path):
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("outcome", 1.0, {"status": "success"}, event_id="e2"),
        ]

        result = run_evolution_step("sess-clean", events, harness_dir)

        assert result.failure_report.proposed_class == "clean"
        assert result.proposed_edits == []
        assert result.verdict is None

    def test_empty_edits_short_circuits(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return []

        monkeypatch.setattr(Evolver, "propose", mock_propose)

        result = run_evolution_step("sess-no-edits", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert result.proposed_edits == []
        assert result.verdict is None

    def test_full_pipeline_runs_critic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        proposed_edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix the failure",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
        )

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return [proposed_edit]

        monkeypatch.setattr(Evolver, "propose", mock_propose)

        result = run_evolution_step("sess-full-pipeline", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert result.proposed_edits == [proposed_edit]
        assert result.verdict is not None
        assert isinstance(result.verdict, CriticVerdict)


class TestEvolutionResultModel:
    def test_result_model_fields(self):
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="test failure",
            proposed_class="tool-error",
        )
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            proposed_edits=[],
            verdict=None,
        )
        assert result.session_id == "sess-test"
        assert result.failure_report.summary == "test failure"
        assert result.proposed_edits == []
        assert result.verdict is None

    def test_result_model_with_verdict(self):
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="test failure",
            proposed_class="tool-error",
        )
        verdict = CriticVerdict(verdict=True, passed_checks=["pytest"], failed_checks=[])
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            proposed_edits=[],
            verdict=verdict,
        )
        assert result.verdict is not None
        assert result.verdict.verdict is True
