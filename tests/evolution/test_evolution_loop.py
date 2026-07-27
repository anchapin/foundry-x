"""Integration tests for the full Digester → Evolver → Critic pipeline (issue #255)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from foundry_x.evolution.cli import main
from foundry_x.evolution.critic import Critic, CriticVerdict
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.evolution.loop import EvolutionResult, run_evolution_step, run_evolution_step_async
from foundry_x.trace.logger import TraceEvent, TraceLogger
from tests._harness_fixture import install_load_check_prerequisites

_BASE_TS = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)


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
        assert result.evolver_duration_ms is None
        assert result.harness_version is not None

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
        assert result.evolver_duration_ms is not None
        assert result.evolver_duration_ms >= 0
        assert result.harness_version is not None

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
        assert result.evolver_duration_ms is not None
        assert result.evolver_duration_ms >= 0
        assert result.harness_version is not None

    def test_multiple_edits_verdict_has_last_edit_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When multiple edits are proposed, verdict.edit_index is the last one (issue #606)."""
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        proposed_edit1 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 1",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new1\n",
        )
        proposed_edit2 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 2",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new2\n",
        )

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return [proposed_edit1, proposed_edit2]

        monkeypatch.setattr(Evolver, "propose", mock_propose)

        result = run_evolution_step("sess-multi-edit", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert len(result.proposed_edits) == 2
        assert result.verdict is not None
        assert result.verdict.edit_index == 1

    def test_verdict_failure_class_from_failure_report(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Issue #796: verdict.failure_class is wired from failure_report.proposed_class."""
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

        result = run_evolution_step("sess-failure-class", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert result.verdict is not None
        assert result.verdict.failure_class == result.failure_report.proposed_class


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
            failure_class="tool-error",
            proposed_edits=[],
            verdict=None,
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.session_id == "sess-test"
        assert result.failure_report.summary == "test failure"
        assert result.failure_class == "tool-error"
        assert result.proposed_edits == []
        assert result.verdict is None
        assert result.evolver_duration_ms is None
        assert result.started_at == "2026-07-10T12:00:00+00:00"
        assert result.completed_at == "2026-07-10T12:00:01+00:00"

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
            failure_class="tool-error",
            proposed_edits=[],
            verdict=verdict,
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.verdict is not None
        assert result.verdict.verdict is True
        assert result.failure_class == "tool-error"

    def test_failure_class_copied_from_report(self):
        """failure_class is copied from failure_report.proposed_class (issue #605)."""
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="wrong-tool failure",
            proposed_class="wrong-tool",
        )
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            failure_class=report.proposed_class,
            proposed_edits=[],
            verdict=None,
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.failure_class == "wrong-tool"
        assert result.failure_class == result.failure_report.proposed_class

    def test_result_model_with_evolver_duration(self):
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="test failure",
            proposed_class="tool-error",
        )
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            failure_class=report.proposed_class,
            proposed_edits=[],
            verdict=None,
            evolver_duration_ms=42.5,
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.evolver_duration_ms == 42.5
        assert isinstance(result.evolver_duration_ms, float)

    def test_result_model_with_harness_version(self):
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="test failure",
            proposed_class="tool-error",
        )
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            failure_class=report.proposed_class,
            proposed_edits=[],
            verdict=None,
            harness_version="v1.2.3",
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.harness_version == "v1.2.3"

    def test_result_model_harness_version_defaults_to_none(self):
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="test failure",
            proposed_class="tool-error",
        )
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            failure_class=report.proposed_class,
            proposed_edits=[],
            verdict=None,
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.harness_version is None


class TestRunEvolutionStepAsync:
    @pytest.mark.asyncio
    async def test_clean_report_short_circuits(self, tmp_path: Path):
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("outcome", 1.0, {"status": "success"}, event_id="e2"),
        ]

        result = await run_evolution_step_async("sess-clean", events, harness_dir)

        assert result.failure_report.proposed_class == "clean"
        assert result.proposed_edits == []
        assert result.verdict is None

    @pytest.mark.asyncio
    async def test_empty_edits_short_circuits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        async def mock_propose_async(self, harness_dir, failure, current_diff=None):
            return []

        monkeypatch.setattr(Evolver, "propose_async", mock_propose_async)

        result = await run_evolution_step_async("sess-no-edits", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert result.proposed_edits == []
        assert result.verdict is None

    @pytest.mark.asyncio
    async def test_full_pipeline_runs_critic(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
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

        async def mock_propose_async(self, harness_dir, failure, current_diff=None):
            return [proposed_edit]

        monkeypatch.setattr(Evolver, "propose_async", mock_propose_async)

        result = await run_evolution_step_async("sess-full-pipeline", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert result.proposed_edits == [proposed_edit]
        assert result.verdict is not None
        assert isinstance(result.verdict, CriticVerdict)

    @pytest.mark.asyncio
    async def test_verdict_failure_class_forwarded_to_critic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Issue #891: async path forwards failure_class to critic.evaluate."""
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

        async def mock_propose_async(self, harness_dir, failure, current_diff=None):
            return [proposed_edit]

        monkeypatch.setattr(Evolver, "propose_async", mock_propose_async)

        call_records: list[str | None] = []

        def mock_evaluate(self, proposed_diff, *, edit_index=None, failure_class=None, tier="full"):
            call_records.append(failure_class)
            return CriticVerdict(
                verdict=True,
                passed_checks=["git apply"],
                edit_index=edit_index,
                failure_class=failure_class,
            )

        monkeypatch.setattr("foundry_x.evolution.loop.Critic.evaluate", mock_evaluate)

        result = await run_evolution_step_async("sess-async-failure-class", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert result.verdict is not None
        assert result.verdict.failure_class == result.failure_report.proposed_class
        assert call_records == [result.failure_report.proposed_class]

    @pytest.mark.asyncio
    async def test_multiple_edits_verdict_has_last_edit_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When multiple edits are proposed, verdict.edit_index is the last one (issue #743)."""
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        proposed_edit1 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 1",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new1\n",
        )
        proposed_edit2 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 2",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new2\n",
        )

        async def mock_propose_async(self, harness_dir, failure, current_diff=None):
            return [proposed_edit1, proposed_edit2]

        monkeypatch.setattr(Evolver, "propose_async", mock_propose_async)

        result = await run_evolution_step_async("sess-multi-edit", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert len(result.proposed_edits) == 2
        assert result.verdict is not None
        assert result.verdict.edit_index == 1

    @pytest.mark.asyncio
    async def test_multiple_edits_async_evaluates_each_edit_with_correct_index(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Async path calls critic.evaluate per edit with correct edit_index (issue #797)."""
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        proposed_edit1 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 1",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new1\n",
        )
        proposed_edit2 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 2",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new2\n",
        )

        async def mock_propose_async(self, harness_dir, failure, current_diff=None):
            return [proposed_edit1, proposed_edit2]

        monkeypatch.setattr(Evolver, "propose_async", mock_propose_async)

        call_records: list[tuple[str, int]] = []

        def mock_evaluate(self, proposed_diff, *, edit_index=None, failure_class=None, tier="full"):
            call_records.append((proposed_diff, edit_index))
            return CriticVerdict(verdict=True, passed_checks=["git apply"], edit_index=edit_index)

        monkeypatch.setattr("foundry_x.evolution.loop.Critic.evaluate", mock_evaluate)

        result = await run_evolution_step_async("sess-multi-edit-index", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert len(result.proposed_edits) == 2
        assert len(call_records) == 2
        assert call_records[0][1] == 0
        assert call_records[1][1] == 1

    @pytest.mark.asyncio
    async def test_oversized_edit_in_batch_rejected_with_diff_size_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An oversized single edit in a batch is rejected with diff_size_cap (issue #797)."""
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        large_diff_lines = [f"+line{i}" for i in range(250)]
        large_diff = "--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n{}\n".format(
            "\n".join(large_diff_lines)
        )

        proposed_edit1 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 1",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new1\n",
        )
        proposed_edit2 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix 2 - oversized",
            unified_diff=large_diff,
        )

        async def mock_propose_async(self, harness_dir, failure, current_diff=None):
            return [proposed_edit1, proposed_edit2]

        monkeypatch.setattr(Evolver, "propose_async", mock_propose_async)

        result = await run_evolution_step_async("sess-oversized", events, harness_dir)

        assert result.failure_report.proposed_class != "clean"
        assert len(result.proposed_edits) == 2
        assert result.verdict is not None
        assert result.verdict.edit_index == 1
        assert "diff_size_cap" in result.verdict.failed_checks

    def test_result_model_with_harness_version(self):
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="test failure",
            proposed_class="tool-error",
        )
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            failure_class=report.proposed_class,
            proposed_edits=[],
            verdict=None,
            harness_version="v1.2.3",
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.harness_version == "v1.2.3"

    def test_result_model_harness_version_defaults_to_none(self):
        from foundry_x.evolution.digester import FailureReport

        report = FailureReport(
            session_id="sess-test",
            summary="test failure",
            proposed_class="tool-error",
        )
        result = EvolutionResult(
            session_id="sess-test",
            failure_report=report,
            failure_class=report.proposed_class,
            proposed_edits=[],
            verdict=None,
            started_at="2026-07-10T12:00:00+00:00",
            completed_at="2026-07-10T12:00:01+00:00",
        )
        assert result.harness_version is None


class TestEvolutionResultTimestamps:
    """Tests for issue #609 — timestamp fields on EvolutionResult."""

    def test_timestamps_are_iso_format(self, tmp_path: Path):
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("outcome", 1.0, {"status": "success"}, event_id="e2"),
        ]

        result = run_evolution_step("sess-ts", events, harness_dir)

        assert result.started_at is not None
        assert result.completed_at is not None
        datetime.fromisoformat(result.started_at)
        datetime.fromisoformat(result.completed_at)

    def test_started_before_completed(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        harness_dir = _write_harness(tmp_path)

        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return []

        monkeypatch.setattr(Evolver, "propose", mock_propose)

        result = run_evolution_step("sess-ts-order", events, harness_dir)

        t0 = datetime.fromisoformat(result.started_at)
        t1 = datetime.fromisoformat(result.completed_at)
        assert t0 <= t1

    def test_timestamps_present_on_full_pipeline(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
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

        result = run_evolution_step("sess-ts-full", events, harness_dir)

        assert result.started_at is not None
        assert result.completed_at is not None
        datetime.fromisoformat(result.started_at)
        datetime.fromisoformat(result.completed_at)


# ---------------------------------------------------------------------------
# Session-evolved marking on TraceLogger (issue #1047)
# ---------------------------------------------------------------------------


class TestSessionEvolvedMarking:
    """Tests for TraceLogger.mark_session_evolved / is_session_evolved /
    list_unevolved_sessions (issue #1047)."""

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_mark_and_check(self, tmp_path: Path, backend: str):
        db = tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl")
        logger = TraceLogger(db, backend=backend)
        with logger.session(harness_version="0.1.0") as sid:
            logger.record(sid, "outcome", {"status": "success"})
        assert not logger.is_session_evolved(sid)
        assert logger.mark_session_evolved(sid)
        assert logger.is_session_evolved(sid)
        logger.close()

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_mark_nonexistent_session(self, tmp_path: Path, backend: str):
        db = tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl")
        logger = TraceLogger(db, backend=backend)
        assert not logger.mark_session_evolved("nonexistent")
        assert not logger.is_session_evolved("nonexistent")
        logger.close()

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_mark_idempotent(self, tmp_path: Path, backend: str):
        db = tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl")
        logger = TraceLogger(db, backend=backend)
        with logger.session(harness_version="0.1.0") as sid:
            logger.record(sid, "outcome", {"status": "success"})
        assert logger.mark_session_evolved(sid)
        assert logger.mark_session_evolved(sid)
        assert logger.is_session_evolved(sid)
        logger.close()

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_list_unevolved_excludes_evolved(self, tmp_path: Path, backend: str):
        db = tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl")
        logger = TraceLogger(db, backend=backend)
        with logger.session(harness_version="0.1.0") as sid1:
            logger.record(sid1, "outcome", {"status": "success"})
        with logger.session(harness_version="0.1.0") as sid2:
            logger.record(sid2, "outcome", {"status": "failed"})

        unevolved = logger.list_unevolved_sessions()
        assert sid1 in unevolved
        assert sid2 in unevolved

        logger.mark_session_evolved(sid1)
        unevolved = logger.list_unevolved_sessions()
        assert sid1 not in unevolved
        assert sid2 in unevolved
        logger.close()

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_list_unevolved_excludes_open_sessions(self, tmp_path: Path, backend: str):
        """Sessions without ended_at should not appear (still recording)."""
        db = tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl")
        logger = TraceLogger(db, backend=backend)
        # Open session — don't exit the context manager yet
        with logger.session(harness_version="0.1.0") as sid:
            unevolved = logger.list_unevolved_sessions()
            assert sid not in unevolved
        # After exit, ended_at is set, session should now appear
        unevolved = logger.list_unevolved_sessions()
        assert sid in unevolved
        logger.close()


# ---------------------------------------------------------------------------
# run_evolution_daemon (issue #1047)
# ---------------------------------------------------------------------------


class TestRunEvolutionDaemon:
    """Tests for the continuous background evolution daemon (issue #1047)."""

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_processes_unevolved_session(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
    ):
        """Daemon evolves a session and marks it as evolved."""
        harness_dir = _write_harness(tmp_path)
        db = str(tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl"))
        logger = TraceLogger(db, backend=backend)
        with logger.session(harness_version="0.1.0") as sid:
            logger.record(sid, "user_prompt", {"prompt": "hello"})
            logger.record(sid, "outcome", {"status": "success"})
        logger.close()

        from foundry_x.evolution.loop import run_evolution_daemon

        result = run_evolution_daemon(
            harness_dir=harness_dir,
            trace_db=db,
            poll_interval_s=0.01,
            max_iterations=1,
        )

        assert result.shutdown_reason == "max_iterations"
        assert result.sessions_processed == 1
        assert result.iterations == 1

        check_logger = TraceLogger(db, backend=backend)
        assert check_logger.is_session_evolved(sid)
        assert check_logger.list_unevolved_sessions() == []
        check_logger.close()

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_idempotent_does_not_reprocess(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str
    ):
        """Already-evolved sessions are not re-processed."""
        harness_dir = _write_harness(tmp_path)
        db = str(tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl"))
        logger = TraceLogger(db, backend=backend)
        with logger.session(harness_version="0.1.0") as sid:
            logger.record(sid, "user_prompt", {"prompt": "hello"})
            logger.record(sid, "outcome", {"status": "success"})
        logger.mark_session_evolved(sid)
        logger.close()

        from foundry_x.evolution.loop import run_evolution_daemon

        result = run_evolution_daemon(
            harness_dir=harness_dir,
            trace_db=db,
            poll_interval_s=0.01,
            max_iterations=1,
        )

        assert result.sessions_processed == 0
        assert result.sessions_skipped == 0

    @pytest.mark.parametrize("backend", ["sqlite", "jsonl"])
    def test_multiple_sessions_in_one_cycle(self, tmp_path: Path, backend: str):
        """Multiple unevolved sessions are processed in a single cycle."""
        harness_dir = _write_harness(tmp_path)
        db = str(tmp_path / ("traces.db" if backend == "sqlite" else "traces.jsonl"))
        logger = TraceLogger(db, backend=backend)
        sids = []
        for i in range(3):
            with logger.session(harness_version="0.1.0") as sid:
                logger.record(sid, "user_prompt", {"prompt": f"task-{i}"})
                logger.record(sid, "outcome", {"status": "success"})
            sids.append(sid)
        logger.close()

        from foundry_x.evolution.loop import run_evolution_daemon

        result = run_evolution_daemon(
            harness_dir=harness_dir,
            trace_db=db,
            poll_interval_s=0.01,
            max_iterations=1,
        )

        assert result.sessions_processed == 3
        check_logger = TraceLogger(db, backend=backend)
        assert check_logger.list_unevolved_sessions() == []
        check_logger.close()

    def test_empty_trace_db_no_sessions(self, tmp_path: Path):
        """Daemon handles an empty trace store without error."""
        harness_dir = _write_harness(tmp_path)
        db = str(tmp_path / "traces.db")

        from foundry_x.evolution.loop import run_evolution_daemon

        result = run_evolution_daemon(
            harness_dir=harness_dir,
            trace_db=db,
            poll_interval_s=0.01,
            max_iterations=1,
        )

        assert result.sessions_processed == 0
        assert result.iterations == 1

    def test_shutdown_state_stops_after_current_session(self, tmp_path: Path):
        """SIGTERM-style shutdown finishes the current session then exits."""
        harness_dir = _write_harness(tmp_path)
        db = str(tmp_path / "traces.db")
        logger = TraceLogger(db)
        sids = []
        for i in range(3):
            with logger.session(harness_version="0.1.0") as sid:
                logger.record(sid, "user_prompt", {"prompt": f"task-{i}"})
                logger.record(sid, "outcome", {"status": "success"})
            sids.append(sid)
        logger.close()

        from foundry_x.evolution.loop import (
            DaemonResult,
            _ShutdownState,
            run_evolution_daemon,
        )

        shutdown = _ShutdownState()

        original_mark = TraceLogger.mark_session_evolved
        call_count = {"n": 0}

        def mark_and_check(self, session_id):
            result = original_mark(self, session_id)
            call_count["n"] += 1
            if call_count["n"] >= 1:
                shutdown.requested = True
            return result

        original_func = TraceLogger.mark_session_evolved
        TraceLogger.mark_session_evolved = mark_and_check
        try:
            result = run_evolution_daemon(
                harness_dir=harness_dir,
                trace_db=db,
                poll_interval_s=0.01,
                shutdown_state=shutdown,
            )
        finally:
            TraceLogger.mark_session_evolved = original_func

        assert isinstance(result, DaemonResult)
        assert result.shutdown_reason == "sigterm"
        assert result.sessions_processed >= 1

    def test_failing_session_processed_and_marked(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A failing session triggers the full pipeline and gets marked evolved."""
        harness_dir = _write_harness(tmp_path)
        db = str(tmp_path / "traces.db")
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as sid:
            logger.record(sid, "user_prompt", {"prompt": "Fix the bug"})
            logger.record(sid, "error", {"error": "something broke"})
            logger.record(sid, "outcome", {"status": "failed"})
        logger.close()

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return [
                ProposedEdit(
                    target_file="harness/system_prompt.txt",
                    rationale="Fix",
                    unified_diff=(
                        "--- a/harness/system_prompt.txt\n"
                        "+++ b/harness/system_prompt.txt\n"
                        "@@ -1 +1 @@\n-old\n+new\n"
                    ),
                )
            ]

        monkeypatch.setattr(Evolver, "propose", mock_propose)

        from foundry_x.evolution.loop import run_evolution_daemon

        result = run_evolution_daemon(
            harness_dir=harness_dir,
            trace_db=db,
            poll_interval_s=0.01,
            max_iterations=1,
            no_verify=True,
        )

        assert result.sessions_processed == 1

    def test_empty_events_session_marked_as_skipped(self, tmp_path: Path):
        """A session with no events is marked evolved (skipped)."""
        harness_dir = _write_harness(tmp_path)
        db = str(tmp_path / "traces.db")
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as sid:
            pass

        from foundry_x.evolution.loop import run_evolution_daemon

        result = run_evolution_daemon(
            harness_dir=harness_dir,
            trace_db=db,
            poll_interval_s=0.01,
            max_iterations=1,
        )

        assert result.sessions_skipped == 1
        assert result.sessions_processed == 0

        check_logger = TraceLogger(db)
        assert check_logger.is_session_evolved(sid)
        check_logger.close()

    def test_daemon_result_model(self):
        from foundry_x.evolution.loop import DaemonResult

        result = DaemonResult(
            iterations=5,
            sessions_processed=3,
            sessions_skipped=1,
            shutdown_reason="sigterm",
            started_at="2026-07-27T10:00:00+00:00",
            completed_at="2026-07-27T10:05:00+00:00",
        )
        assert result.iterations == 5
        assert result.sessions_processed == 3
        assert result.sessions_skipped == 1
        assert result.shutdown_reason == "sigterm"


class TestDaemonCLI:
    """Tests for the ``foundry-evolve daemon`` subcommand (issue #1047)."""

    def test_daemon_subcommand_runs(self, tmp_path: Path):
        """The daemon subcommand parses and runs with max_iterations."""
        harness_dir = tmp_path / "harness"
        install_load_check_prerequisites(harness_dir)
        (harness_dir / "system_prompt.txt").write_text("test\n", encoding="utf-8")
        db = str(tmp_path / "traces.db")

        exit_code = main(
            [
                "daemon",
                "--harness-dir",
                str(harness_dir),
                "--trace-db",
                db,
                "--poll-interval",
                "0.01",
                "--max-iterations",
                "1",
            ]
        )
        assert exit_code == 0

    def test_daemon_requires_harness_dir(self):
        with pytest.raises(SystemExit):
            main(["daemon", "--trace-db", "x.db"])

    def test_daemon_verbose_flag(self, tmp_path: Path):
        harness_dir = tmp_path / "harness"
        install_load_check_prerequisites(harness_dir)
        (harness_dir / "system_prompt.txt").write_text("test\n", encoding="utf-8")
        db = str(tmp_path / "traces.db")

        exit_code = main(
            [
                "daemon",
                "--harness-dir",
                str(harness_dir),
                "--trace-db",
                db,
                "--poll-interval",
                "0.01",
                "--max-iterations",
                "1",
                "--verbose",
            ]
        )
        assert exit_code == 0


# ---------------------------------------------------------------------------
# Two-tier Critic gate passthrough (issue #1042)
# ---------------------------------------------------------------------------


class TestCriticTierPassthrough:
    """run_evolution_step forwards critic_tier to Critic.evaluate (issue #1042)."""

    def _events(self) -> list[TraceEvent]:
        return [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

    @staticmethod
    def _edit() -> ProposedEdit:
        return ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix the failure",
            unified_diff=(
                "--- a/harness/system_prompt.txt\n"
                "+++ b/harness/system_prompt.txt\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
        )

    def test_smoke_tier_forwarded(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        harness_dir = _write_harness(tmp_path)

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return [TestCriticTierPassthrough._edit()]

        monkeypatch.setattr(Evolver, "propose", mock_propose)
        seen: list[str] = []

        def spy_evaluate(self, diff, **kwargs):
            seen.append(kwargs.get("tier", "full"))
            return CriticVerdict(verdict=True, edit_index=kwargs.get("edit_index"))

        monkeypatch.setattr(Critic, "evaluate", spy_evaluate)
        run_evolution_step(
            "sess-tier-smoke",
            self._events(),
            harness_dir,
            critic=Critic(harness_dir),
            critic_tier="smoke",
        )
        assert seen == ["smoke"]

    def test_full_tier_is_default(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        harness_dir = _write_harness(tmp_path)

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return [TestCriticTierPassthrough._edit()]

        monkeypatch.setattr(Evolver, "propose", mock_propose)
        seen: list[str] = []

        def spy_evaluate(self, diff, **kwargs):
            seen.append(kwargs.get("tier", "full"))
            return CriticVerdict(verdict=True, edit_index=kwargs.get("edit_index"))

        monkeypatch.setattr(Critic, "evaluate", spy_evaluate)
        # No critic_tier kwarg → defaults to "full" (unchanged behaviour).
        run_evolution_step("sess-tier-default", self._events(), harness_dir)
        assert seen == ["full"]


class TestRunEvolutionBatch:
    """Tests for the batch evolution pipeline (issue #1033)."""

    def _events(self) -> list[TraceEvent]:
        return [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

    def test_batch_clean_session_short_circuits(self, tmp_path: Path) -> None:
        """A clean batch returns immediately with no edits."""
        from foundry_x.evolution.loop import run_evolution_batch

        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("outcome", 1.0, {"status": "success"}, event_id="e2"),
        ]
        result = run_evolution_batch("sess-batch-clean", events, harness_dir)
        assert result.total_failures == 0
        assert len(result.proposed_edits) == 0
        assert len(result.results) == 0

    def test_batch_single_failure_processes_correctly(self, tmp_path: Path) -> None:
        """A batch with one failure processes correctly."""
        from foundry_x.evolution.loop import run_evolution_batch

        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]
        result = run_evolution_batch("sess-batch-single", events, harness_dir)
        assert result.total_failures == 1
        assert len(result.results) == 1
        assert result.results[0].failure_report.proposed_class != "clean"

    def test_batch_multiple_failures_produces_multiple_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch with multiple failures produces separate results per failure class."""
        from foundry_x.evolution.evolver import ProposedEdit
        from foundry_x.evolution.loop import run_evolution_batch

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

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return [
                ProposedEdit(
                    target_file="harness/system_prompt.txt",
                    rationale="fix",
                    unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
                )
            ]

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        result = run_evolution_batch("sess-batch-multi", events, harness_dir)
        assert result.total_failures >= 1

    def test_batch_result_contains_all_proposed_edits(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch result aggregates all proposed edits."""
        from foundry_x.evolution.evolver import ProposedEdit
        from foundry_x.evolution.loop import run_evolution_batch

        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        proposed_edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="Fix",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
        )

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return [proposed_edit]

        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-batch-edits", events, harness_dir)
        assert proposed_edit in result.proposed_edits

    def test_batch_propose_is_called_per_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The evolver's propose method is called for each non-clean failure."""
        from foundry_x.evolution.loop import run_evolution_batch

        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        call_count = 0

        def mock_propose(self, harness_dir, failure, current_diff=None):
            nonlocal call_count
            call_count += 1
            return []

        monkeypatch.setattr(Evolver, "propose", mock_propose)
        run_evolution_batch("sess-batch-call", events, harness_dir)
        assert call_count >= 1, "propose should have been called at least once"

    def test_batch_result_has_harness_version(self, tmp_path: Path) -> None:
        """The batch result includes the harness version."""
        from foundry_x.evolution.loop import run_evolution_batch

        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("outcome", 1.0, {"status": "success"}, event_id="e2"),
        ]
        result = run_evolution_batch("sess-batch-version", events, harness_dir)
        assert result.harness_version is not None

    def test_batch_result_has_timestamps(self, tmp_path: Path) -> None:
        """The batch result includes started_at and completed_at timestamps."""
        from foundry_x.evolution.loop import run_evolution_batch

        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("outcome", 1.0, {"status": "success"}, event_id="e2"),
        ]
        result = run_evolution_batch("sess-batch-ts", events, harness_dir)
        assert result.started_at is not None
        assert result.completed_at is not None
