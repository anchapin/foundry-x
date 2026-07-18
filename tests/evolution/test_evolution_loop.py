"""Integration tests for the full Digester → Evolver → Critic pipeline (issue #255)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.evolution.loop import EvolutionResult, run_evolution_step, run_evolution_step_async
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

        def mock_evaluate(self, proposed_diff, *, edit_index=None, failure_class=None):
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

        def mock_evaluate(self, proposed_diff, *, edit_index=None, failure_class=None):
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

        large_diff_lines = ["+line{}".format(i) for i in range(250)]
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
