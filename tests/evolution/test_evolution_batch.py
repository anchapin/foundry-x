"""Tests for run_evolution_batch — zero-edit failure-class path (issue #1117),
deduplication (issue #1258), and Critic dedup (issue #1456)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from foundry_x.evolution.critic import Critic, CriticVerdict
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

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return []

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return []

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-zero-edit", events, harness_dir)

        assert result.total_failures == 1
        assert len(result.results) == 1
        assert result.results[0].proposed_edits == []
        assert result.results[0].verdict is None

    def test_zero_edit_and_producing_failure_classes_both_in_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A batch with one class producing edits and another returning [] shows both.

        When batch_edits is empty, propose() is called as fallback. If some
        propose() calls return [] while others return edits, both zero-edit
        and edit-producing results appear (issue #1117 invariant).
        """
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

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return []

        call_count = 0

        def mock_propose(self, harness_dir, failure, current_diff=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [proposed_edit]
            return []

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
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

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return []

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return []

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Evolver, "propose", mock_propose)
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

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return []

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return []

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Evolver, "propose", mock_propose)
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


class TestBatchDeduplication:
    """Issue #1258: BatchEvolutionResult.proposed_edits has at most one edit per target_file."""

    def test_proposed_edits_deduplicated_by_target_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When propose_batch returns edits to the same file, only the first is retained.

        Deduplication is performed by propose_batch itself (issue #1258).
        When use_batch_attribution=True, propose() is NOT called (issue #1344),
        so deduplication relies entirely on propose_batch.
        """
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

        edit_to_prompt = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="fix prompt",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
        )
        edit_to_prompt2 = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="fix prompt differently",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+also new\n",
        )

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return [edit_to_prompt, edit_to_prompt2]

        def mock_propose(self, harness_dir, failure, current_diff=None):
            return [edit_to_prompt]

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-dedup", events, harness_dir, no_verify=True)

        assert result.total_failures >= 2
        assert len(result.results) == result.total_failures
        target_files = [edit.target_file for edit in result.proposed_edits]
        assert target_files.count("harness/system_prompt.txt") <= 1, (
            "proposed_edits should have at most one edit per target_file"
        )

    def test_different_target_files_both_retained(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When failures propose edits to different files, both are retained."""
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

        edit_to_prompt = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="fix prompt",
            unified_diff="--- a/harness/system_prompt.txt\n+++ b/harness/system_prompt.txt\n@@ -1 +1 @@\n-old\n+new\n",
        )
        edit_to_hook = ProposedEdit(
            target_file="harness/hooks/my_hook.py",
            rationale="fix hook",
            unified_diff="--- a/harness/hooks/my_hook.py\n+++ b/harness/hooks/my_hook.py\n@@ -1 +1 @@\n-old\n+new\n",
        )

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return [edit_to_prompt, edit_to_hook]

        call_count = 0

        def mock_propose(self, harness_dir, failure, current_diff=None):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return [edit_to_prompt]
            return [edit_to_hook]

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-multi-target", events, harness_dir, no_verify=True)

        assert result.total_failures >= 2
        assert len(result.results) == result.total_failures
        assert len(result.proposed_edits) == 2, "both edits to different files should be retained"
        target_files = {edit.target_file for edit in result.proposed_edits}
        assert "harness/system_prompt.txt" in target_files
        assert "harness/hooks/my_hook.py" in target_files


class TestNoRedundantProposeCalls:
    """Issue #1344: when propose_batch returns non-empty edits, propose() must NOT be called."""

    def test_run_evolution_batch_no_redundant_propose_calls(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When batch_edits is non-empty, propose() is not called per failure."""
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

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return [proposed_edit]

        propose_call_count = 0

        def mock_propose(self, harness_dir, failure, current_diff=None):
            nonlocal propose_call_count
            propose_call_count += 1
            return [proposed_edit]

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-no-redundant", events, harness_dir, no_verify=True)

        assert propose_call_count == 0, (
            f"propose() should NOT be called when propose_batch returns non-empty edits, "
            f"but was called {propose_call_count} times"
        )
        assert len(result.results) >= 1, "should have results for each failure class"

    def test_run_evolution_batch_fallback_propose_when_batch_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When batch_edits is empty, propose() is called per failure as fallback."""
        harness_dir = _write_harness(tmp_path)
        events = [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event("error", 1.0, {"error": "oops"}, event_id="e2"),
        ]

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return []

        propose_call_count = 0

        def mock_propose(self, harness_dir, failure, current_diff=None):
            nonlocal propose_call_count
            propose_call_count += 1
            return []

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Evolver, "propose", mock_propose)
        result = run_evolution_batch("sess-fallback", events, harness_dir)

        assert propose_call_count == 1, (
            f"propose() should be called once as fallback when batch_edits is empty, "
            f"but was called {propose_call_count} times"
        )
        assert result.total_failures == 1


class TestCriticBatchDedup:
    """Issue #1456: batch path must not re-evaluate identical diffs per failure class."""

    @staticmethod
    def _failure_events() -> list[TraceEvent]:
        return [
            _event("user_prompt", 0.0, {"prompt": "hello"}, event_id="e1"),
            _event(
                "tool_error",
                1.0,
                {"error": "no such tool: frobnicate"},
                event_id="e-wrong-tool",
            ),
            _event("tool_error", 2.0, {"error": "traceback occurred"}, event_id="e-tool-err"),
        ]

    def test_evaluate_called_once_per_unique_diff(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """critic.evaluate is called at most once per unique unified_diff."""
        harness_dir = _write_harness(tmp_path)
        events = self._failure_events()

        edit = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="fix prompt",
            unified_diff=(
                "--- a/harness/system_prompt.txt\n"
                "+++ b/harness/system_prompt.txt\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
        )

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return [edit]

        evaluate_calls: list[str] = []

        def spy_evaluate(self, diff, **kwargs):
            evaluate_calls.append(diff)
            return CriticVerdict(verdict=True, edit_index=kwargs.get("edit_index"))

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Critic, "evaluate", spy_evaluate)
        result = run_evolution_batch("sess-critic-dedup", events, harness_dir)

        unique_diffs = {edit.unified_diff}
        assert len(evaluate_calls) <= len(unique_diffs), (
            f"critic.evaluate called {len(evaluate_calls)} times but there are only "
            f"{len(unique_diffs)} unique diffs"
        )
        assert len(result.results) >= 2, "all failure classes should receive a verdict"
        for r in result.results:
            assert r.verdict is not None, "every result should have a (shared) verdict"

    def test_evaluate_called_once_per_unique_diff_multi_target(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With two unique diffs to different files, evaluate is called at most twice."""
        harness_dir = _write_harness(tmp_path)
        events = self._failure_events()

        edit_a = ProposedEdit(
            target_file="harness/system_prompt.txt",
            rationale="fix prompt",
            unified_diff=(
                "--- a/harness/system_prompt.txt\n"
                "+++ b/harness/system_prompt.txt\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
        )
        edit_b = ProposedEdit(
            target_file="harness/hooks/my_hook.py",
            rationale="fix hook",
            unified_diff=(
                "--- a/harness/hooks/my_hook.py\n"
                "+++ b/harness/hooks/my_hook.py\n"
                "@@ -1 +1 @@\n-old\n+new\n"
            ),
        )

        def mock_propose_batch(self, harness_dir, batch_report, current_diff=None):
            return [edit_a, edit_b]

        evaluate_calls: list[str] = []

        def spy_evaluate(self, diff, **kwargs):
            evaluate_calls.append(diff)
            return CriticVerdict(verdict=True, edit_index=kwargs.get("edit_index"))

        monkeypatch.setattr(Evolver, "propose_batch", mock_propose_batch)
        monkeypatch.setattr(Critic, "evaluate", spy_evaluate)
        result = run_evolution_batch("sess-critic-dedup-multi", events, harness_dir)

        unique_diffs = {edit_a.unified_diff, edit_b.unified_diff}
        assert len(evaluate_calls) <= len(unique_diffs), (
            f"critic.evaluate called {len(evaluate_calls)} times but there are only "
            f"{len(unique_diffs)} unique diffs"
        )
        assert len(result.results) >= 2
        for r in result.results:
            assert r.verdict is not None
