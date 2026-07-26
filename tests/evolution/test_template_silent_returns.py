"""Tests for issue #974: trace events on _propose_from_template silent early returns.

Every early-return path in ``Evolver._propose_from_template`` must emit a
``generation_attempt`` and a ``generation_exhausted`` trace event so the
failure is observable in KPI rollups instead of returning ``[]`` silently.

Paths tested:
- ``template is None`` (unknown failure class)
- ``not json_patch`` (JSON target with no patch)
- JSON merge-patch failure (corrupt fixture)
- ``not unified_diff`` (no-diff)
- ``EvolverGuardError`` from ``validate_edit`` (diff too large)
- ``run_evolution_step`` wiring ``trace_logger`` into the default Evolver
- Regression: the 7 known classes still produce edits
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from foundry_x.evolution.digester import FailureReport
from foundry_x.evolution.evolver import (
    GENERATION_ATTEMPT_KIND,
    GENERATION_EXHAUSTED_KIND,
    Evolver,
)
from foundry_x.evolution.loop import run_evolution_step
from foundry_x.trace.logger import TraceEvent, TraceLogger


def _make_failure(proposed_class: str = "unknown") -> FailureReport:
    return FailureReport(
        session_id="sess-974",
        summary="test failure",
        proposed_class=proposed_class,
    )


def _build_harness(tmp_path: Path) -> Path:
    """Minimal harness with system_prompt.txt for template proposals."""
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("You are a helpful agent.\n", encoding="utf-8")
    return harness_dir


class TestTemplateFailureTraceEvents:
    """Each early-return path must emit generation_attempt + generation_exhausted."""

    def test_unknown_class_emits_trace_events(self, tmp_path: Path) -> None:
        """``template is None`` emits both events with a descriptive error."""
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(trace_logger=trace_logger, session_id="sess-974")

        edits = evolver._propose_from_template(harness_dir, _make_failure("nonexistent-class"))

        assert edits == []
        kinds_recorded = [call.args[1] for call in trace_logger.record.call_args_list]
        assert GENERATION_ATTEMPT_KIND in kinds_recorded
        assert GENERATION_EXHAUSTED_KIND in kinds_recorded

        attempt_call = next(
            c for c in trace_logger.record.call_args_list if c.args[1] == GENERATION_ATTEMPT_KIND
        )
        payload = attempt_call.args[2]
        assert "nonexistent-class" in payload["error"]

    def test_unknown_class_emits_events_with_real_trace_logger(self, tmp_path: Path) -> None:
        """End-to-end with a real TraceLogger to confirm events are persisted."""
        harness_dir = _build_harness(tmp_path)
        db_path = tmp_path / "trace.db"
        logger = TraceLogger(db_path)
        with logger.session("harness-v1") as session_id:
            evolver = Evolver(trace_logger=logger, session_id=session_id)
            edits = evolver._propose_from_template(harness_dir, _make_failure("bogus-class"))

        assert edits == []
        events = list(logger.query_events(kind=GENERATION_ATTEMPT_KIND))
        assert len(events) == 1
        assert "bogus-class" in events[0].payload["error"]

        exhausted = list(logger.query_events(kind=GENERATION_EXHAUSTED_KIND))
        assert len(exhausted) == 1
        assert "bogus-class" in exhausted[0].payload["final_error"]

    def test_no_json_patch_emits_trace_events(self, tmp_path: Path) -> None:
        """JSON target with ``json_patch=None`` emits trace events."""
        harness_dir = _build_harness(tmp_path)
        (harness_dir / "manifest.json").write_text(
            json.dumps({"version": "0.1.0"}) + "\n", encoding="utf-8"
        )
        trace_logger = MagicMock()
        evolver = Evolver(trace_logger=trace_logger, session_id="sess-974")

        failure = _make_failure("context-overflow")

        from foundry_x.evolution.evolver import _PROPOSED_CLASS_EDIT_TEMPLATES

        original = _PROPOSED_CLASS_EDIT_TEMPLATES["context-overflow"]
        # Temporarily swap to a JSON target with no patch
        _PROPOSED_CLASS_EDIT_TEMPLATES["context-overflow"] = (
            "manifest.json",
            "test",
            [],
            None,
        )
        try:
            edits = evolver._propose_from_template(harness_dir, failure)
        finally:
            _PROPOSED_CLASS_EDIT_TEMPLATES["context-overflow"] = original

        assert edits == []
        kinds = [call.args[1] for call in trace_logger.record.call_args_list]
        assert GENERATION_ATTEMPT_KIND in kinds
        assert GENERATION_EXHAUSTED_KIND in kinds

    def test_corrupt_json_emits_trace_events(self, tmp_path: Path) -> None:
        """JSON merge-patch failure (corrupt fixture) emits trace events."""
        harness_dir = _build_harness(tmp_path)
        (harness_dir / "manifest.json").write_text("not valid json {{{\n", encoding="utf-8")
        trace_logger = MagicMock()
        evolver = Evolver(trace_logger=trace_logger, session_id="sess-974")

        from foundry_x.evolution.evolver import _PROPOSED_CLASS_EDIT_TEMPLATES

        original = _PROPOSED_CLASS_EDIT_TEMPLATES["wrong-tool"]
        _PROPOSED_CLASS_EDIT_TEMPLATES["wrong-tool"] = (
            "manifest.json",
            "test",
            [],
            {"new_key": "val"},
        )
        try:
            edits = evolver._propose_from_template(harness_dir, _make_failure("wrong-tool"))
        finally:
            _PROPOSED_CLASS_EDIT_TEMPLATES["wrong-tool"] = original

        assert edits == []
        kinds = [call.args[1] for call in trace_logger.record.call_args_list]
        assert GENERATION_ATTEMPT_KIND in kinds
        assert GENERATION_EXHAUSTED_KIND in kinds
        attempt_call = next(
            c for c in trace_logger.record.call_args_list if c.args[1] == GENERATION_ATTEMPT_KIND
        )
        assert "JSON merge patch failed" in attempt_call.args[2]["error"]

    def test_empty_diff_emits_trace_events(self, tmp_path: Path) -> None:
        """Template that produces an empty diff (no content change) emits events."""
        harness_dir = _build_harness(tmp_path)

        trace_logger = MagicMock()
        evolver = Evolver(trace_logger=trace_logger, session_id="sess-974")

        from foundry_x.evolution.evolver import _PROPOSED_CLASS_EDIT_TEMPLATES

        original = _PROPOSED_CLASS_EDIT_TEMPLATES["wrong-tool"]
        # Template with empty extra_lines: modified == original so difflib
        # produces an empty diff. Pre-seed the file with the trailing blank
        # line the template code adds (original.rstrip("\n") + "\n" + "\n").
        (harness_dir / "system_prompt.txt").write_text(
            "You are a helpful agent.\n\n", encoding="utf-8"
        )
        _PROPOSED_CLASS_EDIT_TEMPLATES["wrong-tool"] = (
            "system_prompt.txt",
            "test",
            [],
            None,
        )
        try:
            edits = evolver._propose_from_template(harness_dir, _make_failure("wrong-tool"))
        finally:
            _PROPOSED_CLASS_EDIT_TEMPLATES["wrong-tool"] = original

        assert edits == []
        kinds = [call.args[1] for call in trace_logger.record.call_args_list]
        assert GENERATION_ATTEMPT_KIND in kinds
        assert GENERATION_EXHAUSTED_KIND in kinds

    def test_guard_error_emits_trace_events(self, tmp_path: Path) -> None:
        """EvolverGuardError from validate_edit emits trace events."""
        harness_dir = _build_harness(tmp_path)

        trace_logger = MagicMock()
        evolver = Evolver(
            trace_logger=trace_logger,
            session_id="sess-974",
            max_diff_lines=1,
        )

        edits = evolver._propose_from_template(harness_dir, _make_failure("wrong-tool"))

        assert edits == []
        kinds = [call.args[1] for call in trace_logger.record.call_args_list]
        assert GENERATION_ATTEMPT_KIND in kinds
        assert GENERATION_EXHAUSTED_KIND in kinds
        attempt_call = next(
            c for c in trace_logger.record.call_args_list if c.args[1] == GENERATION_ATTEMPT_KIND
        )
        assert "failed validation" in attempt_call.args[2]["error"]


class TestNoTraceLoggerNoEvents:
    """When no trace_logger is configured, _propose_from_template still returns []."""

    def test_silent_return_without_logger(self, tmp_path: Path) -> None:
        harness_dir = _build_harness(tmp_path)
        evolver = Evolver()
        edits = evolver._propose_from_template(harness_dir, _make_failure("nonexistent"))
        assert edits == []


class TestLoopWiresTraceLogger:
    """run_evolution_step passes trace_logger into the default Evolver (issue #974 AC #3)."""

    def test_loop_passes_trace_logger_to_default_evolver(self, tmp_path: Path) -> None:
        """When no evolver is provided, the default one gets trace_logger + session_id.

        We verify the wiring by checking that ``proposed_edit`` events
        (emitted by ``_record_proposals``) land in the trace store. If
        ``trace_logger`` were not wired, no events would appear.
        """
        harness_dir = _build_harness(tmp_path)

        # Events that trigger a non-clean report → reaches Evolver
        base_ts = "2026-07-10T12:00:00+00:00"
        events = [
            TraceEvent(
                event_id="e1",
                session_id="sess-loop",
                timestamp=base_ts,
                kind="user_prompt",
                payload={"prompt": "hello"},
            ),
            TraceEvent(
                event_id="e2",
                session_id="sess-loop",
                timestamp="2026-07-10T12:00:01+00:00",
                kind="error",
                payload={"error": "oops"},
            ),
        ]

        db_path = tmp_path / "trace.db"
        logger = TraceLogger(db_path)

        result = run_evolution_step("sess-loop", events, harness_dir, trace_logger=logger)

        # The failure_class is not clean (error event present), and the
        # default Evolver had no model_adapter so it falls back to templates.
        assert result.failure_report.proposed_class != "clean"

        # If templates succeeded, proposed_edit events should be in the store.
        # If templates failed (unknown class), generation_attempt/exhausted events
        # should be there. Either way, trace_logger was wired.
        from foundry_x.evolution.evolver import PROPOSED_EDIT_KIND

        proposed_events = list(logger.query_events(kind=PROPOSED_EDIT_KIND))
        attempt_events = list(logger.query_events(kind=GENERATION_ATTEMPT_KIND))
        # At least one of these must be non-empty to prove trace_logger was wired.
        assert len(proposed_events) + len(attempt_events) > 0, (
            "Expected trace events proving trace_logger was wired into the Evolver"
        )


class TestKnownClassesRegression:
    """All 7 known failure classes still produce template edits (AC #4)."""

    @pytest.mark.parametrize(
        "failure_class",
        [
            "wrong-tool",
            "bad-prompt",
            "state-leak",
            "tool-error",
            "injection-attempt",
            "context-overflow",
            "unknown",
        ],
    )
    def test_template_still_produces_edit(self, failure_class: str, tmp_path: Path) -> None:
        """Each known class produces exactly one edit with no trace events."""
        harness_dir = _build_harness(tmp_path)
        trace_logger = MagicMock()
        evolver = Evolver(trace_logger=trace_logger, session_id="sess-974")

        edits = evolver._propose_from_template(harness_dir, _make_failure(failure_class))

        assert len(edits) == 1
        assert edits[0].target_file.startswith("harness/")

        # No failure trace events should be emitted on the success path
        kinds = [call.args[1] for call in trace_logger.record.call_args_list]
        assert GENERATION_ATTEMPT_KIND not in kinds, (
            f"Success path for {failure_class!r} should not emit generation_attempt"
        )
        assert GENERATION_EXHAUSTED_KIND not in kinds, (
            f"Success path for {failure_class!r} should not emit generation_exhausted"
        )
