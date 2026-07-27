"""Tests for cross-session failure-pattern accumulation (ADR-0030, issue #1038).

Covers:
- ``context_hash_bucket`` determinism and stability
- ``FailurePatternStore`` persistence, record, find_pattern, link_edit, history
- ``pattern_confidence_label`` tier mapping
- Evolver structural fix preference for recurring patterns
- Evolution loop wiring (record + query + annotate)
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from foundry_x.evolution.digester import FailureReport, context_hash_bucket
from foundry_x.evolution.evolver import Evolver
from foundry_x.evolution.loop import run_evolution_step
from foundry_x.evolution.store import (
    PATTERN_MIN_SESSIONS,
    CrossSessionPattern,
    FailurePatternEntry,
    FailurePatternStore,
    pattern_confidence_label,
)
from foundry_x.trace.logger import TraceEvent

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_BASE_TS = datetime(2026, 7, 10, 12, 0, 0, tzinfo=UTC)


def _make_failure(
    proposed_class: str = "wrong-tool",
    session_id: str = "sess-1",
    *,
    kind: str = "tool_error",
    tool_name: str = "bash",
    error_type: str = "",
    signal: str = "kind:tool_error",
) -> FailureReport:
    payload: dict = {"name": tool_name}
    if error_type:
        payload["error_type"] = error_type
    return FailureReport(
        session_id=session_id,
        summary=f"{proposed_class} failure",
        failed_steps=[
            {
                "index": 0,
                "event_id": "evt-1",
                "kind": kind,
                "timestamp": _BASE_TS.isoformat(),
                "signal": signal,
                "payload": payload,
            }
        ],
        suspected_causes=["test cause"],
        proposed_class=proposed_class,
    )


# ---------------------------------------------------------------------------
# context_hash_bucket tests
# ---------------------------------------------------------------------------


class TestContextHashBucket:
    def test_deterministic_for_same_input(self) -> None:
        f1 = _make_failure()
        f2 = _make_failure()
        assert context_hash_bucket(f1) == context_hash_bucket(f2)

    def test_different_class_produces_different_bucket(self) -> None:
        f1 = _make_failure(proposed_class="wrong-tool")
        f2 = _make_failure(proposed_class="state-leak")
        assert context_hash_bucket(f1) != context_hash_bucket(f2)

    def test_different_tool_name_produces_different_bucket(self) -> None:
        f1 = _make_failure(tool_name="bash")
        f2 = _make_failure(tool_name="git")
        assert context_hash_bucket(f1) != context_hash_bucket(f2)

    def test_ignores_session_id_and_summary(self) -> None:
        f1 = _make_failure(
            session_id="sess-A",
        )
        f1.summary = "completely different text"
        f2 = _make_failure(session_id="sess-B")
        f2.summary = "another summary"
        assert context_hash_bucket(f1) == context_hash_bucket(f2)

    def test_returns_64_char_hex(self) -> None:
        bucket = context_hash_bucket(_make_failure())
        assert len(bucket) == 64
        assert all(c in "0123456789abcdef" for c in bucket)

    def test_empty_failed_steps_still_hashes_class(self) -> None:
        f = FailureReport(
            session_id="s",
            summary="clean",
            proposed_class="tool-error",
        )
        bucket = context_hash_bucket(f)
        assert len(bucket) == 64

    def test_whitespace_normalization(self) -> None:
        """Whitespace and case differences should not affect the bucket."""
        f1 = _make_failure(signal="kind:tool_error")
        f2 = _make_failure(signal="  KIND:TOOL_ERROR  ")
        assert context_hash_bucket(f1) == context_hash_bucket(f2)


# ---------------------------------------------------------------------------
# FailurePatternStore tests
# ---------------------------------------------------------------------------


class TestFailurePatternStore:
    def test_persists_across_reopen(self, tmp_path: Path) -> None:
        """The store survives process restart (close + reopen)."""
        db_path = tmp_path / "patterns.db"
        store = FailurePatternStore(db_path)
        failure = _make_failure()
        bucket = context_hash_bucket(failure)
        store.record(
            proposed_class=failure.proposed_class,
            session_id=failure.session_id,
            timestamp=_BASE_TS.isoformat(),
            context_hash=bucket,
        )
        store.close()

        store2 = FailurePatternStore(db_path)
        history = store2.get_pattern_history(failure.proposed_class, bucket)
        assert len(history) == 1
        assert history[0].proposed_class == "wrong-tool"
        store2.close()

    def test_record_and_find_pattern(self, tmp_path: Path) -> None:
        """Recording across N sessions surfaces a pattern at the threshold."""
        store = FailurePatternStore(tmp_path / "patterns.db")
        failure = _make_failure()
        bucket = context_hash_bucket(failure)

        for i in range(PATTERN_MIN_SESSIONS):
            store.record(
                proposed_class=failure.proposed_class,
                session_id=f"sess-{i}",
                timestamp=(_BASE_TS + timedelta(days=i)).isoformat(),
                context_hash=bucket,
            )

        pattern = store.find_pattern(failure.proposed_class, bucket)
        assert pattern is not None
        assert pattern.session_count >= PATTERN_MIN_SESSIONS
        assert len(pattern.session_ids) >= PATTERN_MIN_SESSIONS
        store.close()

    def test_no_pattern_below_threshold(self, tmp_path: Path) -> None:
        store = FailurePatternStore(tmp_path / "patterns.db")
        failure = _make_failure()
        bucket = context_hash_bucket(failure)

        for i in range(PATTERN_MIN_SESSIONS - 1):
            store.record(
                proposed_class=failure.proposed_class,
                session_id=f"sess-{i}",
                timestamp=_BASE_TS.isoformat(),
                context_hash=bucket,
            )

        pattern = store.find_pattern(failure.proposed_class, bucket)
        assert pattern is None
        store.close()

    def test_find_pattern_custom_min_sessions(self, tmp_path: Path) -> None:
        store = FailurePatternStore(tmp_path / "patterns.db")
        failure = _make_failure()
        bucket = context_hash_bucket(failure)

        for i in range(2):
            store.record(
                proposed_class=failure.proposed_class,
                session_id=f"sess-{i}",
                timestamp=_BASE_TS.isoformat(),
                context_hash=bucket,
            )

        pattern = store.find_pattern(failure.proposed_class, bucket, min_sessions=2)
        assert pattern is not None
        assert pattern.session_count == 2
        store.close()

    def test_link_edit(self, tmp_path: Path) -> None:
        store = FailurePatternStore(tmp_path / "patterns.db")
        failure = _make_failure()
        bucket = context_hash_bucket(failure)
        row_id = store.record(
            proposed_class=failure.proposed_class,
            session_id="sess-1",
            timestamp=_BASE_TS.isoformat(),
            context_hash=bucket,
        )
        store.link_edit(row_id, edit_id="edit-abc", resolution="proposed")
        history = store.get_pattern_history(failure.proposed_class, bucket)
        assert history[0].edit_id == "edit-abc"
        assert history[0].resolution == "proposed"
        store.close()

    def test_get_pattern_history_ordering(self, tmp_path: Path) -> None:
        store = FailurePatternStore(tmp_path / "patterns.db")
        failure = _make_failure()
        bucket = context_hash_bucket(failure)

        for i in range(5):
            store.record(
                proposed_class=failure.proposed_class,
                session_id=f"sess-{i}",
                timestamp=(_BASE_TS + timedelta(seconds=i)).isoformat(),
                context_hash=bucket,
            )

        history = store.get_pattern_history(failure.proposed_class, bucket, limit=3)
        assert len(history) == 3
        store.close()

    def test_age_cutoff_filters_old_rows(self, tmp_path: Path) -> None:
        store = FailurePatternStore(tmp_path / "patterns.db")
        failure = _make_failure()
        bucket = context_hash_bucket(failure)

        old_ts = (datetime.now(UTC) - timedelta(days=60)).isoformat()
        for i in range(PATTERN_MIN_SESSIONS):
            store.record(
                proposed_class=failure.proposed_class,
                session_id=f"old-sess-{i}",
                timestamp=old_ts,
                context_hash=bucket,
            )

        pattern = store.find_pattern(failure.proposed_class, bucket)
        assert pattern is None
        store.close()

    def test_wal_mode_enabled(self, tmp_path: Path) -> None:
        """The store uses WAL journal mode like ProposedEditStore."""
        store = FailurePatternStore(tmp_path / "patterns.db")
        mode = store._conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"
        store.close()


# ---------------------------------------------------------------------------
# pattern_confidence_label tests
# ---------------------------------------------------------------------------


class TestPatternConfidenceLabel:
    @pytest.mark.parametrize(
        ("count", "expected"),
        [
            (0, "isolated"),
            (1, "isolated"),
            (2, "emerging"),
            (3, "recurring"),
            (4, "recurring"),
            (5, "recurring"),
            (6, "chronic"),
            (10, "chronic"),
        ],
    )
    def test_label_mapping(self, count: int, expected: str) -> None:
        assert pattern_confidence_label(count) == expected


# ---------------------------------------------------------------------------
# Evolver structural-fix-preference tests
# ---------------------------------------------------------------------------


def _setup_harness(tmp_path: Path) -> Path:
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("original prompt\n", encoding="utf-8")
    hooks_dir = harness_dir / "hooks"
    hooks_dir.mkdir()
    return harness_dir


class TestEvolverStructuralFix:
    def test_isolated_failure_targets_prompt(self, tmp_path: Path) -> None:
        """Isolated failure (seen_across_n_sessions=0) uses standard template."""
        harness_dir = _setup_harness(tmp_path)
        evolver = Evolver()
        failure = _make_failure()
        failure.seen_across_n_sessions = 0

        edits = evolver.propose(harness_dir=harness_dir, failure=failure)
        assert len(edits) == 1
        assert edits[0].target_file == "harness/system_prompt.txt"

    def test_recurring_failure_targets_hook(self, tmp_path: Path) -> None:
        """Recurring failure (>= PATTERN_MIN_SESSIONS) uses structural template."""
        harness_dir = _setup_harness(tmp_path)
        evolver = Evolver()
        failure = _make_failure(proposed_class="wrong-tool")
        failure.seen_across_n_sessions = PATTERN_MIN_SESSIONS

        edits = evolver.propose(harness_dir=harness_dir, failure=failure)
        assert len(edits) == 1
        assert edits[0].target_file.startswith("harness/hooks/")

    def test_recurring_failure_rationale_has_evidence(self, tmp_path: Path) -> None:
        """The structural edit rationale includes cross-session evidence."""
        harness_dir = _setup_harness(tmp_path)
        evolver = Evolver()
        failure = _make_failure(proposed_class="wrong-tool")
        failure.seen_across_n_sessions = 5

        edits = evolver.propose(harness_dir=harness_dir, failure=failure)
        assert len(edits) == 1
        assert "[cross-session pattern: 5 sessions]" in edits[0].rationale

    def test_structural_edit_creates_new_file_diff(self, tmp_path: Path) -> None:
        """Structural template for a non-existent hook creates a new-file diff."""
        harness_dir = _setup_harness(tmp_path)
        evolver = Evolver()
        failure = _make_failure(proposed_class="wrong-tool")
        failure.seen_across_n_sessions = PATTERN_MIN_SESSIONS

        edits = evolver.propose(harness_dir=harness_dir, failure=failure)
        assert len(edits) == 1
        assert "--- a/harness/hooks/" in edits[0].unified_diff
        assert "+++ b/harness/hooks/" in edits[0].unified_diff

    def test_structural_edit_each_failure_class(self, tmp_path: Path) -> None:
        """Every failure class has a structural template."""
        from foundry_x.evolution.evolver import _STRUCTURAL_EDIT_TEMPLATES

        expected_classes = {
            "wrong-tool",
            "bad-prompt",
            "state-leak",
            "tool-error",
            "injection-attempt",
            "context-overflow",
            "unknown",
        }
        assert set(_STRUCTURAL_EDIT_TEMPLATES.keys()) == expected_classes

    def test_llm_prompt_includes_pattern_section(self, tmp_path: Path) -> None:
        """_build_llm_prompt includes CROSS-SESSION PATTERN section for recurring."""
        evolver = Evolver()
        failure = _make_failure()
        failure.seen_across_n_sessions = PATTERN_MIN_SESSIONS

        prompt = evolver._build_llm_prompt(failure)
        assert "CROSS-SESSION PATTERN DETECTED" in prompt
        assert str(PATTERN_MIN_SESSIONS) in prompt

    def test_llm_prompt_omits_pattern_section_for_isolated(self, tmp_path: Path) -> None:
        """_build_llm_prompt omits CROSS-SESSION section for isolated failures."""
        evolver = Evolver()
        failure = _make_failure()
        failure.seen_across_n_sessions = 0

        prompt = evolver._build_llm_prompt(failure)
        assert "CROSS-SESSION PATTERN DETECTED" not in prompt


# ---------------------------------------------------------------------------
# Evolution loop wiring tests
# ---------------------------------------------------------------------------


class TestEvolutionLoopWiring:
    def _make_events(self) -> list[TraceEvent]:
        return [
            TraceEvent(
                event_id="evt-1",
                session_id="sess-test",
                timestamp=_BASE_TS.isoformat(),
                kind="tool_error",
                payload={"error": "no such tool: nonexistent"},
            ),
        ]

    def test_loop_records_and_annotates_pattern(self, tmp_path: Path) -> None:
        """The loop records failures to the store and annotates the report."""
        harness_dir = _setup_harness(tmp_path)
        store = FailurePatternStore(tmp_path / "patterns.db")

        for i in range(PATTERN_MIN_SESSIONS):
            run_evolution_step(
                session_id=f"sess-{i}",
                events=self._make_events(),
                harness_dir=harness_dir,
                no_verify=True,
                failure_pattern_store=store,
            )

        history = store.get_pattern_history(
            "wrong-tool",
            context_hash_bucket(
                FailureReport(
                    session_id="x",
                    summary="",
                    failed_steps=[
                        {"kind": "tool_error", "payload": {"name": ""}, "signal": "kind:tool_error"}
                    ],
                    proposed_class="wrong-tool",
                )
            ),
        )
        assert len(history) >= PATTERN_MIN_SESSIONS
        store.close()

    def test_loop_annotates_seen_across_n_sessions(self, tmp_path: Path) -> None:
        """After enough sessions, the failure report carries the session count."""
        harness_dir = _setup_harness(tmp_path)
        store = FailurePatternStore(tmp_path / "patterns.db")

        last_result = None
        for i in range(PATTERN_MIN_SESSIONS):
            last_result = run_evolution_step(
                session_id=f"sess-{i}",
                events=self._make_events(),
                harness_dir=harness_dir,
                no_verify=True,
                failure_pattern_store=store,
            )

        assert last_result is not None
        assert last_result.failure_report.seen_across_n_sessions >= PATTERN_MIN_SESSIONS
        store.close()

    def test_loop_without_store_works_unmodified(self, tmp_path: Path) -> None:
        """When no store is provided, behavior is unchanged."""
        harness_dir = _setup_harness(tmp_path)
        result = run_evolution_step(
            session_id="sess-1",
            events=self._make_events(),
            harness_dir=harness_dir,
            no_verify=True,
        )
        assert result.failure_report.seen_across_n_sessions == 0
        assert len(result.proposed_edits) >= 1

    def test_recurring_loop_targets_hook(self, tmp_path: Path) -> None:
        """After threshold sessions, the Evolver targets a hook file."""
        harness_dir = _setup_harness(tmp_path)
        store = FailurePatternStore(tmp_path / "patterns.db")

        last_result = None
        for i in range(PATTERN_MIN_SESSIONS + 1):
            last_result = run_evolution_step(
                session_id=f"sess-{i}",
                events=self._make_events(),
                harness_dir=harness_dir,
                no_verify=True,
                failure_pattern_store=store,
            )

        assert last_result is not None
        assert last_result.proposed_edits
        assert last_result.proposed_edits[0].target_file.startswith("harness/hooks/")
        store.close()


# ---------------------------------------------------------------------------
# Model serialization tests
# ---------------------------------------------------------------------------


class TestModels:
    def test_cross_session_pattern_serialization(self) -> None:
        p = CrossSessionPattern(
            proposed_class="wrong-tool",
            context_hash="abc123",
            session_count=3,
            latest_timestamp="2026-07-10T12:00:00+00:00",
            session_ids=["s1", "s2", "s3"],
        )
        d = p.model_dump()
        assert d["session_count"] == 3
        assert d["session_ids"] == ["s1", "s2", "s3"]

    def test_failure_pattern_entry_serialization(self) -> None:
        e = FailurePatternEntry(
            id=1,
            proposed_class="tool-error",
            session_id="sess-1",
            timestamp="2026-07-10T12:00:00+00:00",
            context_hash="hash",
            resolution="pending",
            edit_id="",
            created_at="2026-07-10T12:00:01+00:00",
        )
        assert e.id == 1
        assert e.resolution == "pending"

    def test_failure_report_default_seen_across_n_sessions(self) -> None:
        """New FailureReport defaults seen_across_n_sessions to 0."""
        f = FailureReport(session_id="s", summary="test", proposed_class="clean")
        assert f.seen_across_n_sessions == 0
