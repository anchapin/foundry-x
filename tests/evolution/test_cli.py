"""Tests for foundry-evolve CLI (issue #256, #799)."""

from __future__ import annotations

from pathlib import Path

from foundry_x.evolution.cli import _infer_backend, _run_loop, main
from foundry_x.trace.logger import TraceLogger
from tests._harness_fixture import install_load_check_prerequisites


def _write_minimal_harness(harness_dir: Path) -> None:
    """Write a minimal valid harness that passes load_check."""
    install_load_check_prerequisites(harness_dir)
    # Write a minimal skills/ directory (load_check requires it)
    (harness_dir / "skills").mkdir(exist_ok=True)
    # Ensure system_prompt.txt is non-empty (load_check requirement)
    (harness_dir / "system_prompt.txt").write_text("Test system prompt.\n", encoding="utf-8")


def _populate_failing_session(db_path: Path) -> str:
    """Plant a session with a tool_error event that the Digester will classify."""
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
        logger.record(sid, "task_received", {"prompt": "Fix the bug in auth.py"})
        logger.record(sid, "user_prompt", {"prompt": "Fix the bug in auth.py", "tool_count": 1})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(
            sid,
            "tool_result",
            {
                "name": "read_file",
                "error": "FileNotFoundError: auth.py not found",
                "traceback": "...",
            },
        )
        logger.record(sid, "outcome", {"status": "failed", "reason": "tool_error"})
    return sid


def _populate_clean_session(db_path: Path) -> str:
    """Plant a session with no failure events."""
    logger = TraceLogger(db_path)
    with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
        logger.record(sid, "task_received", {"prompt": "Do the thing"})
        logger.record(sid, "user_prompt", {"prompt": "Do the thing", "tool_count": 1})
        logger.record(sid, "tool_call", {"name": "read_file"})
        logger.record(sid, "tool_result", {"name": "read_file", "output": "file contents"})
        logger.record(sid, "outcome", {"status": "success", "reason": "final_answer", "steps": 1})
    return sid


class TestInferBackend:
    def test_sqlite_db(self):
        assert _infer_backend("logs/traces.db") == "sqlite"

    def test_jsonl_backend(self):
        assert _infer_backend("logs/traces.jsonl") == "jsonl"


class TestExportPrometheus:
    """Tests for --export-prometheus flag (issue #1364)."""

    def test_export_prometheus_emits_kpi_metrics(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--export-prometheus",
            ]
        )

        captured = capsys.readouterr()
        assert "foundryx_kpi_entry" in captured.out
        assert 'kpi="cycle_time_seconds"' in captured.out
        assert 'kpi="improvement_rate"' in captured.out
        assert 'kpi="regression_rate"' in captured.out
        assert rc == 1

    def test_export_prometheus_exit_code_unaffected(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc_with = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--export-prometheus",
            ]
        )

        db2 = tmp_path / "traces2.db"
        sid2 = _populate_failing_session(db2)
        harness2 = tmp_path / "harness2"
        harness2.mkdir()
        _write_minimal_harness(harness2)

        rc_without = main(
            [
                "evolve",
                "--session-id",
                sid2,
                "--trace-db",
                str(db2),
                "--harness-dir",
                str(harness2),
            ]
        )

        assert rc_with == rc_without == 1

    def test_export_prometheus_clean_session(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--export-prometheus",
            ]
        )

        captured = capsys.readouterr()
        assert "foundryx_kpi_entry" in captured.out
        assert 'kpi="cycle_time_seconds"' in captured.out
        assert 'kpi="improvement_rate"' in captured.out
        assert 'kpi="regression_rate"' in captured.out
        assert rc == 0

    def test_export_prometheus_via_run_loop(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id=sid,
            trace_db=str(db),
            harness_dir=harness,
            verbose=False,
            export_prometheus=True,
        )

        captured = capsys.readouterr()
        assert "foundryx_kpi_entry" in captured.out
        assert 'kpi="cycle_time_seconds"' in captured.out
        assert 'kpi="improvement_rate"' in captured.out
        assert 'kpi="regression_rate"' in captured.out
        assert exit_code == 1


class TestFoundryEvolveCLI:
    def test_unknown_session_returns_exit_2(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        TraceLogger(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--session-id",
                "does-not-exist",
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
            ]
        )

        assert rc == 2
        err = capsys.readouterr().err
        assert "No events found" in err

    def test_clean_session_returns_exit_0(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            ["evolve", "--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)]
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "No failure detected" in out

    def test_clean_session_includes_failure_report_summary(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        main(["evolve", "--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)])

        out = capsys.readouterr().out
        assert "Failure Report" in out
        assert "clean" in out

    def test_failing_session_reports_failure_classification(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            ["evolve", "--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)]
        )

        captured = capsys.readouterr()
        assert "Failure Report" in captured.out
        assert "tool-error" in captured.out
        assert "Proposed Edit" in captured.out
        assert "Critic Verdict" in captured.out
        assert rc == 1

    def test_verbose_flag_shows_unified_diff(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--verbose",
            ]
        )

        captured = capsys.readouterr()
        assert "--- a/" in captured.out
        assert "+++ b/" in captured.out
        assert "Proposed Edit" in captured.out
        assert "Critic Verdict" in captured.out
        assert rc == 1

    def test_exit_code_0_for_clean_session(self, tmp_path):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            ["evolve", "--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)]
        )

        assert rc == 0

    def test_failing_session_gets_critic_rejection(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            ["evolve", "--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)]
        )

        assert rc == 1
        captured = capsys.readouterr().out
        assert "Critic Verdict" in captured
        assert "REJECTED" in captured

    def test_jsonl_backend(self, tmp_path, capsys):
        db = tmp_path / "traces.jsonl"
        logger = TraceLogger(db, backend="jsonl")
        with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
            logger.record(sid, "outcome", {"status": "success", "reason": "final_answer"})

        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            ["evolve", "--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)]
        )

        assert rc == 0
        out = capsys.readouterr().out
        assert "No failure detected" in out


class TestRunLoopIntegration:
    """Integration tests for _run_loop (issue #799).

    These tests exercise the full pipeline (TraceLogger -> Digester -> Evolver -> Critic)
    by calling _run_loop directly, testing _infer_backend, _render_failure_report,
    and _render_critic_verdict in a live session context.
    """

    def test_clean_session_via_run_loop_exits_0(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id=sid,
            trace_db=str(db),
            harness_dir=harness,
            verbose=False,
        )

        assert exit_code == 0
        assert _report is not None
        assert _report.proposed_class == "clean"
        assert _edit is None
        assert _verdict is None
        out = capsys.readouterr().out
        assert "Failure Report" in out
        assert "clean" in out

    def test_failing_session_via_run_loop_exits_1(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id=sid,
            trace_db=str(db),
            harness_dir=harness,
            verbose=False,
        )

        assert exit_code == 1
        assert _report is not None
        assert _report.proposed_class != "clean"
        assert _edit is not None
        assert _verdict is not None
        assert _verdict.verdict is False
        out = capsys.readouterr().out
        assert "Failure Report" in out
        assert "tool-error" in out
        assert "Proposed Edit" in out
        assert "Critic Verdict" in out
        assert "REJECTED" in out

    def test_unknown_session_via_run_loop_exits_2(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        TraceLogger(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id="does-not-exist",
            trace_db=str(db),
            harness_dir=harness,
            verbose=False,
        )

        assert exit_code == 2
        assert _report is None
        assert _edit is None
        assert _verdict is None
        err = capsys.readouterr().err
        assert "No events found" in err

    def test_verbose_flag_via_run_loop_shows_diff(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id=sid,
            trace_db=str(db),
            harness_dir=harness,
            verbose=True,
        )

        assert exit_code == 1
        captured = capsys.readouterr()
        assert "--- a/" in captured.out
        assert "+++ b/" in captured.out
        assert "Proposed Edit" in captured.out
        assert "Critic Verdict" in captured.out

    def test_run_loop_infers_sqlite_backend(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id=sid,
            trace_db=str(db),
            harness_dir=harness,
            verbose=False,
        )

        assert exit_code == 0
        assert _report is not None
        assert _report.proposed_class == "clean"

    def test_run_loop_infers_jsonl_backend(self, tmp_path, capsys):
        db = tmp_path / "traces.jsonl"
        logger = TraceLogger(db, backend="jsonl")
        with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
            logger.record(sid, "task_received", {"prompt": "Do the thing"})
            logger.record(sid, "user_prompt", {"prompt": "Do the thing", "tool_count": 1})
            logger.record(sid, "tool_call", {"name": "read_file"})
            logger.record(sid, "tool_result", {"name": "read_file", "output": "file contents"})
            logger.record(
                sid, "outcome", {"status": "success", "reason": "final_answer", "steps": 1}
            )

        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id=sid,
            trace_db=str(db),
            harness_dir=harness,
            verbose=False,
        )

        assert exit_code == 0
        assert _report is not None
        assert _report.proposed_class == "clean"


class TestCycleTime:
    """Tests for cycle time computation and display (issue #1361)."""

    def test_failing_session_includes_cycle_time_in_output(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
            ]
        )

        captured = capsys.readouterr()
        assert "Cycle time:" in captured.out
        assert "s" in captured.out
        assert rc == 1

    def test_failing_session_cycle_time_is_positive(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
            ]
        )

        captured = capsys.readouterr()
        import re

        match = re.search(r"Cycle time:\s+([\d.]+)s", captured.out)
        assert match is not None, "Cycle time line not found in output"
        cycle_time = float(match.group(1))
        assert cycle_time >= 0, f"Cycle time should be non-negative, got {cycle_time}"

    def test_clean_session_does_not_include_cycle_time(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
            ]
        )

        captured = capsys.readouterr()
        assert "Cycle time:" not in captured.out
        assert rc == 0

    def test_render_critic_verdict_with_cycle_time(self):
        from foundry_x.evolution.cli import _render_critic_verdict
        from foundry_x.evolution.critic import CriticVerdict

        verdict = CriticVerdict(
            verdict=True,
            passed_checks=["check1"],
            failed_checks=[],
            notes="Test notes",
        )
        output = _render_critic_verdict(verdict, cycle_time_seconds=42.5)
        assert "Cycle time:" in output
        assert "42.5s" in output

    def test_render_critic_verdict_without_cycle_time(self):
        from foundry_x.evolution.cli import _render_critic_verdict
        from foundry_x.evolution.critic import CriticVerdict

        verdict = CriticVerdict(
            verdict=True,
            passed_checks=["check1"],
            failed_checks=[],
            notes="Test notes",
        )
        output = _render_critic_verdict(verdict, cycle_time_seconds=None)
        assert "Cycle time:" not in output
