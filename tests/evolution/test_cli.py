"""Tests for foundry-evolve CLI (issue #256)."""

from __future__ import annotations

from pathlib import Path

from foundry_x.evolution.cli import _build_parser, _infer_backend, main
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


class TestBuildParser:
    def test_required_args(self):
        parser = _build_parser()
        args = parser.parse_args(["--session-id", "abc", "--harness-dir", "/tmp/harness"])
        assert args.session_id == "abc"
        assert args.harness_dir == Path("/tmp/harness")
        assert args.verbose is False
        assert args.trace_db == "logs/traces.db"

    def test_verbose_flag(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--session-id", "abc", "--harness-dir", "/tmp/harness", "--verbose"]
        )
        assert args.verbose is True

    def test_custom_trace_db(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "--session-id",
                "abc",
                "--harness-dir",
                "/tmp/harness",
                "--trace-db",
                "/custom/path.db",
            ]
        )
        assert args.trace_db == "/custom/path.db"


class TestFoundryEvolveCLI:
    def test_unknown_session_returns_exit_2(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        TraceLogger(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            ["--session-id", "does-not-exist", "--trace-db", str(db), "--harness-dir", str(harness)]
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

        rc = main(["--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "No failure detected" in out

    def test_clean_session_includes_failure_report_summary(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        main(["--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)])

        out = capsys.readouterr().out
        assert "Failure Report" in out
        assert "clean" in out

    def test_failing_session_reports_failure_classification(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(["--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)])

        captured = capsys.readouterr()
        # Digester should produce a FailureReport with tool-error classification
        assert "Failure Report" in captured.out
        # Exit code 2 because Evolver is not implemented (message goes to stderr)
        assert rc == 2
        assert "not yet implemented" in captured.err

    def test_verbose_flag_shows_unified_diff(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--verbose",
            ]
        )

        # Exit code is 2 because Evolver is not yet implemented
        assert rc == 2
        captured = capsys.readouterr()
        assert "not yet implemented" in captured.err

    def test_exit_code_0_for_clean_session(self, tmp_path):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(["--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)])

        assert rc == 0

    def test_exit_code_2_for_evolver_not_implemented(self, tmp_path, capsys):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(["--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)])

        assert rc == 2
        err = capsys.readouterr().err
        assert "not yet implemented" in err

    def test_jsonl_backend(self, tmp_path, capsys):
        db = tmp_path / "traces.jsonl"
        logger = TraceLogger(db, backend="jsonl")
        with logger.session(harness_version="0.1.0", model_id="test-model") as sid:
            logger.record(sid, "outcome", {"status": "success", "reason": "final_answer"})

        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(["--session-id", sid, "--trace-db", str(db), "--harness-dir", str(harness)])

        assert rc == 0
        out = capsys.readouterr().out
        assert "No failure detected" in out
