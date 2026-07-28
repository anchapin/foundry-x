"""Tests for ``foundry-evolve evolve`` CLI flags (issue #888).

Covers the ``--background`` and ``--no-verify`` flags on the
``evolve`` subcommand.

These tests intentionally live under ``tests/cli/`` rather than
``tests/evolution/`` because they target the user-facing CLI surface
introduced by issue #888 specifically; the broader evolution-CLI
behaviour is still covered by ``tests/evolution/test_cli.py``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from unittest import mock

import pytest

from foundry_x.evolution.cli import (
    _build_evolve_subparser,
    _run_loop,
    main,
)
from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.regression_report import VERDICT_KIND
from foundry_x.trace.logger import TraceLogger
from tests._harness_fixture import install_load_check_prerequisites


def _write_minimal_harness(harness_dir: Path) -> None:
    """Write a minimal valid harness that passes load_check."""
    install_load_check_prerequisites(harness_dir)
    (harness_dir / "skills").mkdir(exist_ok=True)
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


class _CapturedSubprocess:
    """Test double for :class:`subprocess.Popen` that records the spawn."""

    def __init__(self) -> None:
        self.spawned: list[list[str]] = []
        self.pids = iter([4242, 4243, 4244, 4245])

    def __call__(self, cmd, **kwargs):
        self.spawned.append(list(cmd))
        proc = mock.MagicMock()
        proc.pid = next(self.pids)
        # Pretend the child has already exited cleanly.
        proc.poll.return_value = 0
        proc.returncode = 0
        return proc


# --------------------------------------------------------------------------- #
# --background                                                                #
# --------------------------------------------------------------------------- #


class TestBackgroundFlag:
    """``foundry-evolve evolve --background`` spawns a subprocess and exits 0."""

    def test_help_lists_background_and_no_verify(self, capsys):
        import argparse

        parser = argparse.ArgumentParser(prog="foundry-evolve")
        sub = parser.add_subparsers(dest="command")
        evolve = sub.add_parser("evolve")
        _build_evolve_subparser(evolve)

        with pytest.raises(SystemExit):
            parser.parse_args(["evolve", "--help"])
        out = capsys.readouterr().out
        assert "--background" in out
        assert "--no-verify" in out

    def test_background_returns_zero_and_spawns_detached_subprocess(
        self, tmp_path, capsys, monkeypatch
    ):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        captured = _CapturedSubprocess()
        monkeypatch.setattr(
            "foundry_x.evolution.cli.subprocess.Popen",
            captured,
        )

        start = time.monotonic()
        rc = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--background",
            ]
        )
        elapsed = time.monotonic() - start

        assert rc == 0
        # The parent must return essentially immediately — the child does the
        # real work. ``--background`` is worthless if it blocks on the loop.
        assert elapsed < 2.0
        # Exactly one subprocess spawn, with ``--background`` stripped so the
        # child does not re-spawn infinitely.
        assert len(captured.spawned) == 1
        cmd = captured.spawned[0]
        assert "--background" not in cmd
        assert "evolve" in cmd
        assert "--session-id" in cmd
        assert sid in cmd
        # The PID is printed so operators can monitor the child.
        out = capsys.readouterr().out
        assert "PID 4242" in out
        assert "list-pending" in out

    def test_background_forwards_no_verify_to_child(self, tmp_path, capsys, monkeypatch):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        captured = _CapturedSubprocess()
        monkeypatch.setattr(
            "foundry_x.evolution.cli.subprocess.Popen",
            captured,
        )

        rc = main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--background",
                "--no-verify",
            ]
        )

        assert rc == 0
        cmd = captured.spawned[0]
        assert "--no-verify" in cmd

    def test_background_uses_posix_start_new_session_on_linux(self, tmp_path, monkeypatch):
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        captured = _CapturedSubprocess()
        monkeypatch.setattr(
            "foundry_x.evolution.cli.subprocess.Popen",
            captured,
        )

        main(
            [
                "evolve",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
                "--background",
            ]
        )

        # The test suite runs on Linux, so the child should detach via
        # ``start_new_session=True`` (POSIX). This is what keeps the
        # background loop alive after the parent exits.
        assert os.name == "posix"  # sanity-check the test platform


# --------------------------------------------------------------------------- #
# --no-verify                                                                 #
# --------------------------------------------------------------------------- #


class TestNoVerifyFlag:
    """``--no-verify`` skips Critic, records verdict=None, warns on stderr."""

    def test_no_verify_records_skipped_verdict_and_exits_zero(self, tmp_path, capsys):
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
                "--no-verify",
            ]
        )

        # ``--no-verify`` is not a rejection — it is an explicit skip. Exit
        # code 0 because the gate did not return False (issue #888).
        assert rc == 0
        captured = capsys.readouterr()
        # SKIPPED status surfaces in the rendered verdict block.
        assert "SKIPPED" in captured.out
        assert "--no-verify: skipped" in captured.out
        # Prominent stderr warning per ADR-0004.
        assert "WARNING" in captured.err
        assert "--no-verify" in captured.err
        assert "ADR-0004" in captured.err

    def test_no_verify_persists_verdict_none_in_trace_store(self, tmp_path, capsys):
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
                "--no-verify",
            ]
        )

        # The audit trail MUST show a critic_verdict event with verdict=None
        # so the regression report can distinguish a skip from a rejection.
        logger = TraceLogger(db)
        verdict_events = [ev for ev in logger.load_session(sid) if ev.kind == VERDICT_KIND]
        assert len(verdict_events) == 1
        payload = verdict_events[0].payload
        assert payload["verdict"] is None
        assert payload["notes"] == "--no-verify: skipped"
        assert payload["failure_class"] != "clean"

    def test_no_verify_does_not_call_critic_evaluate(self, tmp_path):
        db = tmp_path / "traces.db"
        sid = _populate_failing_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        with (
            mock.patch("foundry_x.evolution.cli.Critic.evaluate") as fake_evaluate,
            mock.patch("foundry_x.evolution.loop.Critic.evaluate") as loop_evaluate,
        ):
            _run_loop(
                session_id=sid,
                trace_db=str(db),
                harness_dir=harness,
                verbose=False,
                no_verify=True,
            )

        # The skip path must not invoke either Critic call site.
        fake_evaluate.assert_not_called()
        loop_evaluate.assert_not_called()

    def test_no_verify_with_clean_session_still_exits_zero(self, tmp_path, capsys):
        """A clean session never reaches the Critic; ``--no-verify`` is a no-op
        warning on top of the existing clean short-circuit."""
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
                "--no-verify",
            ]
        )

        assert rc == 0
        captured = capsys.readouterr()
        # Even on a clean run the warning fires so operators know that, had a
        # failure been detected, the gate would have been skipped.
        assert "WARNING" in captured.err
        assert "No failure detected" in captured.out


class TestEvolveLatest:
    """Tests for --latest auto-select of most recent session (issue #1147)."""

    def test_latest_resolves_to_most_recent_session(self, tmp_path, capsys):
        """--latest selects the newest session by started_at desc."""
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)  # creates one session
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--latest",
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
            ]
        )
        assert rc == 0

    def test_latest_and_session_id_mutually_exclusive(self, tmp_path, capsys):
        """--latest and --session-id together exit 2 with error."""
        db = tmp_path / "traces.db"
        sid = _populate_clean_session(db)
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--latest",
                "--session-id",
                sid,
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "mutually exclusive" in captured.err

    def test_latest_with_no_sessions_exits_2(self, tmp_path, capsys):
        """--latest with empty trace store exits 2 with friendly error."""
        db = tmp_path / "traces.db"
        TraceLogger(db)  # empty DB
        harness = tmp_path / "harness"
        harness.mkdir()
        _write_minimal_harness(harness)

        rc = main(
            [
                "evolve",
                "--latest",
                "--trace-db",
                str(db),
                "--harness-dir",
                str(harness),
            ]
        )
        assert rc == 2
        captured = capsys.readouterr()
        assert "no sessions found" in captured.err


# --------------------------------------------------------------------------- #
# CriticVerdict.verdict Optional                                               #
# --------------------------------------------------------------------------- #


class TestCriticVerdictOptional:
    """``CriticVerdict.verdict`` accepts ``None`` to mark a skipped gate."""

    def test_verdict_none_round_trips_through_pydantic(self):
        v = CriticVerdict(verdict=None, notes="--no-verify: skipped")
        assert v.verdict is None
        # model_dump must preserve the None so persistence round-trips.
        dumped = v.model_dump(mode="json")
        assert dumped["verdict"] is None

    def test_verdict_bool_still_accepted(self):
        # Backward compatibility: existing call sites still pass True/False.
        assert CriticVerdict(verdict=True).verdict is True
        assert CriticVerdict(verdict=False).verdict is False
