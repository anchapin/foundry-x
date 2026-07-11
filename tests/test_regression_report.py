from __future__ import annotations

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.regression_report import (
    generate_regression_report,
    record_verdict,
)
from foundry_x.observability.cli import main as cli_main
from foundry_x.trace.logger import TraceLogger


def _section(report: str, heading: str) -> str:
    parts = report.split(f"## {heading}")
    if len(parts) < 2:
        return ""
    return parts[1].split("\n## ", 1)[0]


def _three_sessions(logger: TraceLogger) -> list[str]:
    sessions: list[str] = []
    for _ in range(3):
        with logger.session(harness_version="test-0.0") as sid:
            sessions.append(sid)
    return sessions


def test_regression_report_detects_regressions_and_new_passes(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    sid_a, sid_b, sid_c = _three_sessions(logger)

    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=False, failed_checks=["task-A"], passed_checks=["task-B"]),
    )
    record_verdict(logger, sid_c, CriticVerdict(approved=True, passed_checks=["task-A"]))

    report = generate_regression_report(logger)

    assert "## Regression Summary" in report
    assert "## Regressed Tasks" in report
    assert "## New Passes" in report

    regressed = _section(report, "Regressed Tasks")
    assert "task-A" in regressed
    assert sid_b in regressed
    assert sid_a in regressed

    new_passes = _section(report, "New Passes")
    assert "task-A" in new_passes
    assert sid_c in new_passes
    assert sid_b in new_passes


def test_regression_summary_counts(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    sid_a, sid_b, sid_c = _three_sessions(logger)

    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["t1"]))
    record_verdict(logger, sid_b, CriticVerdict(approved=False, failed_checks=["t1"]))
    record_verdict(logger, sid_c, CriticVerdict(approved=True, passed_checks=["t1"]))

    summary = _section(generate_regression_report(logger), "Regression Summary")
    assert "Total verdicts: 3" in summary
    assert "Approvals: 2" in summary
    assert "Rejections: 1" in summary


def test_no_regressions_when_nothing_previously_passed(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    sid_a, sid_b = _three_sessions(logger)[:2]

    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=True, passed_checks=["task-A", "task-B"]),
    )

    report = generate_regression_report(logger)
    assert "_None._" in _section(report, "Regressed Tasks")
    assert "_None._" in _section(report, "New Passes")


def test_record_verdict_roundtrips_through_logger(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    with logger.session(harness_version="test-0.0") as sid:
        record_verdict(
            logger,
            sid,
            CriticVerdict(
                approved=False,
                passed_checks=["p1"],
                failed_checks=["f1"],
                notes="boom",
            ),
        )
        events = logger.load_session(sid)

    verdict_events = [e for e in events if e.kind == "critic_verdict"]
    assert len(verdict_events) == 1
    payload = verdict_events[0].payload
    assert payload["approved"] is False
    assert payload["passed_checks"] == ["p1"]
    assert payload["failed_checks"] == ["f1"]
    assert payload["notes"] == "boom"


def test_since_filter_excludes_old_verdicts(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    sid_a, sid_b = _three_sessions(logger)[:2]

    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=False, failed_checks=["task-A"]),
    )

    full = generate_regression_report(logger)
    assert "Total verdicts: 2" in _section(full, "Regression Summary")
    assert "task-A" in _section(full, "Regressed Tasks")

    future = generate_regression_report(logger, since="9999-01-01T00:00:00+00:00")
    assert "Total verdicts: 0" in _section(future, "Regression Summary")
    assert "_None._" in _section(future, "Regressed Tasks")


def test_cli_regression_report_stdout(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    sid_a, sid_b = _three_sessions(logger)[:2]
    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=False, failed_checks=["task-A"]),
    )

    rc = cli_main(["regression-report", "--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "## Regressed Tasks" in captured.out
    assert "task-A" in captured.out


def test_cli_regression_report_out_file(tmp_path):
    db = tmp_path / "traces.db"
    out = tmp_path / "report.md"
    logger = TraceLogger(db)
    sid_a, sid_b = _three_sessions(logger)[:2]
    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=False, failed_checks=["task-A"]),
    )

    rc = cli_main(["regression-report", "--db", str(db), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "## Regressed Tasks" in out.read_text(encoding="utf-8")


def test_cli_regression_report_fail_on_regression_exits_nonzero_with_regressions(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    sid_a, sid_b = _three_sessions(logger)[:2]
    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=False, failed_checks=["task-A"]),
    )

    rc = cli_main(
        [
            "regression-report",
            "--db",
            str(db),
            "--fail-on-regression",
        ]
    )
    assert rc == 1


def test_cli_regression_report_fail_on_regression_exits_zero_when_clean(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    sid_a, sid_b = _three_sessions(logger)[:2]
    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=True, passed_checks=["task-A", "task-B"]),
    )

    rc = cli_main(
        [
            "regression-report",
            "--db",
            str(db),
            "--fail-on-regression",
        ]
    )
    assert rc == 0


def test_cli_regression_report_fail_on_regression_writes_out_before_exit(tmp_path):
    db = tmp_path / "traces.db"
    out = tmp_path / "report.md"
    logger = TraceLogger(db)
    sid_a, sid_b = _three_sessions(logger)[:2]
    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=False, failed_checks=["task-A"]),
    )

    rc = cli_main(
        [
            "regression-report",
            "--db",
            str(db),
            "--out",
            str(out),
            "--fail-on-regression",
        ]
    )
    assert rc == 1
    assert out.exists()
    body = out.read_text(encoding="utf-8")
    assert "## Regressed Tasks" in body
    assert "task-A" in body


def test_cli_regression_report_without_fail_flag_does_not_gate_exit(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    sid_a, sid_b = _three_sessions(logger)[:2]
    record_verdict(logger, sid_a, CriticVerdict(approved=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(approved=False, failed_checks=["task-A"]),
    )

    rc = cli_main(["regression-report", "--db", str(db)])
    assert rc == 0
