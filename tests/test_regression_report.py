from __future__ import annotations

import json

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.regression_report import (
    RegressionAnalysis,
    analyze_regressions,
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

    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"], passed_checks=["task-B"]),
    )
    record_verdict(logger, sid_c, CriticVerdict(verdict=True, passed_checks=["task-A"]))

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

    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["t1"]))
    record_verdict(logger, sid_b, CriticVerdict(verdict=False, failed_checks=["t1"]))
    record_verdict(logger, sid_c, CriticVerdict(verdict=True, passed_checks=["t1"]))

    summary = _section(generate_regression_report(logger), "Regression Summary")
    assert "Total verdicts: 3" in summary
    assert "Approvals: 2" in summary
    assert "Rejections: 1" in summary


def test_no_regressions_when_nothing_previously_passed(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    sid_a, sid_b = _three_sessions(logger)[:2]

    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=True, passed_checks=["task-A", "task-B"]),
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
                verdict=False,
                passed_checks=["p1"],
                failed_checks=["f1"],
                notes="boom",
            ),
        )
        events = logger.load_session(sid)

    verdict_events = [e for e in events if e.kind == "critic_verdict"]
    assert len(verdict_events) == 1
    payload = verdict_events[0].payload
    assert payload["verdict"] is False
    assert payload["passed_checks"] == ["p1"]
    assert payload["failed_checks"] == ["f1"]
    assert payload["notes"] == "boom"


def test_since_filter_excludes_old_verdicts(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    sid_a, sid_b = _three_sessions(logger)[:2]

    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"]),
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
    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"]),
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
    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"]),
    )

    rc = cli_main(["regression-report", "--db", str(db), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    assert "## Regressed Tasks" in out.read_text(encoding="utf-8")


def test_cli_regression_report_fail_on_regression_exits_nonzero_with_regressions(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    sid_a, sid_b = _three_sessions(logger)[:2]
    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"]),
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
    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=True, passed_checks=["task-A", "task-B"]),
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
    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"]),
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
    record_verdict(logger, sid_a, CriticVerdict(verdict=True, passed_checks=["task-A"]))
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"]),
    )

    rc = cli_main(["regression-report", "--db", str(db)])
    assert rc == 0


def _plant_mixed_sessions(logger: TraceLogger) -> tuple[str, str, str]:
    """Plant 3 sessions with two distinct tasks (issue #182 fixture).

    Returns ``(sid_a, sid_b, sid_c)``.

    - session A: ``task-A`` passes, ``task-B`` fails (baseline)
    - session B: ``task-A`` regresses (now failing), ``task-B`` now passes
    - session C: ``task-A`` recovers (new pass), ``task-B`` regresses again
    """
    sid_a, sid_b, sid_c = _three_sessions(logger)
    record_verdict(
        logger,
        sid_a,
        CriticVerdict(verdict=False, passed_checks=["task-A"], failed_checks=["task-B"]),
    )
    record_verdict(
        logger,
        sid_b,
        CriticVerdict(verdict=False, failed_checks=["task-A"], passed_checks=["task-B"]),
    )
    record_verdict(
        logger,
        sid_c,
        CriticVerdict(verdict=False, passed_checks=["task-A"], failed_checks=["task-B"]),
    )
    return sid_a, sid_b, sid_c


def test_analyze_regressions_filters_by_task(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    sid_a, sid_b, sid_c = _plant_mixed_sessions(logger)

    full = analyze_regressions(logger)
    assert {r.task for r in full.regressions} == {"task-A", "task-B"}
    assert {p.task for p in full.new_passes} == {"task-A", "task-B"}

    only_a = analyze_regressions(logger, task="task-A")
    assert {r.task for r in only_a.regressions} == {"task-A"}
    assert {p.task for p in only_a.new_passes} == {"task-A"}
    # Summary counts stay at the full population (issue #182).
    assert only_a.total == full.total == 3
    assert only_a.approvals == 0
    assert only_a.rejections == 3

    only_b = analyze_regressions(logger, task="task-B")
    assert {r.task for r in only_b.regressions} == {"task-B"}
    assert {p.task for p in only_b.new_passes} == {"task-B"}

    report = generate_regression_report(logger, task="task-A")
    regressed_a = _section(report, "Regressed Tasks")
    assert "task-A" in regressed_a
    assert "task-B" not in regressed_a
    new_passes_a = _section(report, "New Passes")
    assert "task-A" in new_passes_a
    assert "task-B" not in new_passes_a


def test_analyze_regressions_task_with_no_matches_returns_no_rows_message(tmp_path):
    logger = TraceLogger(tmp_path / "traces.db")
    _plant_mixed_sessions(logger)

    report = generate_regression_report(logger, task="does-not-exist")
    assert report == "no rows for task does-not-exist\n"


def test_cli_regression_report_task_filter_markdown(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(["regression-report", "--db", str(db), "--task", "task-A"])
    captured = capsys.readouterr()
    assert rc == 0
    assert "## Regressed Tasks" in captured.out
    assert "task-A" in captured.out
    assert "task-B" not in captured.out
    assert "## New Passes" in captured.out


def test_cli_regression_report_task_filter_json(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(
        [
            "regression-report",
            "--db",
            str(db),
            "--task",
            "task-A",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert payload["total"] == 3
    assert payload["approvals"] == 0
    assert payload["rejections"] == 3
    assert {row["task"] for row in payload["regressions"]} == {"task-A"}
    assert {row["task"] for row in payload["new_passes"]} == {"task-A"}


def test_cli_regression_report_task_filter_no_match(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(
        [
            "regression-report",
            "--db",
            str(db),
            "--task",
            "does-not-exist",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "no rows for task does-not-exist\n"


def test_cli_regression_report_without_task_keeps_full_population(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(["regression-report", "--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "task-A" in captured.out
    assert "task-B" in captured.out


def test_cli_regression_report_format_json_unfiltered(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(["regression-report", "--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0
    payload = json.loads(captured.out)
    assert {row["task"] for row in payload["regressions"]} == {"task-A", "task-B"}
    assert {row["task"] for row in payload["new_passes"]} == {"task-A", "task-B"}


def test_cli_regression_report_json_parses_as_regression_analysis(tmp_path, capsys):
    """Issue #269: --format json emits a payload that round-trips through the
    RegressionAnalysis pydantic model, with the expected regressions count."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(["regression-report", "--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0
    analysis = RegressionAnalysis.model_validate_json(captured.out)
    # _plant_mixed_sessions yields 2 regressions (task-A in B, task-B in C).
    assert len(analysis.regressions) == 2
    assert {r.task for r in analysis.regressions} == {"task-A", "task-B"}
    assert analysis.total == 3
    assert analysis.approvals == 0
    assert analysis.rejections == 3


def test_cli_regression_report_out_json_autoselects_format(tmp_path):
    """Issue #269: --out report.json selects json even without --format."""
    db = tmp_path / "traces.db"
    out = tmp_path / "report.json"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(["regression-report", "--db", str(db), "--out", str(out)])
    assert rc == 0
    assert out.exists()
    analysis = RegressionAnalysis.model_validate_json(out.read_text(encoding="utf-8"))
    assert {r.task for r in analysis.regressions} == {"task-A", "task-B"}


def test_cli_regression_report_out_markdown_stays_markdown(tmp_path):
    """Issue #269: a non-.json --out keeps the markdown default."""
    db = tmp_path / "traces.db"
    out = tmp_path / "report.md"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(["regression-report", "--db", str(db), "--out", str(out)])
    assert rc == 0
    body = out.read_text(encoding="utf-8")
    assert "## Regressed Tasks" in body


def test_cli_regression_report_format_json_with_fail_on_regression_gates(tmp_path, capsys):
    """Issue #269: --fail-on-regression gates exit code under json output too."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _plant_mixed_sessions(logger)

    rc = cli_main(
        [
            "regression-report",
            "--db",
            str(db),
            "--format",
            "json",
            "--fail-on-regression",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 1
    assert json.loads(captured.out)["regressions"]


# ---------------------------------------------------------------------------
# Issue #466: token budget aborts in regression report.
# The report surfaces token budget aborts as a distinct failure category
# separate from task regressions.
# ---------------------------------------------------------------------------


def _plant_task_aborted(logger, session_id, reason):
    """Plant a ``task_aborted`` event for a session."""
    import time

    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        time.sleep(0.01)
        logger.record(
            sid,
            kind="task_aborted",
            payload={"reason": reason, "tokens_used": 8000, "token_budget": 5000},
        )


def test_analyze_regressions_includes_token_budget_abort_count(tmp_path):
    """``token_budget_abort_count`` is counted in the regression analysis."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _plant_task_aborted(logger, "sess-1", "token_budget")
    _plant_task_aborted(logger, "sess-2", "token_budget")
    _plant_task_aborted(logger, "sess-3", "wall_clock")

    result = analyze_regressions(logger)

    assert result.token_budget_abort_count == 2


def test_analyze_regressions_token_budget_abort_zero_when_no_aborts(tmp_path):
    """``token_budget_abort_count`` is 0 when no session hit the token budget."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _plant_task_aborted(logger, "sess-1", "wall_clock")

    result = analyze_regressions(logger)

    assert result.token_budget_abort_count == 0


def test_generate_regression_report_shows_token_budget_aborts_section(tmp_path):
    """Regression report includes a Token Budget Aborts section."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _plant_task_aborted(logger, "sess-1", "token_budget")

    report = generate_regression_report(logger)

    assert "## Token Budget Aborts" in report
    assert "1 session(s)" in report


def test_cli_regression_report_json_includes_token_budget_abort_count(tmp_path, capsys):
    """JSON output includes token_budget_abort_count."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _plant_task_aborted(logger, "sess-1", "token_budget")
    _plant_task_aborted(logger, "sess-2", "token_budget")

    rc = cli_main(["regression-report", "--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["token_budget_abort_count"] == 2
