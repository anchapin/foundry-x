"""Tests for the append-only KPI history log (issue #183).

The history log gives the regression signal a temporal axis:
``foundry-kpis --log-to logs/kpi-history.jsonl`` appends one JSON
line per run, and ``foundry-kpis --from-history logs/kpi-history.jsonl``
prints a Markdown trend table that preserves append order.

The on-disk contract is the round-trip property: every persisted
line, parsed as :class:`KpiSummary`, must yield the same three PRD
KPIs as the originating run, and the per-session ``injection_blocks``
map must not appear in the read-back (the "minus per-session map"
half of the contract). Extra fields (``timestamp``,
``harness_version``) are ignored on parse — pydantic's default
``extra='ignore'`` model config makes the round-trip lossless for
the three KPIs.
"""

from __future__ import annotations

import fnmatch
import json
from pathlib import Path

import pytest

from foundry_x.observability.kpis import (
    KpiHistoryEntry,
    KpiSummary,
    append_kpi_history,
    main,
    read_kpi_history,
    render_history_markdown,
)
from foundry_x.observability.regression_report import record_verdict
from foundry_x.evolution.critic import CriticVerdict
from foundry_x.trace.logger import TraceLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
GITIGNORE = REPO_ROOT / ".gitignore"


def _seed_session(
    logger: TraceLogger,
    harness_version: str,
    verdict: bool = True,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
    injection_block_count: int = 0,
) -> str:
    """Seed a session with a real persisted CriticVerdict (issue #98).

    Mirrors the helper in ``test_kpis.py`` so the CLI tests can run
    ``foundry-kpis --log-to`` against a real trace store without
    depending on the sibling test module.
    """
    import time

    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        time.sleep(0.01)
        record_verdict(
            logger,
            sid,
            CriticVerdict(
                verdict=verdict,
                passed_checks=passed_checks or [],
                failed_checks=failed_checks or [],
            ),
        )
        for i in range(injection_block_count):
            logger.record(
                sid,
                kind="injection_blocked",
                payload={
                    "markers": ["ignore_previous"],
                    "tool": "read_file",
                    "preview": f"block {i}",
                },
            )
    return sid


# ---------------------------------------------------------------------------
# Round-trip contract: every JSON line parses as a KpiSummary with matching
# three-KPI fields, and the per-session injection_blocks map is absent
# (the "minus per-session map" half of the contract).
# ---------------------------------------------------------------------------


def test_append_kpi_history_writes_one_json_line(tmp_path):
    path = tmp_path / "kpi-history.jsonl"
    summary = KpiSummary(
        cycle_time_seconds=1.5,
        regression_rate=0.1,
        improvement_rate=0.9,
    )

    append_kpi_history(path, summary, harness_version="v1")

    text = path.read_text(encoding="utf-8")
    # Exactly one line — the append-only invariant.
    assert text.count("\n") == 1
    assert not text.endswith("\n\n")


def test_append_kpi_history_round_trips_through_kpi_summary(tmp_path):
    """Persisted line parses as KpiSummary with the three KPIs intact.

    This is the explicit acceptance criterion "JSON history round-trips
    through KpiSummary minus per-session map": the per-session
    ``injection_blocks`` field on the input is dropped, and the three
    numeric KPIs survive.
    """
    path = tmp_path / "kpi-history.jsonl"
    summary = KpiSummary(
        cycle_time_seconds=2.5,
        regression_rate=0.2,
        improvement_rate=0.8,
        # The per-session map that MUST NOT round-trip.
        injection_blocks={"session-abc": 3},
    )

    append_kpi_history(path, summary, harness_version="v1")

    line = path.read_text(encoding="utf-8").strip()
    parsed = KpiSummary.model_validate_json(line)
    assert parsed.cycle_time_seconds == pytest.approx(2.5)
    assert parsed.regression_rate == pytest.approx(0.2)
    assert parsed.improvement_rate == pytest.approx(0.8)
    # Minus the per-session map: the read-back has no injection_blocks.
    assert parsed.injection_blocks == {}


def test_append_kpi_history_omits_harness_version_when_unset(tmp_path):
    """The optional ``harness_version`` key is absent when not supplied.

    The history format only includes the field when the operator ran
    ``foundry-kpis --harness-version X --log-to ...``. Otherwise the
    line is the three KPIs plus ``timestamp``.
    """
    path = tmp_path / "kpi-history.jsonl"
    append_kpi_history(path, KpiSummary(cycle_time_seconds=1.0))

    line = path.read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert "harness_version" not in payload
    assert "timestamp" in payload
    # Still a valid KpiSummary round-trip.
    assert KpiSummary.model_validate_json(line).cycle_time_seconds == pytest.approx(1.0)


def test_append_kpi_history_creates_parent_directory(tmp_path):
    """The append path does not require a pre-existing directory."""
    path = tmp_path / "nested" / "logs" / "kpi-history.jsonl"
    append_kpi_history(path, KpiSummary(improvement_rate=0.5))

    assert path.exists()
    assert path.read_text(encoding="utf-8").strip()


def test_append_kpi_history_concurrent_appends_interleave_at_line_boundaries(tmp_path):
    """Multiple appends produce one JSON object per line, in order.

    Guards the append-only invariant under the simplest possible
    stress: three sequential appends must yield three parseable
    JSON lines with no embedded newlines or merge artifacts.
    """
    path = tmp_path / "kpi-history.jsonl"
    for i, value in enumerate([0.1, 0.5, 0.9]):
        append_kpi_history(path, KpiSummary(improvement_rate=value))

    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3
    rates = [json.loads(line)["improvement_rate"] for line in lines]
    assert rates == [0.1, 0.5, 0.9]


# ---------------------------------------------------------------------------
# Read path: file order, blank-line tolerance, missing file, malformed lines.
# ---------------------------------------------------------------------------


def test_read_kpi_history_missing_file_returns_empty(tmp_path):
    assert read_kpi_history(tmp_path / "does-not-exist.jsonl") == []


def test_read_kpi_history_skips_blank_lines(tmp_path):
    """A trailing newline (the standard end-of-file artifact) does not break parsing."""
    path = tmp_path / "kpi-history.jsonl"
    append_kpi_history(path, KpiSummary(improvement_rate=0.5))
    # Append a stray blank line — common when humans `echo >>` the log.
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    entries = read_kpi_history(path)
    assert len(entries) == 1
    assert entries[0].improvement_rate == pytest.approx(0.5)


def test_read_kpi_history_skips_malformed_lines(tmp_path):
    """A single garbage line does not blank the rest of the trend table."""
    path = tmp_path / "kpi-history.jsonl"
    append_kpi_history(path, KpiSummary(improvement_rate=0.1))
    path.write_text(path.read_text(encoding="utf-8") + "{not valid json\n", encoding="utf-8")
    append_kpi_history(path, KpiSummary(improvement_rate=0.9))

    entries = read_kpi_history(path)
    assert [e.improvement_rate for e in entries] == [pytest.approx(0.1), pytest.approx(0.9)]


# ---------------------------------------------------------------------------
# The 5-entry trend test from the issue's acceptance criteria.
# ---------------------------------------------------------------------------


def test_render_history_markdown_five_entries_in_correct_order(tmp_path):
    """The issue's canonical test: plant 5 entries, assert 5 rows in order."""
    path = tmp_path / "kpi-history.jsonl"
    # Plant five distinct runs with monotonically changing improvement_rate
    # so the order is observable in the rendered table.
    rates = [0.10, 0.25, 0.40, 0.55, 0.70]
    for rate in rates:
        append_kpi_history(
            path,
            KpiSummary(
                cycle_time_seconds=1.0 + rate,
                regression_rate=0.05,
                improvement_rate=rate,
            ),
            harness_version="v0",
        )

    entries = read_kpi_history(path)
    assert len(entries) == 5, "expected 5 history entries to be planted"

    table = render_history_markdown(entries)
    lines = table.splitlines()

    # Header + 5 data rows.
    assert lines[0] == "| Timestamp | Cycle Time (s) | Regression Rate | Improvement Rate |"
    assert lines[1] == "| --- | --- | --- | --- |"
    assert len(lines) == 7

    # Each data row carries the matching improvement_rate in append order.
    for idx, rate in enumerate(rates, start=2):
        row = lines[idx]
        # ``_format_value`` produces ``f"{value:.2f}"`` → "0.10", "0.25", ...
        assert f"| {rate:.2f} |" in row


def test_render_history_markdown_empty_history_renders_placeholder():
    table = render_history_markdown([])
    # The placeholder keeps CI summary cells template-stable.
    assert "No KPI history" in table


def test_render_history_markdown_handles_none_cycle_time(tmp_path):
    """A run with no measurable cycle time renders N/A, not 0.00.

    The renderer uses the same ``_format_value`` helper as the
    single-summary table, so the convention stays consistent.
    """
    entries = [
        KpiHistoryEntry(
            timestamp="2026-07-11T00:00:00+00:00",
            cycle_time_seconds=None,
            regression_rate=0.0,
            improvement_rate=0.0,
        )
    ]
    table = render_history_markdown(entries)
    assert "N/A" in table


# ---------------------------------------------------------------------------
# CLI integration: --log-to and --from-history.
# ---------------------------------------------------------------------------


def test_main_log_to_appends_to_jsonl(tmp_path):
    db = tmp_path / "traces.db"
    log = tmp_path / "kpi-history.jsonl"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db), "--log-to", str(log)])

    assert rc == 0
    text = log.read_text(encoding="utf-8")
    # Exactly one line per run.
    assert text.count("\n") == 1
    payload = json.loads(text.strip())
    # The persisted line is the KpiSummary minus per-session map plus
    # the timestamp; harness_version is absent because --harness-version
    # was not supplied.
    assert set(payload.keys()) == {
        "cycle_time_seconds",
        "regression_rate",
        "improvement_rate",
        "timestamp",
    }
    assert "injection_blocks" not in payload


def test_main_log_to_persists_harness_version_when_filtered(tmp_path):
    db = tmp_path / "traces.db"
    log = tmp_path / "kpi-history.jsonl"
    logger = TraceLogger(db)
    _seed_session(logger, "v2", verdict=True)

    rc = main(["--db", str(db), "--harness-version", "v2", "--log-to", str(log)])

    assert rc == 0
    payload = json.loads(log.read_text(encoding="utf-8").strip())
    assert payload["harness_version"] == "v2"


def test_main_log_to_creates_parent_directory(tmp_path):
    db = tmp_path / "traces.db"
    log = tmp_path / "nested" / "kpi-history.jsonl"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db), "--log-to", str(log)])

    assert rc == 0
    assert log.exists()


def test_main_log_to_increments_across_runs(tmp_path):
    """Two CLI invocations produce two history lines, in order."""
    db = tmp_path / "traces.db"
    log = tmp_path / "kpi-history.jsonl"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    main(["--db", str(db), "--log-to", str(log)])
    _seed_session(logger, "v1", verdict=False, failed_checks=["x"])
    main(["--db", str(db), "--log-to", str(log)])

    entries = read_kpi_history(log)
    assert len(entries) == 2


def test_main_from_history_prints_trend_table(tmp_path, capsys):
    """``--from-history`` short-circuits before reading the trace store."""
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(improvement_rate=0.4))
    append_kpi_history(log, KpiSummary(improvement_rate=0.6))

    # --from-history does not need --db; the trend is purely a file read.
    rc = main(["--from-history", str(log)])
    captured = capsys.readouterr()

    assert rc == 0
    out = captured.out
    assert "| Timestamp | Cycle Time (s) |" in out
    # Both rows appear, in append order: 0.40 then 0.60.
    assert "0.40" in out
    assert "0.60" in out
    assert out.index("0.40") < out.index("0.60")


def test_main_from_history_does_not_require_trace_db(tmp_path, capsys):
    """A bogus --db path is fine when --from-history short-circuits."""
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(improvement_rate=0.5))

    rc = main(["--db", "/nonexistent/trace.db", "--from-history", str(log)])
    captured = capsys.readouterr()

    assert rc == 0
    assert "0.50" in captured.out


def test_main_from_history_missing_file_renders_placeholder(tmp_path, capsys):
    rc = main(["--from-history", str(tmp_path / "missing.jsonl")])
    captured = capsys.readouterr()

    assert rc == 0
    # The empty-history placeholder is part of the contract: a missing
    # file is a valid starting state, not an error.
    assert "No KPI history" in captured.out


def test_main_from_history_writes_to_out_path(tmp_path):
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(improvement_rate=0.5))
    out = tmp_path / "trend.md"

    rc = main(["--from-history", str(log), "--out", str(out)])

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "| Timestamp | Cycle Time (s) |" in text
    assert "0.50" in text


# ---------------------------------------------------------------------------
# Static guard: the history file is gitignored. A future contributor who
# wipes ``logs/*.jsonl`` from ``.gitignore`` would silently start
# committing KPI history to the repo; this test fails loudly so the
# regression is caught at CI time, not when reviewing a 10 MB diff.
# ---------------------------------------------------------------------------


def test_kpi_history_file_is_gitignored():
    assert GITIGNORE.exists(), f"missing .gitignore at {GITIGNORE}"
    patterns = [
        line.strip()
        for line in GITIGNORE.read_text().splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    # The contract is "*.jsonl under logs/" — the issue's example path
    # is logs/kpi-history.jsonl, so the glob must cover it.
    matches = [p for p in patterns if fnmatch.fnmatch("logs/kpi-history.jsonl", p)]
    assert matches, (
        "logs/kpi-history.jsonl is not covered by any .gitignore pattern. "
        "Issue #183 requires the append-only KPI history log to be "
        "gitignored so it never gets committed."
    )
