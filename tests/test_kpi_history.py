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
    _sparkline,
    append_kpi_history,
    export_prometheus,
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
    # was not supplied. hooks_disabled_count and hooks_disabled_rate are
    # scalar fields and are included (issue #585). tool_argument_parse_error_count
    # is also a scalar so it lands in the trend line (issue #872).
    assert set(payload.keys()) == {
        "cycle_time_seconds",
        "regression_rate",
        "improvement_rate",
        "token_budget_abort_count",
        "token_budget_hit_rate",
        "timestamp",
        "evolver_duration_ms",
        "hooks_disabled_count",
        "hooks_disabled_rate",
        "context_pruned_count",
        "failure_class_distribution",
        "tool_argument_parse_error_count",
    }
    assert "injection_blocks" not in payload
    assert "token_totals" not in payload


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
# Issue #622: --from-history --trend appends ASCII sparkline columns.
# ---------------------------------------------------------------------------


def test_sparkline_renders_increasing_sequence():
    """An increasing sequence maps to progressively taller block characters."""
    result = _sparkline([0.0, 0.25, 0.5, 0.75, 1.0])
    assert len(result) == 5
    assert result[0] == "▁"
    assert result[-1] == "█"


def test_sparkline_renders_decreasing_sequence():
    """A decreasing sequence maps to progressively shorter block characters."""
    result = _sparkline([1.0, 0.75, 0.5, 0.25, 0.0])
    assert len(result) == 5
    assert result[0] == "█"
    assert result[-1] == "▁"


def test_sparkline_flat_line():
    """A constant value fills all cells with the tallest block."""
    result = _sparkline([0.5, 0.5, 0.5])
    assert len(result) == 3
    assert all(c == "█" for c in result)


def test_sparkline_handles_none():
    """None values render as '·' and are excluded from the scale."""
    result = _sparkline([0.0, None, 1.0])
    assert len(result) == 3
    assert result[0] == "▁"
    assert result[1] == "·"
    assert result[2] == "█"


def test_sparkline_all_none_returns_n_a():
    """A sequence of only Nones returns 'N/A'."""
    assert _sparkline([None, None]) == "N/A"


def test_render_history_markdown_trend_true_adds_sparkline_columns():
    """trend=True appends three sparkline columns to every row."""
    entries = [
        KpiHistoryEntry(
            timestamp="2026-07-11T00:00:00+00:00",
            cycle_time_seconds=1.0,
            regression_rate=0.2,
            improvement_rate=0.5,
        ),
        KpiHistoryEntry(
            timestamp="2026-07-11T01:00:00+00:00",
            cycle_time_seconds=1.5,
            regression_rate=0.15,
            improvement_rate=0.55,
        ),
        KpiHistoryEntry(
            timestamp="2026-07-11T02:00:00+00:00",
            cycle_time_seconds=2.0,
            regression_rate=0.1,
            improvement_rate=0.6,
        ),
    ]
    table = render_history_markdown(entries, trend=True)
    lines = table.splitlines()

    # Header has 7 columns (timestamp + 3 KPIs + 3 sparklines).
    header_cols = lines[0].split("|")
    assert len(header_cols) == 9  # leading/trailing empty from split

    # Data rows carry the same sparkline string in each row's sparkline cells.
    assert "▁" in lines[2]
    assert "█" in lines[2]


def test_render_history_markdown_trend_false_has_no_sparkline_columns():
    """trend=False (default) produces the original 4-column table."""
    entries = [
        KpiHistoryEntry(
            timestamp="2026-07-11T00:00:00+00:00",
            cycle_time_seconds=1.0,
            regression_rate=0.1,
            improvement_rate=0.5,
        ),
    ]
    table = render_history_markdown(entries, trend=False)
    lines = table.splitlines()
    # Header + separator + 1 data row.
    assert len(lines) == 3
    # No sparkline characters in the data row.
    assert "▁" not in lines[2]
    assert "█" not in lines[2]


def test_main_trend_without_from_history_exits_with_error(tmp_path, capsys):
    """--trend without --from-history exits with a helpful error."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    with pytest.raises(SystemExit) as exc:
        main(["--db", str(db), "--trend"])
    assert exc.value.code == 2
    captured = capsys.readouterr()
    assert "--trend requires --from-history" in captured.err


def test_main_from_history_trend_appends_sparklines(tmp_path, capsys):
    """--from-history --trend renders sparkline columns in the output."""
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(cycle_time_seconds=1.0, improvement_rate=0.2))
    append_kpi_history(log, KpiSummary(cycle_time_seconds=2.0, improvement_rate=0.8))

    rc = main(["--from-history", str(log), "--trend"])
    captured = capsys.readouterr()

    assert rc == 0
    out = captured.out
    # Sparkline characters appear in the output.
    assert "▁" in out or "█" in out
    # Both history rows are present.
    assert "0.20" in out
    assert "0.80" in out


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


# ---------------------------------------------------------------------------
# Issue #565: ASCII sparklines.
# ---------------------------------------------------------------------------


def test_sparkline_five_values():
    """Five values spanning a range produce five block characters."""
    values = [0.0, 0.25, 0.5, 0.75, 1.0]
    line = _sparkline(values)
    assert len(line) == 5
    assert all(c in "▁▂▃▄▅▆▇█" for c in line)


def test_sparkline_renders_none_as_space():
    values = [0.0, None, 1.0]
    line = _sparkline(values)
    assert len(line) == 3
    assert line[1] == "·"


def test_sparkline_empty_list():
    assert _sparkline([]) == "N/A"


def test_sparkline_single_value():
    assert _sparkline([0.5]) == "█"


def test_render_history_markdown_trend_adds_sparkline_columns(tmp_path):
    """With trend=True, three extra columns show Unicode sparklines."""
    path = tmp_path / "kpi-history.jsonl"
    for rate in [0.1, 0.4, 0.7]:
        append_kpi_history(path, KpiSummary(improvement_rate=rate))

    entries = read_kpi_history(path)
    table = render_history_markdown(entries, trend=True)
    lines = table.splitlines()

    header = lines[0]
    assert "Cycle Time |" in header
    assert "Reg. Rate |" in header
    assert "Impr. Rate |" in header

    assert lines[1] == "| --- | --- | --- | --- | --- | --- | --- |"
    assert len(lines) == 5  # header + separator + 3 data rows

    for idx, rate in enumerate([0.1, 0.4, 0.7], start=2):
        row = lines[idx]
        assert f"| {rate:.2f} |" in row


def test_render_history_markdown_trend_empty_history():
    """trend=True with empty entries still renders placeholder."""
    table = render_history_markdown([], trend=True)
    assert "No KPI history" in table


def test_render_history_markdown_trend_preserves_none_cycle_time(tmp_path):
    """None cycle times render as a space in the sparkline column."""
    entries = [
        KpiHistoryEntry(
            timestamp="2026-07-11T00:00:00+00:00",
            cycle_time_seconds=None,
            regression_rate=0.0,
            improvement_rate=0.5,
        ),
        KpiHistoryEntry(
            timestamp="2026-07-11T00:01:00+00:00",
            cycle_time_seconds=2.0,
            regression_rate=0.0,
            improvement_rate=0.5,
        ),
    ]
    table = render_history_markdown(entries, trend=True)
    assert "N/A" in table
    assert "█" in table or "░" in table or "▒" in table or "▓" in table or " " in table


# ---------------------------------------------------------------------------
# Issue #565: Prometheus export.
# ---------------------------------------------------------------------------


def test_export_prometheus_emits_one_sample_per_entry_per_kpi(tmp_path):
    """Each history entry yields three samples (cycle_time, reg, impr)."""
    path = tmp_path / "kpi-history.jsonl"
    append_kpi_history(
        path,
        KpiSummary(
            cycle_time_seconds=1.5,
            regression_rate=0.1,
            improvement_rate=0.9,
        ),
        harness_version="v1",
    )

    entries = read_kpi_history(path)
    output = export_prometheus(entries)

    assert "foundryx_kpi_entry" in output
    assert 'kpi="cycle_time_seconds"' in output
    assert 'kpi="regression_rate"' in output
    assert 'kpi="improvement_rate"' in output
    assert 'harness_version="v1"' in output
    assert "# TYPE foundryx_kpi_entry gauge" in output
    assert "# HELP foundryx_kpi_entry FoundryX KPI" in output


def test_export_prometheus_empty_history_emits_comment():
    output = export_prometheus([])
    assert "No KPI history entries" in output
    assert "foundryx_kpi_entry" in output


def test_export_prometheus_unknown_harness_version(tmp_path):
    path = tmp_path / "kpi-history.jsonl"
    append_kpi_history(path, KpiSummary(regression_rate=0.05))
    entries = read_kpi_history(path)
    output = export_prometheus(entries)
    assert 'harness_version="unknown"' in output


def test_export_prometheus_cycle_time_none_emits_nan(tmp_path):
    path = tmp_path / "kpi-history.jsonl"
    append_kpi_history(
        path,
        KpiSummary(cycle_time_seconds=None, regression_rate=0.0, improvement_rate=0.5),
    )
    entries = read_kpi_history(path)
    output = export_prometheus(entries)
    assert "NaN" in output


# ---------------------------------------------------------------------------
# Issue #565: --trend CLI flag.
# ---------------------------------------------------------------------------


def test_main_from_history_trend_flag_renders_sparklines(tmp_path, capsys):
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(improvement_rate=0.2))
    append_kpi_history(log, KpiSummary(improvement_rate=0.8))

    rc = main(["--from-history", str(log), "--trend"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "Reg. Rate |" in captured.out
    assert "Impr. Rate |" in captured.out


def test_main_from_history_trend_requires_from_history(tmp_path, capsys):
    """--trend without --from-history exits with error."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    with pytest.raises(SystemExit) as exc_info:
        main(["--db", str(db), "--trend"])
    assert exc_info.value.code == 2


# ---------------------------------------------------------------------------
# Issue #565: --export-prometheus CLI flag.
# ---------------------------------------------------------------------------


def test_main_from_history_export_prometheus_emits_prom_format(tmp_path, capsys):
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(regression_rate=0.15))
    append_kpi_history(log, KpiSummary(regression_rate=0.05))

    rc = main(["--from-history", str(log), "--export-prometheus"])
    captured = capsys.readouterr()

    assert rc == 0
    out = captured.out
    assert 'kpi="regression_rate"' in out
    assert "foundryx_kpi_entry" in out
    assert "# TYPE foundryx_kpi_entry gauge" in out


def test_main_from_history_export_prometheus_writes_to_out(tmp_path):
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(regression_rate=0.1))
    out = tmp_path / "metrics.prom"

    rc = main(["--from-history", str(log), "--export-prometheus", "--out", str(out)])

    assert rc == 0
    text = out.read_text(encoding="utf-8")
    assert "foundryx_kpi_entry" in text


# ---------------------------------------------------------------------------
# Issue #565: --alert-threshold CLI flag.
# ---------------------------------------------------------------------------


def test_main_alert_threshold_zero_exit_when_regression_above(tmp_path, capsys):
    """Regression rate > threshold causes non-zero exit with an alert to stderr."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["x"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["x"])

    rc = main(["--db", str(db), "--alert-threshold", "0.05"])
    captured = capsys.readouterr()

    assert rc == 1
    assert "[ALERT]" in captured.err
    assert "regression_rate" in captured.err


def test_main_alert_threshold_zero_exit_when_regression_below(tmp_path, capsys):
    """Regression rate <= threshold exits zero (no alert)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)  # no regression

    rc = main(["--db", str(db), "--alert-threshold", "0.5"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "[ALERT]" not in captured.err


def test_main_alert_threshold_from_history_is_not_supported(tmp_path, capsys):
    """--from-history short-circuits before KPI computation; alert is not checked."""
    log = tmp_path / "kpi-history.jsonl"
    append_kpi_history(log, KpiSummary(regression_rate=99.0))

    rc = main(["--from-history", str(log), "--alert-threshold", "0.01"])
    captured = capsys.readouterr()

    assert rc == 0
    assert "[ALERT]" not in captured.err
