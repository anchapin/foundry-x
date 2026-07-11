"""Tests for the KPI computation and CLI (issues #39, #98, #101).

Issue #98: verdicts are seeded through
:func:`~foundry_x.observability.regression_report.record_verdict` so the tests
exercise the real persisted :class:`CriticVerdict` payload shape
(``approved`` / ``passed_checks`` / ``failed_checks``) rather than a synthetic
``{"verdict", "regression"}`` fixture that ``record_verdict`` never emits.
"""

from __future__ import annotations

import json
import time

import pytest

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.kpis import KpiSummary, compute_kpis, main
from foundry_x.observability.regression_report import record_verdict
from foundry_x.trace.logger import TraceLogger


def _seed_session(
    logger: TraceLogger,
    harness_version: str,
    approved: bool | None = None,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
    injection_block_count: int = 0,
) -> str:
    """Create a session with task_received + optional persisted critic_verdict.

    When ``approved`` is not ``None`` a real CriticVerdict is persisted via
    ``record_verdict`` (issue #98), so the trace store holds the same
    ``VerdictRecord`` payload the production path writes.

    Issue #120 adds the optional ``injection_block_count`` parameter: when
    >0, that many ``injection_blocked`` events are planted so the per-
    session KPI counter has something to surface.
    """
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        if approved is not None:
            # Small delay so cycle-time is measurably positive.
            time.sleep(0.01)
            record_verdict(
                logger,
                sid,
                CriticVerdict(
                    approved=approved,
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


def test_compute_kpis_with_planted_data(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # 2 approved, 1 rejected → improvement 2/3. The rejected session fails
    # "bench", which the two prior sessions passed → 1 regressed session of 3.
    _seed_session(logger, "v1", approved=True, passed_checks=["bench"])
    _seed_session(logger, "v1", approved=True, passed_checks=["bench"])
    _seed_session(logger, "v1", approved=False, failed_checks=["bench"])

    summary = compute_kpis(logger)

    assert isinstance(summary, KpiSummary)
    assert summary.cycle_time_seconds is not None
    assert summary.cycle_time_seconds > 0.0
    assert 0.0 <= summary.regression_rate <= 1.0
    assert summary.improvement_rate == 2 / 3
    assert summary.regression_rate == 1 / 3
    assert summary.injection_blocks == {}


def test_regression_rate_counts_prior_pass_now_failing(tmp_path):
    """A task passing then failing in a later verdict counts as a regression."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", approved=True, passed_checks=["smoke"])
    _seed_session(logger, "v1", approved=False, failed_checks=["smoke"])

    summary = compute_kpis(logger)

    # 1 of 2 sessions regressed; 1 of 2 verdicts approved.
    assert summary.regression_rate == 1 / 2
    assert summary.improvement_rate == 1 / 2


def test_no_regression_when_failure_never_passed(tmp_path):
    """A failing task that was never previously passing is not a regression."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", approved=True, passed_checks=["smoke"])
    _seed_session(logger, "v1", approved=False, failed_checks=["brand_new"])

    summary = compute_kpis(logger)

    assert summary.regression_rate == 0.0
    assert summary.improvement_rate == 1 / 2


def test_compute_kpis_empty_db(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    summary = compute_kpis(logger)
    assert summary.cycle_time_seconds is None
    assert summary.regression_rate == 0.0
    assert summary.improvement_rate == 0.0
    assert summary.injection_blocks == {}


def test_compute_kpis_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)
    _seed_session(logger, "v2", approved=False)

    summary = compute_kpis(logger, harness_version="v1")
    assert summary.improvement_rate == 1.0

    summary_v2 = compute_kpis(logger, harness_version="v2")
    assert summary_v2.improvement_rate == 0.0


def test_main_prints_markdown_table(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)
    _seed_session(logger, "v1", approved=False)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()

    assert rc == 0
    output = captured.out
    assert "Cycle Time" in output
    assert "Regression Rate" in output
    assert "Improvement Rate" in output
    # No injection blocks planted → no extra section.
    assert "Injection Blocked" not in output


# ---------------------------------------------------------------------------
# Issue #120: per-session ``injection_blocked`` count is surfaced by the
# ``foundry-kpis`` CLI when ≥1 session has ≥1 block. A clean trace store
# stays compact (no extra rows in the markdown table).
# ---------------------------------------------------------------------------


def test_injection_blocks_counted_per_session(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_session(logger, "v1", approved=True, injection_block_count=2)
    s2 = _seed_session(logger, "v1", approved=True, injection_block_count=1)
    # Clean session contributes nothing to the map.
    _seed_session(logger, "v1", approved=True)

    summary = compute_kpis(logger)

    assert summary.injection_blocks == {s1: 2, s2: 1}
    assert sum(summary.injection_blocks.values()) == 3


def test_injection_blocks_empty_when_no_events(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)

    summary = compute_kpis(logger)
    assert summary.injection_blocks == {}


def test_main_renders_injection_block_section_when_present(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_session(logger, "v1", approved=True, injection_block_count=3)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Injection Blocked" in output
    assert "3 block(s) across 1 session(s)" in output
    assert s1 in output
    assert "| 3 |" in output


def test_main_omits_injection_block_section_when_clean(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    # Compact output for a clean store — no extra section, no extra table.
    assert "Injection Blocked" not in captured.out


# ---------------------------------------------------------------------------
# Issue #101: machine-readable JSON snapshot of the KPI summary.  The top-
# level key set is the stable contract CI / dashboards depend on; the
# pydantic round-trip guarantees the JSON shape matches KpiSummary.
# ---------------------------------------------------------------------------


def test_main_json_format_emits_stable_top_level_keys(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    # Stable contract: every KpiSummary field is present at the top level
    # so downstream tooling can `payload["cycle_time_seconds"]` etc.
    assert set(payload.keys()) == {
        "cycle_time_seconds",
        "regression_rate",
        "improvement_rate",
        "injection_blocks",
    }


def test_main_json_round_trips_through_kpi_summary(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)
    _seed_session(logger, "v1", approved=False, failed_checks=["task"])

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    parsed = KpiSummary.model_validate_json(captured.out)
    assert parsed == compute_kpis(logger)


def test_main_format_auto_detects_json_from_out_extension(tmp_path):
    db = tmp_path / "traces.db"
    out = tmp_path / "kpis.json"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)

    rc = main(["--db", str(db), "--out", str(out)])
    assert rc == 0

    assert out.exists()
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert "cycle_time_seconds" in payload
    assert "regression_rate" in payload


def test_main_explicit_markdown_format_overrides_json_extension(tmp_path):
    db = tmp_path / "traces.db"
    out = tmp_path / "anything.json"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", approved=True)

    rc = main(["--db", str(db), "--format", "markdown", "--out", str(out)])
    assert rc == 0

    text = out.read_text(encoding="utf-8")
    # Explicit --format wins over extension: Markdown table is written.
    assert "Cycle Time" in text
    with pytest.raises(json.JSONDecodeError):
        json.loads(text)


def test_main_json_includes_injection_blocks_when_present(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_session(logger, "v1", approved=True, injection_block_count=2)
    s2 = _seed_session(logger, "v1", approved=True, injection_block_count=1)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["injection_blocks"] == {s1: 2, s2: 1}
    assert sum(payload["injection_blocks"].values()) == 3
