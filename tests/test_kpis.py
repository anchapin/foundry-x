"""Tests for the KPI computation and CLI (issues #39, #98, #101).

Issue #98: verdicts are seeded through
:func:`~foundry_x.observability.regression_report.record_verdict` so the tests
exercise the real persisted :class:`CriticVerdict` payload shape
(``approved`` / ``passed_checks`` / ``failed_checks``) rather than a synthetic
``{"verdict", "regression"}`` fixture that ``record_verdict`` never emits.
"""

from __future__ import annotations

import json
import re
import time

import pytest

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.kpis import (
    KpiComparison,
    KpiSummary,
    _format_delta,
    compare_kpis,
    compute_kpis,
    main,
)
from foundry_x.observability.regression_report import record_verdict
from foundry_x.trace.logger import TraceLogger


def _seed_session(
    logger: TraceLogger,
    harness_version: str,
    verdict: bool | None = None,
    passed_checks: list[str] | None = None,
    failed_checks: list[str] | None = None,
    injection_block_count: int = 0,
) -> str:
    """Create a session with task_received + optional persisted critic_verdict.

    When ``verdict`` is not ``None`` a real CriticVerdict is persisted via
    ``record_verdict`` (issue #98), so the trace store holds the same
    ``VerdictRecord`` payload the production path writes.

    Issue #120 adds the optional ``injection_block_count`` parameter: when
    >0, that many ``injection_blocked`` events are planted so the per-
    session KPI counter has something to surface.
    """
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        if verdict is not None:
            # Small delay so cycle-time is measurably positive.
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


def test_compute_kpis_with_planted_data(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # 2 approved, 1 rejected → improvement 2/3. The rejected session fails
    # "bench", which the two prior sessions passed → 1 regressed session of 3.
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["bench"])

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

    _seed_session(logger, "v1", verdict=True, passed_checks=["smoke"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["smoke"])

    summary = compute_kpis(logger)

    # 1 of 2 sessions regressed; 1 of 2 verdicts approved.
    assert summary.regression_rate == 1 / 2
    assert summary.improvement_rate == 1 / 2


def test_no_regression_when_failure_never_passed(tmp_path):
    """A failing task that was never previously passing is not a regression."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_session(logger, "v1", verdict=True, passed_checks=["smoke"])
    _seed_session(logger, "v1", verdict=False, failed_checks=["brand_new"])

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
    assert summary.token_totals == {}


def test_compute_kpis_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v2", verdict=False)

    summary = compute_kpis(logger, harness_version="v1")
    assert summary.improvement_rate == 1.0

    summary_v2 = compute_kpis(logger, harness_version="v2")
    assert summary_v2.improvement_rate == 0.0


def test_main_prints_markdown_table(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=False)

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

    s1 = _seed_session(logger, "v1", verdict=True, injection_block_count=2)
    s2 = _seed_session(logger, "v1", verdict=True, injection_block_count=1)
    # Clean session contributes nothing to the map.
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)

    assert summary.injection_blocks == {s1: 2, s2: 1}
    assert sum(summary.injection_blocks.values()) == 3


def test_injection_blocks_empty_when_no_events(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)
    assert summary.injection_blocks == {}


def test_main_renders_injection_block_section_when_present(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_session(logger, "v1", verdict=True, injection_block_count=3)

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
    _seed_session(logger, "v1", verdict=True)

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
    _seed_session(logger, "v1", verdict=True)

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
        "token_totals",
        "token_budget_hit_rate",
    }


def test_main_json_round_trips_through_kpi_summary(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=False, failed_checks=["task"])

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    parsed = KpiSummary.model_validate_json(captured.out)
    assert parsed == compute_kpis(logger)


def test_main_format_auto_detects_json_from_out_extension(tmp_path):
    db = tmp_path / "traces.db"
    out = tmp_path / "kpis.json"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

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
    _seed_session(logger, "v1", verdict=True)

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
    s1 = _seed_session(logger, "v1", verdict=True, injection_block_count=2)
    s2 = _seed_session(logger, "v1", verdict=True, injection_block_count=1)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["injection_blocks"] == {s1: 2, s2: 1}
    assert sum(payload["injection_blocks"].values()) == 3


# ---------------------------------------------------------------------------
# Issue #100: compare baseline vs candidate harness versions. The CLI gains
# --baseline-harness-version / --candidate-harness-version flags; when both
# are supplied it prints Baseline / Candidate / Delta columns whose delta
# sign convention follows the PRD (improvement up = good; regression and
# cycle-time up = bad).
# ---------------------------------------------------------------------------


def test_format_delta_sign_convention():
    # Improvement-rate increase is positive (good).
    assert _format_delta(0.33, 0.50, higher_is_better=True) == "+0.17 (positive)"
    # Improvement-rate decrease is negative (bad).
    assert _format_delta(1.00, 0.50, higher_is_better=True) == "-0.50 (negative)"
    # Regression-rate increase is negative (bad).
    assert _format_delta(0.00, 0.50, higher_is_better=False) == "+0.50 (negative)"
    # Cycle-time increase is negative (bad).
    assert _format_delta(5.00, 10.00, higher_is_better=False) == "+5.00 (negative)"
    # Cycle-time decrease is positive (good).
    assert _format_delta(10.00, 5.00, higher_is_better=False) == "-5.00 (positive)"
    # No movement is neutral.
    assert _format_delta(0.50, 0.50, higher_is_better=True) == "+0.00 (neutral)"
    # An unmeasured side is N/A.
    assert _format_delta(None, 1.0, higher_is_better=True) == "N/A"
    assert _format_delta(1.0, None, higher_is_better=False) == "N/A"


def test_compare_kpis_returns_candidate_minus_baseline_deltas(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: both approved, both pass "bench".
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2: one passes "bench", one regresses it.
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=False, failed_checks=["bench"])

    comparison = compare_kpis(logger, "v1", "v2")

    assert isinstance(comparison, KpiComparison)
    assert comparison.baseline.improvement_rate == 1.0
    assert comparison.candidate.improvement_rate == 0.5
    assert comparison.baseline.regression_rate == 0.0
    assert comparison.candidate.regression_rate == 0.5
    assert comparison.deltas["improvement_rate"] == pytest.approx(-0.5)
    assert comparison.deltas["regression_rate"] == pytest.approx(0.5)
    # Both versions have verdicts, so cycle-time deltas are real numbers.
    assert comparison.deltas["cycle_time_seconds"] is not None


def test_main_comparison_prints_baseline_candidate_delta_columns(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline v1: both approved, both pass "bench".
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    # Candidate v2: one passes "bench", one regresses it.
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=False, failed_checks=["bench"])

    rc = main(
        [
            "--db",
            str(db),
            "--baseline-harness-version",
            "v1",
            "--candidate-harness-version",
            "v2",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    lines = output.splitlines()
    assert "| KPI | Baseline | Candidate | Delta |" in lines

    def _row(name: str) -> str:
        return next(line for line in lines if line.lstrip().startswith(f"| {name}"))

    improvement = _row("Improvement Rate")
    # 1.00 baseline, 0.50 candidate; decrease → marked negative (bad).
    assert "1.00 | 0.50 | -0.50 (negative)" in improvement

    regression = _row("Regression Rate")
    # 0.00 baseline, 0.50 candidate; increase → marked negative (bad).
    assert "0.00 | 0.50 | +0.50 (negative)" in regression

    # Cycle-time delta is rendered for all three KPIs: both sides measured,
    # so the cell is a signed value carrying a PRD mark (not N/A).
    cycle = _row("Cycle Time (seconds)").strip()
    assert re.search(r"\| [-+]?\d+\.\d{2} \((positive|negative|neutral)\) \|$", cycle)


def test_main_comparison_marks_improvement_increase_as_positive(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    # Baseline: rejected (improvement 0.0).
    _seed_session(logger, "v1", verdict=False, failed_checks=["x"])
    # Candidate: approved (improvement 1.0) → improvement increases.
    _seed_session(logger, "v2", verdict=True, passed_checks=["x"])

    rc = main(
        [
            "--db",
            str(db),
            "--baseline-harness-version",
            "v1",
            "--candidate-harness-version",
            "v2",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0

    improvement = next(line for line in captured.out.splitlines() if "Improvement Rate" in line)
    # 0.00 → 1.00 is an improvement-rate increase → marked positive.
    assert "0.00 | 1.00 | +1.00 (positive)" in improvement


def test_main_comparison_json_structure(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=False, failed_checks=["bench"])

    rc = main(
        [
            "--db",
            str(db),
            "--baseline-harness-version",
            "v1",
            "--candidate-harness-version",
            "v2",
            "--format",
            "json",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert set(payload.keys()) == {"baseline", "candidate", "deltas"}
    assert payload["baseline"]["improvement_rate"] == 1.0
    assert payload["candidate"]["improvement_rate"] == 0.0
    assert payload["deltas"]["improvement_rate"] == -1.0


def test_main_comparison_requires_both_versions(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    with pytest.raises(SystemExit) as exc:
        main(["--db", str(db), "--baseline-harness-version", "v1"])
    assert exc.value.code == 2


# ---------------------------------------------------------------------------
# Issue #271: per-session ``token_totals`` is surfaced by the
# ``foundry-kpis`` CLI when ``model_response`` events carry ``token_usage``.
# A trace store with no token accounting stays compact (no token section or
# map entries). Mirrors the ``injection_blocked`` "show only when present"
# contract so the JSON contract is additive, not breaking.
# ---------------------------------------------------------------------------


def _seed_model_response_usage(
    logger: TraceLogger,
    harness_version: str,
    usage_payloads: list[dict[str, int] | None],
) -> str:
    """Plant a session whose ``model_response`` events carry ``token_usage``.

    Each entry in *usage_payloads* becomes one ``model_response`` event whose
    ``usage`` key matches the runner's wire format
    (``{"prompt_tokens", "completion_tokens", "total_tokens"}``) and whose
    ``tokens_used`` is the running cumulative total, exactly as
    :func:`~foundry_x.execution.runner.run_task` records it (issues #191, #197).
    A ``None`` entry simulates an endpoint that omits usage accounting.
    """
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        running = 0
        for step, usage in enumerate(usage_payloads):
            if usage is not None:
                running += usage["total_tokens"]
            logger.record(
                sid,
                kind="model_response",
                payload={
                    "step": step,
                    "finish_reason": "stop",
                    "usage": usage,
                    "tokens_used": running,
                },
            )
    return sid


def test_token_totals_sums_total_tokens_per_session(tmp_path):
    """``token_totals`` accumulates ``usage.total_tokens`` per session."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_model_response_usage(
        logger,
        "v1",
        [
            {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
        ],
    )

    summary = compute_kpis(logger)

    assert summary.token_totals == {s1: 45}


def test_token_totals_multiple_sessions(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 10, "completion_tokens": 0, "total_tokens": 10}]
    )
    s2 = _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50}]
    )

    summary = compute_kpis(logger)

    assert summary.token_totals == {s1: 10, s2: 50}
    assert sum(summary.token_totals.values()) == 60


def test_token_totals_skips_events_with_null_usage(tmp_path):
    """A ``model_response`` with ``usage: None`` contributes zero tokens."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    s1 = _seed_model_response_usage(
        logger,
        "v1",
        [None, {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10}],
    )

    summary = compute_kpis(logger)

    assert summary.token_totals == {s1: 10}


def test_token_totals_omits_session_with_no_usage(tmp_path):
    """A session whose ``model_response`` events all omit ``usage`` is absent."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    _seed_model_response_usage(logger, "v1", [None, None])

    summary = compute_kpis(logger)

    assert summary.token_totals == {}


def test_token_totals_empty_when_no_model_response_events(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    summary = compute_kpis(logger)

    assert summary.token_totals == {}


def test_token_totals_respects_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2}]
    )
    _seed_model_response_usage(
        logger, "v2", [{"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6}]
    )

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert list(summary_v1.token_totals.values()) == [2]
    assert list(summary_v2.token_totals.values()) == [6]


def test_main_markdown_renders_token_usage_section(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}]
    )

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Token Usage" in output
    assert "150 token(s) across 1 session(s)" in output
    assert "| Session | Tokens |" in output
    assert s1 in output
    assert "| 150 |" in output


def test_main_markdown_omits_token_usage_when_clean(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    # Compact output for a store with no token accounting.
    assert "Token Usage" not in captured.out


def test_main_json_includes_token_totals(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_model_response_usage(
        logger, "v1", [{"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10}]
    )

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["token_totals"] == {s1: 10}


# ---------------------------------------------------------------------------
# Issue #551: ``token_budget_hit_rate`` is a fourth tracked metric exposed
# via ``foundry-kpis``. It is the fraction of sessions that recorded at least
# one ``task_aborted(reason="token_budget")`` event.
# ---------------------------------------------------------------------------


def _seed_token_budget_abort(
    logger: TraceLogger,
    harness_version: str,
    abort_token_budget: bool = True,
) -> str:
    """Plant a session with an optional ``task_aborted(reason="token_budget")`` event.

    When *abort_token_budget* is True, the session records one
    ``task_aborted`` event with ``reason="token_budget"``. Otherwise
    the session has no abort events at all.
    """
    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        if abort_token_budget:
            logger.record(
                sid,
                kind="task_aborted",
                payload={
                    "reason": "token_budget",
                    "token_budget": 100000,
                    "timeout_s": None,
                },
            )
    return sid


def test_token_budget_hit_rate_fraction_of_sessions_with_abort(tmp_path):
    """``token_budget_hit_rate`` = sessions with >=1 token-budget abort / total sessions."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    # 2 sessions with token-budget abort, 1 clean session → rate = 2/3.
    _seed_token_budget_abort(logger, "v1", abort_token_budget=True)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=True)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=False)

    summary = compute_kpis(logger)

    assert summary.token_budget_hit_rate == pytest.approx(2 / 3)


def test_token_budget_hit_rate_zero_when_no_aborts(tmp_path):
    """Zero sessions with token-budget abort → rate = 0.0."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=False)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=False)

    summary = compute_kpis(logger)

    assert summary.token_budget_hit_rate == 0.0


def test_token_budget_hit_rate_all_sessions_abort(tmp_path):
    """All sessions with token-budget abort → rate = 1.0."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=True)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=True)

    summary = compute_kpis(logger)

    assert summary.token_budget_hit_rate == 1.0


def test_token_budget_hit_rate_respects_harness_version(tmp_path):
    """Harness version filter applies to both sessions and abort events."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=True)
    _seed_token_budget_abort(logger, "v2", abort_token_budget=False)

    summary_v1 = compute_kpis(logger, harness_version="v1")
    summary_v2 = compute_kpis(logger, harness_version="v2")

    assert summary_v1.token_budget_hit_rate == 1.0
    assert summary_v2.token_budget_hit_rate == 0.0


def test_token_budget_hit_rate_zero_on_empty_db(tmp_path):
    """No sessions → rate = 0.0."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    summary = compute_kpis(logger)

    assert summary.token_budget_hit_rate == 0.0


def test_main_markdown_includes_token_budget_hit_rate(tmp_path, capsys):
    """Markdown output includes Token Budget Hit Rate in the KPI table."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Token Budget Hit Rate" in output


def test_main_json_includes_token_budget_hit_rate(tmp_path, capsys):
    """JSON output includes token_budget_hit_rate at the top level."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=True)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert "token_budget_hit_rate" in payload


def test_compare_kpis_includes_token_budget_hit_rate_delta(tmp_path):
    """KpiComparison deltas include token_budget_hit_rate."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=False)
    _seed_token_budget_abort(logger, "v2", abort_token_budget=True)

    comparison = compare_kpis(logger, "v1", "v2")

    assert "token_budget_hit_rate" in comparison.deltas
    # baseline=0.0, candidate=1.0 → delta = 1.0
    assert comparison.deltas["token_budget_hit_rate"] == pytest.approx(1.0)


def test_main_comparison_includes_token_budget_hit_rate_delta(tmp_path, capsys):
    """Comparison markdown table includes Token Budget Hit Rate row."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_token_budget_abort(logger, "v1", abort_token_budget=False)
    _seed_token_budget_abort(logger, "v2", abort_token_budget=True)

    rc = main(
        [
            "--db",
            str(db),
            "--baseline-harness-version",
            "v1",
            "--candidate-harness-version",
            "v2",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Token Budget Hit Rate" in output


def test_token_budget_hit_rate_not_confused_with_wall_clock_abort(tmp_path):
    """Only ``reason="token_budget"`` counts; wall_clock aborts are ignored."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        logger.record(
            sid,
            kind="task_aborted",
            payload={
                "reason": "wall_clock",
                "token_budget": None,
                "timeout_s": 300.0,
            },
        )

    summary = compute_kpis(logger)

    # wall_clock abort should NOT count toward token_budget_hit_rate
    assert summary.token_budget_hit_rate == 0.0
