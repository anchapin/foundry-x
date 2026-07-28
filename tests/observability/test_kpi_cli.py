"""KPI computation tests — split from test_kpis.py (issue #1282).

This file is kept for backwards compatibility. Tests have been moved to:
- tests/observability/test_cycle_time_kpi.py
- tests/observability/test_regression_rate_kpi.py
- tests/observability/test_improvement_rate_kpi.py
- tests/observability/test_context_efficiency_kpi.py
- tests/observability/test_token_budget_kpi.py
- tests/observability/test_kpi_comparison.py
- tests/observability/test_kpi_cli.py
"""

from __future__ import annotations

import json
import re
import time

import pytest

from foundry_x.evolution.critic import CriticVerdict
from foundry_x.observability.kpis import (
    KpiSummary,
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
    failure_class: str | None = None,
    hook_registry_error: bool = False,
    wall_clock_abort: bool = False,
    token_budget_abort: tuple[int, int] | None = None,
    event_limit_abort: bool = False,
    tool_argument_parse_error_count: int = 0,
    generation_exhausted_count: int = 0,
) -> str:
    """Create a session with task_received + optional persisted critic_verdict.

    When ``verdict`` is not ``None`` a real CriticVerdict is persisted via
    ``record_verdict`` (issue #98), so the trace store holds the same
    ``VerdictRecord`` payload the production path writes.

    Issue #120 adds the optional ``injection_block_count`` parameter: when
    >0, that many ``injection_blocked`` events are planted so the per-
    session KPI counter has something to surface.

    Issue #705 adds the optional ``failure_class`` parameter: when provided,
    it is stored in the ``CriticVerdict`` so the KPI aggregation can
    compute ``failure_class_distribution``.

    Issue #800 adds ``hook_registry_error`` and ``wall_clock_abort`` parameters
    to plant the corresponding trace events for KPI computation.

    Issue #872 adds ``tool_argument_parse_error_count``: when >0, that many
    ``tool_argument_parse_error`` events are planted so the KPI aggregation
    can surface the count emitted by the runner at
    ``src/foundry_x/execution/runner.py:1684``.

    Issue #953 adds ``generation_exhausted_count``: when >0, that many
    ``generation_exhausted`` events are planted so the KPI aggregation can
    surface the evolver LLM failure count and rate.

    Issue #1112 adds ``token_budget_abort``: when provided as
    ``(tokens_used, token_budget)``, plants a ``task_aborted(reason="token_budget")``
    event with those values so the KPI aggregation can compute the overrun percentage.
    Issue #1113 adds ``event_limit_abort`` parameter to plant ``task_aborted``
    events with reason="event_limit" for cycle-time exclusion breakdown testing.
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
                    failure_class=failure_class,
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
        if hook_registry_error:
            logger.record(
                sid,
                kind="hook_registry_error",
                payload={"error_type": "ImportError", "message": "test error"},
            )
        if wall_clock_abort:
            logger.record(
                sid,
                kind="task_aborted",
                payload={"reason": "wall_clock", "timeout_s": 1.0, "token_budget": None},
            )
        if token_budget_abort is not None:
            tokens_used, token_budget = token_budget_abort
            logger.record(
                sid,
                kind="task_aborted",
                payload={
                    "reason": "token_budget",
                    "tokens_used": tokens_used,
                    "token_budget": token_budget,
                },
            )
        if event_limit_abort:
            logger.record(
                sid,
                kind="task_aborted",
                payload={"reason": "event_limit", "event_limit": 100},
            )
        for i in range(tool_argument_parse_error_count):
            logger.record(
                sid,
                kind="tool_argument_parse_error",
                payload={
                    "step": i,
                    "call_id": f"call-{i}",
                    "name": "read_file",
                    "raw": "not-json",
                    "error": "JSONDecodeError: malformed",
                },
            )
        for i in range(generation_exhausted_count):
            logger.record(
                sid,
                kind="generation_exhausted",
                payload={
                    "max_retries": 2,
                    "final_error": f"generation failed: {i}",
                },
            )
    return sid


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
                    "token_usage": usage,
                    "tokens_used": running,
                },
            )
    return sid


def _seed_context_pruned(
    logger: TraceLogger,
    harness_version: str,
    prune_count: int = 0,
) -> str:
    """Create a session with ``context_pruned`` events (issue #626).

    When ``prune_count`` > 0, that many ``context_pruned`` events are planted
    so the per-session KPI counter has something to surface.
    """
    from foundry_x.observability.kpis import CONTEXT_PRUNED_KIND

    with logger.session(harness_version=harness_version) as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "do work"})
        for i in range(prune_count):
            logger.record(
                sid,
                kind=CONTEXT_PRUNED_KIND,
                payload={"dropped": i + 1, "threshold": 200},
            )
    return sid


def test_main_markdown_renders_excluded_from_cycle_time_when_present(tmp_path, capsys):
    """``foundry-kpis`` renders the exclusion count when > 0 (issue #895)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", verdict=None)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    assert "Excluded From Cycle Time" in captured.out
    assert "1 session(s)" in captured.out


def test_main_markdown_omits_excluded_from_cycle_time_when_clean(tmp_path, capsys):
    """A clean store (no exclusions) keeps the summary compact."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    assert "Excluded From Cycle Time" not in captured.out


def test_main_comparison_renders_excluded_from_cycle_time_row(tmp_path, capsys):
    """The baseline/candidate table includes an exclusion-count row."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=True, passed_checks=["bench"])
    _seed_session(logger, "v2", verdict=None)

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

    row = next(line for line in captured.out.splitlines() if "Excluded From Cycle Time" in line)
    # Baseline 0, candidate 1, delta +1 marked negative (lower is better).
    assert "0 | 1 | +1.00 (negative)" in row


# ---------------------------------------------------------------------------
# Issue #1113: ``excluded_from_cycle_time`` breakdown by abort reason:
# ``excluded_wall_clock``, ``excluded_token_budget``, ``excluded_event_limit``,
# and ``excluded_other``.  ``_cycle_time()`` inspects the ``reason`` field of
# ``task_aborted`` events to attribute each excluded session to a category.
# ---------------------------------------------------------------------------


def test_main_markdown_renders_exclusion_breakdown_table(tmp_path, capsys):
    """``foundry-kpis`` renders the per-abort-reason breakdown table (issue #1113)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", wall_clock_abort=True)
    _seed_session(logger, "v1", token_budget_abort=(5000, 10000))

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    assert "Excluded From Cycle Time" in captured.out
    assert "wall_clock" in captured.out
    assert "token_budget" in captured.out
    assert "event_limit" in captured.out
    assert "other" in captured.out
    assert "| wall_clock | 1 |" in captured.out
    assert "| token_budget | 1 |" in captured.out


def test_main_comparison_renders_exclusion_breakdown_rows(tmp_path, capsys):
    """The baseline/candidate table renders per-abort-reason exclusion rows (issue #1113)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)
    _seed_session(logger, "v1", wall_clock_abort=True)
    _seed_session(logger, "v2", verdict=True)
    _seed_session(logger, "v2", token_budget_abort=(5000, 10000))

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

    assert "Excl. wall_clock" in captured.out
    assert "Excl. token_budget" in captured.out
    assert "Excl. event_limit" in captured.out
    assert "Excl. other" in captured.out


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
# Issue #705: per-entry ``failure_class_distribution`` aggregated from
# ``critic_verdict`` events that carry a ``failure_class``.  The
# distribution is surfaced in both the default markdown/JSON output and in
# the ``--from-history`` trend table.
# ---------------------------------------------------------------------------


def test_main_renders_failure_class_distribution_in_markdown(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, failure_class="tool-error")
    _seed_session(logger, "v1", verdict=True, failure_class="wrong-tool")

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Failure Class Distribution" in output
    assert "tool-error" in output
    assert "wrong-tool" in output
    assert "2 verdict(s) across 2 class(es)" in output


def test_main_omits_failure_class_distribution_when_empty(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)  # no failure_class

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Failure Class Distribution" not in captured.out


def test_main_json_includes_failure_class_distribution(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, failure_class="state-leak")

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["failure_class_distribution"] == {"state-leak": 1}


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
    # ``server_restart_count`` is the issue #899 auxiliary metric.
    # ``excluded_from_cycle_time`` is the issue #895 cycle-time coverage
    # signal (sessions with task_received but no usable critic_verdict).
    # ``per_skill`` / ``per_task_family`` / ``per_difficulty_tier`` are the
    # issue #898 slice fields (empty unless ``--group-by`` is supplied).
    # ``per_model_id`` / ``per_quantization`` / ``per_harness_version``
    # are the issue #1039 session-level slice fields.
    assert set(payload.keys()) == {
        "cycle_time_seconds",
        "regression_rate",
        "improvement_rate",
        "injection_blocks",
        "token_totals",
        "evolver_duration_ms",
        "hooks_disabled_count",
        "hooks_disabled_rate",
        "token_budget_abort_count",
        "token_budget_hit_rate",
        "token_budget_overrun_pct",
        "context_efficiency",
        "streaming_quality",
        "context_pruned_count",
        "wall_clock_abort_count",
        "failure_class_distribution",
        "model_retry_count",
        "tool_argument_parse_error_count",
        "event_limit_abort_count",
        "server_restart_count",
        "model_cost_count",
        "model_rate_limit_count",
        "fetch_blocked_count",
        "total_model_cost_usd",
        "excluded_from_cycle_time",
        "excluded_wall_clock",
        "excluded_token_budget",
        "excluded_event_limit",
        "excluded_other",
        "evolver_llm_failure_count",
        "evolver_llm_failure_rate",
        "per_skill",
        "per_task_family",
        "per_difficulty_tier",
        "per_model_id",
        "per_quantization",
        "per_harness_version",
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
    assert set(payload.keys()) == {
        "baseline",
        "candidate",
        "deltas",
        "baseline_session_count",
        "candidate_session_count",
        "slice_deltas",
    }
    assert payload["baseline"]["improvement_rate"] == 1.0
    assert payload["candidate"]["improvement_rate"] == 0.0
    assert payload["deltas"]["improvement_rate"] == -1.0
    assert payload["baseline_session_count"] == 1
    assert payload["candidate_session_count"] == 1


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
# Issue #621: --cycle-time-alert-threshold exits non-zero when
# cycle_time_seconds exceeds the configured value. The exit message names
# the triggering KPI and value. Both thresholds can be specified
# simultaneously (regression_rate and cycle_time).
# Issue #626: per-session ``context_pruned_count`` is surfaced by the
# ``foundry-kpis`` CLI when ≥1 session has ≥1 prune. A clean trace store
# stays compact (no extra rows in the markdown table).
# ---------------------------------------------------------------------------


def test_main_markdown_renders_context_pruned_section(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_context_pruned(logger, "v1", prune_count=3)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Context Pruned" in output
    assert "3 prune(s) across 1 session(s)" in output
    assert "| Session | context_pruned |" in output
    assert s1 in output
    assert "| 3 |" in output


def test_main_markdown_omits_context_pruned_when_clean(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=0)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    assert "Context Pruned" not in captured.out


def test_main_json_includes_context_pruned_count(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    s1 = _seed_context_pruned(logger, "v1", prune_count=2)
    s2 = _seed_context_pruned(logger, "v1", prune_count=1)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["context_pruned_count"] == {s1: 2, s2: 1}
    assert sum(payload["context_pruned_count"].values()) == 3


def test_main_json_format_emits_context_pruned_in_top_level_keys(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=1)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()

    assert rc == 0
    payload = json.loads(captured.out)
    assert "context_pruned_count" in payload


# Issue #951: context_efficiency KPI is computed from context_pruned events.
# Formula: 1 - (sum(dropped) / sum(threshold + dropped)) per session, mean across sessions.
# ---------------------------------------------------------------------------


def test_main_markdown_renders_context_efficiency(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=1)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0

    output = captured.out
    assert "Context Efficiency" in output


def test_main_json_includes_context_efficiency(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_context_pruned(logger, "v1", prune_count=1)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert "context_efficiency" in payload
    assert payload["context_efficiency"] is not None


# Issue #621: --cycle-time-alert-threshold exits non-zero when
# cycle_time_seconds exceeds the configured value. The exit message names
# the triggering KPI and value. Both thresholds can be specified
# simultaneously (regression_rate and cycle_time).
# ---------------------------------------------------------------------------


def test_main_markdown_renders_parse_error_section(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, tool_argument_parse_error_count=4)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()

    assert rc == 0
    output = captured.out
    assert "Tool Argument Parse Errors" in output
    assert "4 malformed tool-call argument(s) emitted by the runner." in output


def test_main_markdown_omits_parse_error_section_when_clean(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True)

    rc = main(["--db", str(db)])
    captured = capsys.readouterr()
    assert rc == 0
    # Clean store → no Parse Errors section, mirroring wall-clock aborts.
    assert "Tool Argument Parse Errors" not in captured.out


def test_main_json_includes_tool_argument_parse_error_count(tmp_path, capsys):
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, tool_argument_parse_error_count=6)

    rc = main(["--db", str(db), "--format", "json"])
    captured = capsys.readouterr()
    assert rc == 0

    payload = json.loads(captured.out)
    assert payload["tool_argument_parse_error_count"] == 6


def test_main_comparison_renders_parse_error_row(tmp_path, capsys):
    """The comparison markdown surfaces a Tool Argument Parse Error row (issue #872)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["bench"])
    _seed_session(
        logger,
        "v2",
        verdict=True,
        passed_checks=["bench"],
        tool_argument_parse_error_count=2,
    )

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

    rows = captured.out.splitlines()
    parse_error_row = next(
        line for line in rows if line.lstrip().startswith("| Tool Argument Parse Error Count")
    )
    # 0 baseline, 2 candidate → delta +2 (negative/bad because
    # higher-is-better=False for parse-error count).
    assert "0 | 2 | +2.00 (negative)" in parse_error_row


# ---------------------------------------------------------------------------
# Issue #898: per-skill / per-task-family / per-difficulty-tier slices.
# ---------------------------------------------------------------------------


def test_main_validate_metadata_cli_exit_0(tmp_path, capsys):
    """``foundry-kpis --validate-metadata`` exits 0 and prints validation table."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])

    rc = main(
        [
            "--db",
            str(db),
            "--validate-metadata",
        ]
    )
    captured = capsys.readouterr()
    assert rc == 0
    assert "Task Metadata Validation" in captured.out or captured.out == ""


def test_main_validate_metadata_cli_with_harness_version(tmp_path, capsys):
    """``foundry-kpis --validate-metadata --harness-version`` filters by harness version."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_session(logger, "v1", verdict=True, passed_checks=["task_a"])
    _seed_session(logger, "v2", verdict=True, passed_checks=["task_a"])

    rc = main(
        [
            "--db",
            str(db),
            "--harness-version",
            "v1",
            "--validate-metadata",
        ]
    )
    assert rc == 0


# -- Issue #1039: session-level KPI slices --------------------------------
