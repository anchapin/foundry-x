"""KPI and session-card coverage for ``task_aborted(reason="event_limit")`` events (issue #869).

The runner records a ``task_aborted`` event with ``reason="event_limit"`` when
the per-session event cap (``FOUNDRY_MAX_EVENTS_PER_SESSION``) is exceeded —
see ``foundry_x.execution.runner._check_event_limit``. The KPI counter is a
session-aggregated scalar surfaced in the markdown table when non-zero and in
the JSON contract as ``event_limit_abort_count``.
"""

from __future__ import annotations

import json

from foundry_x.observability.cli import main as observability_main
from foundry_x.observability.kpis import KpiSummary, compare_kpis, compute_kpis
from foundry_x.observability.kpis import main as kpi_main
from foundry_x.observability.session_card import format_session_card
from foundry_x.trace.logger import TraceLogger


def _seed_event_limit_aborts(
    logger: TraceLogger,
    harness_version: str,
    count: int,
    *,
    include_task_received: bool = True,
) -> str:
    """Create one session with *count* production-shaped event-limit abort events.

    Mirrors the payload recorded by ``foundry_x.execution.runner._check_event_limit``
    (issue #869): ``{"reason": "event_limit", "event_count": int, "max_events_per_session": int}``.
    A ``task_received`` event is recorded first so the session registers as a
    real task the harness attempted to run.
    """
    with logger.session(harness_version=harness_version) as session_id:
        if include_task_received:
            logger.record(session_id, "task_received", {"prompt": "drive agent"})
        for cap_hit in range(count):
            logger.record(
                session_id,
                "task_aborted",
                {
                    "reason": "event_limit",
                    "event_count": 100 + cap_hit,
                    "max_events_per_session": 100,
                },
            )
    return session_id


def test_event_limit_abort_count_defaults_to_zero_and_round_trips(tmp_path):
    """A clean trace store yields zero, and the field survives pydantic round-trip."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_event_limit_aborts(logger, "v1", 0)

    summary = compute_kpis(logger)
    round_tripped = KpiSummary.model_validate(summary.model_dump())

    assert summary.event_limit_abort_count == 0
    assert round_tripped == summary


def test_event_limit_abort_count_aggregates_across_sessions(tmp_path):
    """Multiple sessions contribute to a single counter, independent of session count."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_event_limit_aborts(logger, "v1", 2)
    _seed_event_limit_aborts(logger, "v1", 3)
    _seed_event_limit_aborts(logger, "v1", 0)  # clean session contributes nothing

    summary = compute_kpis(logger)

    assert summary.event_limit_abort_count == 5


def test_event_limit_abort_count_respects_harness_version_filter(tmp_path):
    """The counter narrows to the requested harness version."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_event_limit_aborts(logger, "v1", 4)
    _seed_event_limit_aborts(logger, "v2", 7)

    assert compute_kpis(logger, harness_version="v1").event_limit_abort_count == 4
    assert compute_kpis(logger, harness_version="v2").event_limit_abort_count == 7
    # Unfiltered run aggregates both versions.
    assert compute_kpis(logger).event_limit_abort_count == 11


def test_event_limit_abort_count_ignores_other_abort_reasons(tmp_path):
    """A session whose task_aborted reason is NOT ``event_limit`` is excluded."""
    logger = TraceLogger(tmp_path / "traces.db")
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, "task_received", {"prompt": "do work"})
        logger.record(sid, "task_aborted", {"reason": "wall_clock", "timeout_s": 1.0})
        logger.record(sid, "task_aborted", {"reason": "token_budget", "token_budget": 1000})
    _seed_event_limit_aborts(logger, "v1", 2)

    summary = compute_kpis(logger)

    assert summary.event_limit_abort_count == 2
    # Wall-clock and token-budget counters are unaffected — they have their own
    # aggregation paths; this test pins that the new counter does not double-count.
    assert summary.wall_clock_abort_count == 1
    assert summary.token_budget_abort_count == 1


def test_event_limit_abort_count_is_surfaced_in_kpi_cli_markdown(tmp_path, capsys):
    """The ``foundry-kpis`` CLI renders an Event Limit Aborts section when count > 0."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_event_limit_aborts(logger, "v1", 3)

    assert kpi_main(["--db", str(db)]) == 0
    markdown = capsys.readouterr().out
    assert "Event Limit Aborts: 3 session(s) hit the per-session event cap." in markdown


def test_event_limit_abort_count_omits_markdown_section_when_clean(tmp_path, capsys):
    """A clean store stays compact — no Event Limit Aborts section is rendered."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_event_limit_aborts(logger, "v1", 0)

    assert kpi_main(["--db", str(db)]) == 0
    assert "Event Limit Aborts" not in capsys.readouterr().out


def test_event_limit_abort_count_is_surfaced_in_kpi_cli_json(tmp_path, capsys):
    """The JSON contract exposes ``event_limit_abort_count`` at the top level."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_event_limit_aborts(logger, "v1", 6)

    assert kpi_main(["--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["event_limit_abort_count"] == 6


def test_event_limit_abort_count_is_in_comparison_aggregates(tmp_path):
    """Baseline/candidate comparison exposes the new counter as a signed delta (issue #869)."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_event_limit_aborts(logger, "baseline", 1)
    _seed_event_limit_aborts(logger, "candidate", 4)

    comparison = compare_kpis(logger, "baseline", "candidate")

    assert comparison.baseline.event_limit_abort_count == 1
    assert comparison.candidate.event_limit_abort_count == 4
    assert comparison.deltas["event_limit_abort_count"] == 3


def test_event_limit_abort_count_appears_in_comparison_markdown(tmp_path, capsys):
    """The baseline/candidate CLI comparison renders the new row."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_event_limit_aborts(logger, "baseline", 1)
    _seed_event_limit_aborts(logger, "candidate", 2)

    rc = kpi_main(
        [
            "--db",
            str(db),
            "--baseline-harness-version",
            "baseline",
            "--candidate-harness-version",
            "candidate",
        ]
    )
    assert rc == 0

    markdown = capsys.readouterr().out
    assert "Event Limit Abort Count" in markdown
    # Both sides render as integers; the delta is signed with the
    # higher-is-better=False convention (more aborts → negative).
    assert "| 1 | 2 |" in markdown


def test_session_card_counts_event_limit_task_aborted_events(tmp_path):
    """The session card surfaces ``task_aborted`` events with ``reason="event_limit"``.

    The kind's ``"abort"`` substring matches ``_ERROR_KIND_RE`` in
    ``session_card.py``, so ``event_limit`` aborts flow into the
    ``errors_by_kind`` bucket as ``task_aborted=N`` without any
    card-specific code path (issue #869).
    """
    logger = TraceLogger(tmp_path / "traces.db")
    session_id = _seed_event_limit_aborts(logger, "v1", 2)
    session = next(item for item in logger.list_sessions() if item.session_id == session_id)

    card = format_session_card(session, logger.load_session(session_id))
    assert "task_aborted=2" in card


def test_session_card_cli_surfaces_event_limit_abort_count(tmp_path, capsys):
    """The ``fx-trace session-card`` CLI command surfaces the abort count via ``errors_by_kind``."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    session_id = _seed_event_limit_aborts(logger, "v1", 2)

    rc = observability_main(["session-card", "--db", str(db), "--session-id", session_id])
    assert rc == 0
    assert "task_aborted=2" in capsys.readouterr().out
