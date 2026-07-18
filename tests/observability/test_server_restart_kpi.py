"""KPI and CLI coverage for ``server_unavailable`` events (issue #899).

The runner emits ``server_unavailable`` whenever the
``FoundryServerManager`` reports a mid-session ``/health`` failure.
This counter aggregates the events into the ``server_restart_count``
field on :class:`~foundry_x.observability.kpis.KpiSummary`, surfaced
in the markdown table when non-zero and in the JSON contract as
``server_restart_count``.

Pattern mirrors the existing
``tests/observability/test_event_limit_kpi.py`` family.
"""

from __future__ import annotations

import json

from foundry_x.observability.cli import main as observability_main
from foundry_x.observability.kpis import KpiSummary, compare_kpis, compute_kpis
from foundry_x.observability.kpis import main as kpi_main
from foundry_x.trace.logger import TraceLogger


def _seed_server_restart(
    logger: TraceLogger,
    harness_version: str,
    count: int,
    *,
    include_task_received: bool = True,
) -> str:
    """Record *count* ``server_unavailable`` events in one harness-version session.

    Mirrors the payload recorded by
    ``foundry_x.execution.runner._handle_server_unavailable``: a
    ``task_received`` marker plus ``server_unavailable`` events with
    ``step``, ``host``, ``health_url``, ``restart_attempted``. A
    single session is created so the KPI counter aggregates per
    harness-version as expected.
    """
    with logger.session(harness_version=harness_version) as session_id:
        if include_task_received:
            logger.record(session_id, "task_received", {"prompt": "drive agent"})
        for i in range(count):
            logger.record(
                session_id,
                "server_unavailable",
                {
                    "step": i,
                    "host": "http://127.0.0.1:8080",
                    "health_url": "http://127.0.0.1:8080/health",
                    "restart_attempted": True,
                },
            )
    return session_id


def test_server_restart_count_defaults_to_zero_and_round_trips(tmp_path):
    """A clean trace store yields zero, and the field survives pydantic round-trip."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_server_restart(logger, "v1", 0)

    summary = compute_kpis(logger)
    round_tripped = KpiSummary.model_validate(summary.model_dump())

    assert summary.server_restart_count == 0
    assert round_tripped == summary


def test_server_restart_count_aggregates_across_sessions(tmp_path):
    """Multiple sessions contribute to a single counter, independent of session count."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_server_restart(logger, "v1", 2)
    _seed_server_restart(logger, "v1", 3)
    _seed_server_restart(logger, "v1", 0)  # clean session contributes nothing

    summary = compute_kpis(logger)

    assert summary.server_restart_count == 5


def test_server_restart_count_respects_harness_version_filter(tmp_path):
    """The counter narrows to the requested harness version."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_server_restart(logger, "v1", 4)
    _seed_server_restart(logger, "v2", 7)

    assert compute_kpis(logger, harness_version="v1").server_restart_count == 4
    assert compute_kpis(logger, harness_version="v2").server_restart_count == 7
    # Unfiltered run aggregates both versions.
    assert compute_kpis(logger).server_restart_count == 11


def test_server_restart_count_ignores_other_kinds(tmp_path):
    """Sessions with no ``server_unavailable`` events do not contribute."""
    logger = TraceLogger(tmp_path / "traces.db")
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, "task_received", {"prompt": "do work"})
        logger.record(sid, "model_retry", {"attempt": 1, "error_type": "ConnectError"})
        logger.record(sid, "model_error", {"step": 0, "error_type": "ValueError"})
    _seed_server_restart(logger, "v1", 2)

    summary = compute_kpis(logger)

    assert summary.server_restart_count == 2
    # Model-quality counters are unaffected — they have their own aggregation paths.
    assert summary.model_retry_count == 1


def test_server_restart_count_is_surfaced_in_kpi_cli_markdown(tmp_path, capsys):
    """The ``foundry-kpis`` CLI renders a Server Restarts section when count > 0."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_server_restart(logger, "v1", 3)

    assert kpi_main(["--db", str(db)]) == 0
    markdown = capsys.readouterr().out
    assert "Server Restarts: 3 server_unavailable event(s)" in markdown


def test_server_restart_count_omits_markdown_section_when_clean(tmp_path, capsys):
    """A clean store stays compact — no Server Restarts section is rendered."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_server_restart(logger, "v1", 0)

    assert kpi_main(["--db", str(db)]) == 0
    assert "Server Restarts" not in capsys.readouterr().out


def test_server_restart_count_is_surfaced_in_kpi_cli_json(tmp_path, capsys):
    """The JSON contract exposes ``server_restart_count`` at the top level."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_server_restart(logger, "v1", 6)

    assert kpi_main(["--db", str(db), "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["server_restart_count"] == 6


def test_server_restart_count_is_in_comparison_aggregates(tmp_path):
    """Baseline/candidate comparison exposes the new counter as a signed delta."""
    logger = TraceLogger(tmp_path / "traces.db")
    _seed_server_restart(logger, "baseline", 1)
    _seed_server_restart(logger, "candidate", 4)

    comparison = compare_kpis(logger, "baseline", "candidate")

    assert comparison.baseline.server_restart_count == 1
    assert comparison.candidate.server_restart_count == 4
    assert comparison.deltas["server_restart_count"] == 3


def test_server_restart_count_appears_in_comparison_markdown(tmp_path, capsys):
    """The baseline/candidate CLI comparison renders the new row."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    _seed_server_restart(logger, "baseline", 1)
    _seed_server_restart(logger, "candidate", 2)

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
    assert "Server Restart Count" in markdown
    # Both sides render as integers; the delta is signed with the
    # higher-is-better=False convention (more restarts → negative).
    assert "| 1 | 2 |" in markdown


def test_session_card_counts_server_unavailable_events(tmp_path):
    """The session card surfaces ``server_unavailable`` events via the
    generic ``errors_by_kind`` bucket.

    Mirrors how ``task_aborted`` events flow into the session card
    (see :mod:`tests.observability.test_event_limit_kpi`); the
    substring ``unavailable`` in the kind matches the card's error
    regex without requiring a card-specific code path.
    """
    logger = TraceLogger(tmp_path / "traces.db")
    from foundry_x.observability.session_card import format_session_card

    session_id = _seed_server_restart(logger, "v1", 2)
    session = next(item for item in logger.list_sessions() if item.session_id == session_id)

    card = format_session_card(session, logger.load_session(session_id))
    # The card may or may not surface the count by exact label; we
    # assert the substring is visible so the operator sees the signal.
    assert "server_unavailable" in card


def test_session_card_cli_surfaces_server_unavailable(tmp_path, capsys):
    """The ``fx-trace session-card`` CLI surfaces the new event kind."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    session_id = _seed_server_restart(logger, "v1", 2)

    rc = observability_main(["session-card", "--db", str(db), "--session-id", session_id])
    assert rc == 0
    # The card text contains the event kind; mirrors the contract used
    # by the event-limit card test.
    assert "server_unavailable" in capsys.readouterr().out
