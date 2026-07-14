"""Cross-session outcome roll-up table (issue #184).

Acceptance pinned by this module:
  - ``fx-trace session-summary [--harness-version V] [--limit N]``
    prints a table with columns ``session_id  started_at  duration
    outcome.status  outcome.reason  steps`` for every recorded session.
  - Rows are newest-first by ``started_at``.
  - Sessions without an ``outcome`` event render ``_`` in the three
    trailing columns.
  - Empty store prints ``"no sessions"`` and exits 0.
  - ``--harness-version`` filters to a single harness build,
    ``--limit`` truncates after newest-first ordering.
  - The CLI does **not** import ``Digester`` or ``Critic``.
"""

from __future__ import annotations

import inspect

from foundry_x.observability.cli import main as cli_main
from foundry_x.observability.session_summary import (
    OUTCOME_KIND,
    SessionSummaryRow,
    build_session_summary,
    render_session_summary,
)
from foundry_x.trace.logger import TraceLogger

_FOUR_SESSIONS = [
    # (sid, harness_version, started_at, ended_at, outcome_payload | None)
    (
        "sess-0001-old",
        "0.1.0",
        "2026-07-10T10:00:00+00:00",
        "2026-07-10T10:00:05+00:00",
        {"status": "success", "reason": "final_answer", "steps": 2},
    ),
    (
        "sess-0002-mid",
        "0.1.0",
        "2026-07-10T11:00:00+00:00",
        "2026-07-10T11:00:30+00:00",
        {"status": "truncated", "reason": "max_steps", "steps": 12},
    ),
    # Session 3 was started but the runner raised before reaching the
    # terminal ``outcome`` event — the table must render underscores
    # in the three outcome-derived columns for it.
    (
        "sess-0003-no-outcome",
        "0.1.0",
        "2026-07-10T12:00:00+00:00",
        "2026-07-10T12:00:02+00:00",
        None,
    ),
    (
        "sess-0004-new",
        "0.2.0",
        "2026-07-10T13:00:00+00:00",
        "2026-07-10T13:01:00+00:00",
        {"status": "failed", "reason": "model_error", "steps": 4},
    ),
]


def _plant_four_sessions(db_path) -> None:
    """Plant four deterministic sessions (issue #184 acceptance test).

    Uses the underlying sqlite store directly so timestamps are
    deterministic. Three sessions carry an ``outcome`` event; one
    does not, to exercise the underscore-rendering acceptance
    criterion.
    """
    import json
    import sqlite3
    import uuid

    # Opening the logger creates the schema (and the ended_at column).
    TraceLogger(db_path)
    with sqlite3.connect(db_path) as conn:
        for sid, harness_version, started_at, ended_at, outcome_payload in _FOUR_SESSIONS:
            conn.execute(
                "INSERT INTO sessions "
                "(session_id, started_at, harness_version, model_id, metadata, ended_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (sid, started_at, harness_version, None, "{}", ended_at),
            )
            if outcome_payload is None:
                continue
            conn.execute(
                "INSERT INTO events (event_id, session_id, timestamp, kind, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    sid,
                    started_at,
                    OUTCOME_KIND,
                    json.dumps(outcome_payload),
                ),
            )


def test_build_session_summary_returns_one_row_per_session(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))

    assert len(rows) == 4
    assert all(isinstance(row, SessionSummaryRow) for row in rows)


def test_build_session_summary_orders_newest_first(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))

    started_at_values = [row.started_at for row in rows]
    # Newest-first: the list of started_at strings sorts descending.
    assert started_at_values == sorted(started_at_values, reverse=True)
    assert rows[0].session_id == "sess-0004-new"
    assert rows[-1].session_id == "sess-0001-old"


def test_build_session_summary_session_without_outcome_has_none_fields(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))

    no_outcome = next(row for row in rows if row.session_id == "sess-0003-no-outcome")
    assert no_outcome.outcome_status is None
    assert no_outcome.outcome_reason is None
    assert no_outcome.steps is None


def test_build_session_summary_filters_by_harness_version(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db), harness_version="0.2.0")

    assert [row.session_id for row in rows] == ["sess-0004-new"]


def test_render_session_summary_emits_header_and_underscores(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))
    rendered = render_session_summary(rows)

    lines = rendered.splitlines()
    # Header + 4 session rows.
    assert len(lines) == 5
    header = lines[0]
    for column in (
        "session_id",
        "started_at",
        "duration",
        "outcome.status",
        "outcome.reason",
        "steps",
    ):
        assert column in header

    no_outcome_line = next(ln for ln in lines[1:] if "sess-0003-no-outcome" in ln)
    assert no_outcome_line.count("_") >= 3


def test_render_session_summary_includes_recorded_outcome_fields(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))
    rendered = render_session_summary(rows)

    new_line = next(ln for ln in rendered.splitlines() if "sess-0004-new" in ln)
    assert "failed" in new_line
    assert "model_error" in new_line
    # ``steps`` is no longer the right-most column (token_budget_hit is);
    # we check that "4" appears in the correct position by looking for it
    # in the context of the steps column.
    assert "    4  false" in new_line


def test_render_session_summary_respects_limit(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))
    rendered = render_session_summary(rows, limit=2)

    data_lines = rendered.splitlines()[1:]
    assert len(data_lines) == 2
    # limit applies AFTER newest-first sort: rows 0 and 1 from the
    # full ordering are sess-0004-new and sess-0003-no-outcome.
    assert "sess-0004-new" in data_lines[0]
    assert "sess-0003-no-outcome" in data_lines[1]


def test_render_session_summary_empty_store_says_no_sessions():
    assert render_session_summary([]) == "no sessions"


def test_render_session_summary_long_outcome_is_truncated(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    rows = build_session_summary(TraceLogger(db))
    # Append a session whose ``reason`` exceeds the cell width to verify
    # the column-width contract — the rendered line must not grow past
    # the fixed width.
    long_reason = "x" * 80
    rows.append(
        SessionSummaryRow(
            session_id="sess-0099-long",
            started_at="2026-07-10T14:00:00+00:00",
            duration_seconds=1.0,
            outcome_status="success",
            outcome_reason=long_reason,
            steps=1,
        )
    )
    rendered = render_session_summary(rows)
    long_line = next(ln for ln in rendered.splitlines() if "sess-0099-long" in ln)
    # Each fixed-width cell is separated by two spaces. The right-most
    # ``token_budget_hit`` column ("_" for this row) stays at the end
    # regardless of truncation of the reason cell; the ``steps`` column
    # ("    1") is now second-to-last.
    assert "    1      _" in long_line


# --- CLI integration ---


def test_cli_session_summary_prints_table(tmp_path, capsys):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rc = cli_main(["session-summary", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "session_id" in out
    assert "started_at" in out
    assert "duration" in out
    assert "outcome.status" in out
    assert "outcome.reason" in out
    assert "steps" in out
    # All four session ids appear.
    for sid, *_ in _FOUR_SESSIONS:
        assert sid in out


def test_cli_session_summary_empty_store_prints_no_sessions(tmp_path, capsys):
    db = tmp_path / "traces.db"
    TraceLogger(db)

    rc = cli_main(["session-summary", "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "no sessions" in out


def test_cli_session_summary_respects_harness_version_filter(tmp_path, capsys):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rc = cli_main(["session-summary", "--db", str(db), "--harness-version", "0.2.0"])

    assert rc == 0
    out_lines = capsys.readouterr().out.splitlines()
    # Header + exactly one data row (the only 0.2.0 session).
    assert len(out_lines) == 2
    assert "sess-0004-new" in out_lines[1]
    # No other sessions leak through the harness-version filter.
    for sid, *_ in _FOUR_SESSIONS:
        if sid == "sess-0004-new":
            continue
        assert sid not in out_lines[1]


def test_cli_session_summary_respects_limit(tmp_path, capsys):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rc = cli_main(["session-summary", "--db", str(db), "--limit", "2"])

    assert rc == 0
    out_lines = [ln for ln in capsys.readouterr().out.splitlines() if ln.strip()]
    # Header + 2 data rows.
    assert len(out_lines) == 3


def test_cli_session_summary_does_not_import_digester_or_critic():
    """Issue #184 acceptance: the new subcommand must not pull in the
    evolution pipeline. Inspecting the rendered ``session-summary``
    branch in ``cli.py`` is the simplest, source-of-truth check that no
    Digester/Critic import was sneaked in for this view.
    """
    source = inspect.getsource(cli_main)
    branch = source.split('if args.command == "session-summary"', 1)[1]
    # Stop scanning at the next ``if args.command`` / ``return`` so a
    # later branch cannot accidentally satisfy the assertion.
    for marker in ('if args.command == "', "return 1"):
        idx = branch.find(marker)
        if idx != -1:
            branch = branch[:idx]
    assert "Digester" not in branch
    assert "Critic" not in branch


# ---------------------------------------------------------------------------
# Issue #466: token_budget_hit column in session-summary.
# ---------------------------------------------------------------------------


def _plant_session_with_token_budget_abort(db_path, sid, harness_version, started_at, ended_at):
    """Plant a session with both an outcome and a token_budget task_aborted event."""
    import json
    import sqlite3
    import uuid

    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "INSERT INTO sessions "
            "(session_id, started_at, harness_version, model_id, metadata, ended_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (sid, started_at, harness_version, None, "{}", ended_at),
        )
        conn.execute(
            "INSERT INTO events (event_id, session_id, timestamp, kind, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                sid,
                started_at,
                "outcome",
                json.dumps({"status": "failed", "reason": "token_budget", "steps": 3}),
            ),
        )
        conn.execute(
            "INSERT INTO events (event_id, session_id, timestamp, kind, payload) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                str(uuid.uuid4()),
                sid,
                started_at,
                "task_aborted",
                json.dumps({"reason": "token_budget", "tokens_used": 8000, "token_budget": 5000}),
            ),
        )


def test_build_session_summary_token_budget_hit_true_when_aborted(tmp_path):
    """``token_budget_hit`` is True when session has a token_budget task_aborted."""
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    _plant_session_with_token_budget_abort(
        db,
        "sess-0005-abort",
        "0.2.0",
        "2026-07-10T14:00:00+00:00",
        "2026-07-10T14:00:30+00:00",
    )

    rows = build_session_summary(TraceLogger(db))

    abort_row = next(row for row in rows if row.session_id == "sess-0005-abort")
    assert abort_row.token_budget_hit is True


def test_build_session_summary_token_budget_hit_false_when_no_abort(tmp_path):
    """``token_budget_hit`` is False when session has outcome but no token budget abort."""
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))

    success_row = next(row for row in rows if row.session_id == "sess-0001-old")
    assert success_row.token_budget_hit is False


def test_render_session_summary_shows_token_budget_hit_column(tmp_path):
    """Rendered output includes token_budget_hit values."""
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    _plant_session_with_token_budget_abort(
        db,
        "sess-0005-abort",
        "0.2.0",
        "2026-07-10T14:00:00+00:00",
        "2026-07-10T14:00:30+00:00",
    )

    rows = build_session_summary(TraceLogger(db))
    rendered = render_session_summary(rows)

    lines = rendered.splitlines()
    header = lines[0]
    assert "budget_hit" in header
    abort_line = next(ln for ln in lines if "sess-0005-abort" in ln)
    assert "true" in abort_line
