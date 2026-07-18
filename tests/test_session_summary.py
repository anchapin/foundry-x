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
    CONTEXT_PRUNED_KIND,
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
# Issue #626: per-session ``context_pruned`` count is surfaced in
# ``SessionSummaryRow`` and rendered in the session summary table.
# ---------------------------------------------------------------------------


def _plant_context_pruned_events(db_path, session_id, count):
    """Plant *count* ``context_pruned`` events for *session_id* (issue #626)."""
    import json
    import sqlite3
    import uuid

    with sqlite3.connect(db_path) as conn:
        for i in range(count):
            conn.execute(
                "INSERT INTO events (event_id, session_id, timestamp, kind, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    session_id,
                    "2026-07-10T10:00:00+00:00",
                    CONTEXT_PRUNED_KIND,
                    json.dumps({"dropped": i + 1, "threshold": 200}),
                ),
            )


def test_build_session_summary_context_pruned_is_none_when_no_prunes(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rows = build_session_summary(TraceLogger(db))

    for row in rows:
        assert row.context_pruned is None


def test_build_session_summary_context_pruned_count_populated(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    _plant_context_pruned_events(db, "sess-0001-old", 3)
    _plant_context_pruned_events(db, "sess-0002-mid", 1)

    rows = build_session_summary(TraceLogger(db))

    row_map = {row.session_id: row for row in rows}
    assert row_map["sess-0001-old"].context_pruned == 3
    assert row_map["sess-0002-mid"].context_pruned == 1
    assert row_map["sess-0003-no-outcome"].context_pruned is None
    assert row_map["sess-0004-new"].context_pruned is None


def test_build_session_summary_context_pruned_respects_harness_version_filter(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    _plant_context_pruned_events(db, "sess-0001-old", 2)
    _plant_context_pruned_events(db, "sess-0002-mid", 1)
    _plant_context_pruned_events(db, "sess-0004-new", 5)

    rows = build_session_summary(TraceLogger(db), harness_version="0.1.0")

    assert len(rows) == 3
    row_map = {row.session_id: row for row in rows}
    assert row_map["sess-0001-old"].context_pruned == 2
    assert row_map["sess-0002-mid"].context_pruned == 1


def test_render_session_summary_context_pruned_not_in_text_table(tmp_path):
    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    _plant_context_pruned_events(db, "sess-0001-old", 2)

    rows = build_session_summary(TraceLogger(db))
    rendered = render_session_summary(rows)

    lines = rendered.splitlines()
    header = lines[0]
    assert "context_pruned" not in header


# --- Issue #624: --format json and --out ---------------------------------------


def test_cli_session_summary_format_json_emits_json_array(tmp_path, capsys):
    """Issue #624 / #737: --format json emits a JSON object with failure_class_distribution and rows."""
    import json

    db = tmp_path / "traces.db"
    _plant_four_sessions(db)

    rc = cli_main(["session-summary", "--db", str(db), "--format", "json"])

    assert rc == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "failure_class_distribution" in parsed
    assert "rows" in parsed
    assert len(parsed["rows"]) == 4
    for row in parsed["rows"]:
        assert "session_id" in row
        assert "started_at" in row
        assert "duration_seconds" in row
        assert "outcome_status" in row
        assert "outcome_reason" in row
        assert "steps" in row


def test_cli_session_summary_out_writes_to_file(tmp_path, capsys):
    """Issue #624 / #737: --out writes to file; stdout is empty."""
    import json

    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    out_file = tmp_path / "summary.json"

    rc = cli_main(["session-summary", "--db", str(db), "--format", "json", "--out", str(out_file)])

    assert rc == 0
    # stdout is empty (no table printed).
    stdout = capsys.readouterr().out
    assert stdout == ""
    # File contains a single JSON object.
    content = out_file.read_text(encoding="utf-8")
    parsed = json.loads(content)
    assert "failure_class_distribution" in parsed
    assert "rows" in parsed
    assert len(parsed["rows"]) == 4


def test_cli_session_summary_format_json_infers_from_out_extension(tmp_path, capsys):
    """Issue #624 / #737: when --out ends in .json, json format is selected automatically."""
    import json

    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    out_file = tmp_path / "summary.json"

    rc = cli_main(["session-summary", "--db", str(db), "--out", str(out_file)])

    assert rc == 0
    content = out_file.read_text(encoding="utf-8")
    # Verify it's valid JSON by parsing.
    parsed = json.loads(content)
    assert "failure_class_distribution" in parsed
    assert "rows" in parsed
    assert len(parsed["rows"]) == 4


def test_cli_session_summary_format_json_with_limit(tmp_path, capsys):
    """Issue #624 / #737: --limit applies to JSON output (newest-first ordering)."""
    import json

    db = tmp_path / "traces.db"
    _plant_four_sessions(db)
    out_file = tmp_path / "summary.json"

    rc = cli_main(
        [
            "session-summary",
            "--db",
            str(db),
            "--format",
            "json",
            "--limit",
            "2",
            "--out",
            str(out_file),
        ]
    )

    assert rc == 0
    content = out_file.read_text(encoding="utf-8")
    parsed = json.loads(content)
    # Verify rows are newest-first.
    assert len(parsed["rows"]) == 2
    assert parsed["rows"][0]["session_id"] == "sess-0004-new"


# --- Issue #737: failure_class_distribution in session-summary --------------------


def _plant_sessions_with_verdicts(db_path):
    """Plant sessions with critic_verdict events that have failure_class."""
    import uuid

    from foundry_x.observability.regression_report import VERDICT_KIND

    TraceLogger(db_path)
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        sessions = [
            (
                "sess-fail-001",
                "0.1.0",
                "2026-07-15T10:00:00+00:00",
                "2026-07-15T10:00:10+00:00",
                {"status": "failed", "reason": "bad-prompt", "steps": 3},
                "bad-prompt",
            ),
            (
                "sess-fail-002",
                "0.1.0",
                "2026-07-15T11:00:00+00:00",
                "2026-07-15T11:00:15+00:00",
                {"status": "failed", "reason": "bad-prompt", "steps": 5},
                "bad-prompt",
            ),
            (
                "sess-fail-003",
                "0.1.0",
                "2026-07-15T12:00:00+00:00",
                "2026-07-15T12:00:08+00:00",
                {"status": "failed", "reason": "tool-error", "steps": 2},
                "tool-error",
            ),
        ]
        for sid, harness_version, started_at, ended_at, outcome_payload, failure_class in sessions:
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
                    OUTCOME_KIND,
                    __import__("json").dumps(outcome_payload),
                ),
            )
            verdict_payload = {
                "verdict": False,
                "passed_checks": [],
                "failed_checks": ["check1"],
                "notes": "",
                "failure_class": failure_class,
            }
            conn.execute(
                "INSERT INTO events (event_id, session_id, timestamp, kind, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    str(uuid.uuid4()),
                    sid,
                    started_at,
                    VERDICT_KIND,
                    __import__("json").dumps(verdict_payload),
                ),
            )


def test_build_session_summary_includes_failure_class(tmp_path):
    """Issue #737: build_session_summary populates failure_class from verdict records."""
    db = tmp_path / "traces.db"
    _plant_sessions_with_verdicts(db)

    rows = build_session_summary(TraceLogger(db))

    assert len(rows) == 3
    failure_classes = {row.failure_class for row in rows}
    assert failure_classes == {"bad-prompt", "tool-error"}


def test_failure_class_distribution_computed_from_rows(tmp_path):
    """Issue #737: _failure_class_distribution aggregates failure classes from rows."""
    from foundry_x.observability.session_summary import _failure_class_distribution

    db = tmp_path / "traces.db"
    _plant_sessions_with_verdicts(db)

    rows = build_session_summary(TraceLogger(db))
    distribution = _failure_class_distribution(rows)

    assert distribution == {"bad-prompt": 2, "tool-error": 1}


def test_cli_session_summary_json_includes_failure_class_distribution(tmp_path, capsys):
    """Issue #737: --format json output includes failure_class_distribution."""
    import json

    db = tmp_path / "traces.db"
    _plant_sessions_with_verdicts(db)

    rc = cli_main(["session-summary", "--db", str(db), "--format", "json"])
    assert rc == 0

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert "failure_class_distribution" in parsed
    assert parsed["failure_class_distribution"] == {"bad-prompt": 2, "tool-error": 1}


def test_cli_session_summary_markdown_includes_failure_class_distribution(tmp_path, capsys):
    """Issue #737: markdown output includes failure class breakdown."""
    db = tmp_path / "traces.db"
    _plant_sessions_with_verdicts(db)

    rc = cli_main(["session-summary", "--db", str(db)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "Failure Class Distribution" in out
    assert "bad-prompt" in out
    assert "tool-error" in out
