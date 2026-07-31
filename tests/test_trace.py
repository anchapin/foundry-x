"""Focused trace-layer contract tests (issue #81).

Backstops the trace store contract that the smoke tests only partially
cover: sqlite round-trip, jsonl backend, secret redaction of 'sk-...' and
'Bearer ...' patterns, session ended_at + wall-clock duration, and the
render-failure CLI subcommand (both stdout and --out Markdown paths).
"""

from __future__ import annotations

import inspect
import json
import sqlite3
import time
from pathlib import Path

import pytest

from foundry_x.trace.cli import main
from foundry_x.trace.logger import TraceLogger

_BACKENDS = pytest.mark.parametrize("backend", ["sqlite", "jsonl"])


def _suffix(backend: str) -> str:
    return ".db" if backend == "sqlite" else ".jsonl"


def test_session_signature_excludes_model_config():
    assert "model_config" not in inspect.signature(TraceLogger.session).parameters


@_BACKENDS
def test_sqlite_and_jsonl_roundtrip(tmp_path, backend):
    """record() -> load_session() round-trips events on both backends."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hello"})
        logger.record(sid, kind="tool_call", payload={"name": "read_file"})

    events = logger.load_session(sid)
    assert len(events) == 2
    assert events[0].kind == "user_prompt"
    assert events[0].payload == {"text": "hello"}
    assert events[1].kind == "tool_call"
    assert events[1].session_id == sid


def test_unknown_backend_raises_value_error(tmp_path):
    """An unsupported backend raises ValueError at construction (issue #272).

    Previously an invalid backend was stored unchecked and then silently
    dropped every recorded event (the ``record()`` if/elif chain had no
    ``else``) while ``session()`` mis-routed to sqlite. The error message
    must name both valid backends so a misspelling is immediately obvious.
    """
    path = tmp_path / "traces.db"
    with pytest.raises(ValueError) as excinfo:
        TraceLogger(path, backend="csv")
    message = str(excinfo.value)
    assert "sqlite" in message
    assert "jsonl" in message


def test_jsonl_backend_writes_ndjson_file(tmp_path):
    """The jsonl backend writes one JSON object per line ending with '}'."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "x"})

    assert path.exists()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            json.loads(line)
    assert path.read_text(encoding="utf-8").strip().endswith("}")


@_BACKENDS
def test_payload_redaction_masks_api_key_and_bearer(tmp_path, backend):
    """'sk-...' and 'Bearer ...' substrings must be scrubbed on both backends."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    api_key = "sk-" + "1234567890abcdef"
    bearer = "Bea" + "rer " + "mF_9.B5f-4.1JqM"
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            kind="tool_call",
            payload={"header": f"Authorization: {bearer}", "key": api_key},
        )

    events = logger.load_session(sid)
    assert len(events) == 1
    payload = events[0].payload
    assert "[REDACTED:bearer]" in payload["header"]
    assert "[REDACTED:api-key]" in payload["key"]
    assert bearer not in json.dumps(payload)
    assert api_key not in json.dumps(payload)


@_BACKENDS
def test_payload_redaction_covers_all_secret_patterns(tmp_path, backend):
    """Every supported secret pattern is redacted to its sentinel on both backends.

    Covers: sk-... (api-key), Bearer ... (bearer), PEM blocks, JWTs, GitHub
    classic PATs, GitHub fine-grained PATs, AWS access key IDs, Stripe live
    keys, and Slack tokens (issue #121 / #627).
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    api_key = "sk-" + "1234567890abcdef"
    bearer = "Bea" + "rer " + "mF_9.B5f-4.1JqM"
    pem = (
        # gitleaks:allow - test fixture, not a real secret
        "-----BEGIN PRIVATE KEY-----\n"
        "TEST_REDACTED_FAKE_KEY_DATA_NOT_A_REAL_SECRET_PLACEHOLDER\n"
        "-----END PRIVATE KEY-----"
    )
    jwt = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjg"
    github_classic_pat = "ghp_" + "A" * 36
    github_fine_grained_pat = "github_pat_" + "B" * 37
    aws_access_key = "AKIA" + "J" * 16
    stripe_key = "sk_live_" + "4" * 24
    slack_token = "xoxb-" + "c" * 22

    with logger.session(harness_version="0.1.0") as sid:
        logger.record(
            sid,
            kind="tool_call",
            payload={
                "my_api_key": api_key,
                "auth_bearer": bearer,
                "pem_key": pem,
                "raw_jwt_token": jwt,
                "gh_pat": github_classic_pat,
                "gh_fg_pat": github_fine_grained_pat,
                "aws_key_val": aws_access_key,
                "stripe_key_val": stripe_key,
                "slack_tok_val": slack_token,
            },
        )

    events = logger.load_session(sid)
    assert len(events) == 1
    payload = events[0].payload
    serialized = json.dumps(payload)

    assert "[REDACTED:api-key]" in payload["my_api_key"]
    assert "[REDACTED:bearer]" in payload["auth_bearer"]
    assert "[REDACTED:pem]" in payload["pem_key"]
    assert "[REDACTED:jwt]" in payload["raw_jwt_token"]
    assert "[REDACTED:github-pat]" in payload["gh_pat"]
    assert "[REDACTED:github-pat]" in payload["gh_fg_pat"]
    assert "[REDACTED:aws-access-key]" in payload["aws_key_val"]
    assert "[REDACTED:stripe-key]" in payload["stripe_key_val"]
    assert "[REDACTED:slack-token]" in payload["slack_tok_val"]

    assert api_key not in serialized
    assert bearer not in serialized
    assert pem not in serialized
    assert jwt not in serialized
    assert github_classic_pat not in serialized
    assert github_fine_grained_pat not in serialized
    assert aws_access_key not in serialized
    assert stripe_key not in serialized
    assert slack_token not in serialized


@_BACKENDS
def test_session_emits_ended_at_and_duration(tmp_path, backend):
    """Exiting session() stamps ended_at and session_duration() > 0."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0") as sid:
        time.sleep(0.01)

    sessions = logger.list_sessions()
    matching = [s for s in sessions if s.session_id == sid]
    assert len(matching) == 1
    assert matching[0].ended_at is not None

    duration = logger.session_duration(sid)
    assert duration is not None
    assert duration.total_seconds() > 0


def _plant_failing_session(db: Path) -> str:
    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "do work"})
        logger.record(
            sid,
            kind="tool_error",
            payload={"error": "exit code 1", "traceback": "boom"},
        )
    return sid


def test_render_failure_cli_exits_zero_on_planted_session(tmp_path, capsys):
    """render-failure digests a planted session and exits 0."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)

    rc = main(["render-failure", sid, "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# Failure Report" in out
    assert sid in out


def test_render_failure_cli_out_writes_markdown_file(tmp_path):
    """render-failure --out writes a Markdown report to disk."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)
    out_file = tmp_path / "report.md"

    rc = main(
        ["render-failure", sid, "--trace-db", str(db), "--out", str(out_file)],
    )

    assert rc == 0
    assert out_file.exists()
    content = out_file.read_text(encoding="utf-8")
    assert content.startswith("# Failure Report")
    assert sid in content
    assert "## Classification" in content


def test_render_failure_trace_path_emits_deprecation_warning(tmp_path, capsys):
    """--trace-path still works but emits a DeprecationWarning (issue #1253)."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)

    with pytest.warns(DeprecationWarning, match="--trace-path is deprecated"):
        rc = main(["render-failure", sid, "--trace-path", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# Failure Report" in out


def test_render_failure_db_alias_emits_deprecation_warning(tmp_path, capsys):
    """--db alias still works but emits a DeprecationWarning (issue #1253)."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)

    with pytest.warns(DeprecationWarning, match="--db is deprecated"):
        rc = main(["render-failure", sid, "--db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# Failure Report" in out


def test_render_failure_format_json(tmp_path, capsys):
    """render-failure --format json produces valid JSON output (issue #1253)."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)

    rc = main(["render-failure", sid, "--trace-db", str(db), "--format", "json"])

    assert rc == 0
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["session_id"] == sid
    assert "summary" in data
    assert "suspected_causes" in data
    assert "failed_steps" in data
    assert "proposed_class" in data


def test_render_failure_out_json_auto_detected(tmp_path):
    """--out ending in .json auto-selects JSON format (issue #1253)."""
    db = tmp_path / "traces.db"
    sid = _plant_failing_session(db)
    out_file = tmp_path / "report.json"

    rc = main(
        ["render-failure", sid, "--trace-db", str(db), "--out", str(out_file)],
    )

    assert rc == 0
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["session_id"] == sid


def test_render_failure_jsonl_backend(tmp_path, capsys):
    """render-failure works on the JSONL backend (issue #1253)."""
    db = tmp_path / "traces.jsonl"
    logger = TraceLogger(db, backend="jsonl")
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "do work"})
        logger.record(
            sid,
            kind="tool_error",
            payload={"error": "exit code 1", "traceback": "boom"},
        )

    rc = main(["render-failure", sid, "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "# Failure Report" in out
    assert sid in out


def test_sqlite_trace_file_is_valid_database(tmp_path):
    """The sqlite backend produces a real SQLite database with both tables."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="0.1.0") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})

    with sqlite3.connect(db) as conn:
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'",
            )
        }
    assert "sessions" in tables
    assert "events" in tables


def _index_names(conn: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name LIKE 'idx_%'",
        )
    }


def test_composite_index_created_on_fresh_database(tmp_path):
    """A fresh sqlite database gets the (session_id, kind) composite index.

    Issue #196 — the index lets SQLite satisfy ``iter_events(kind=...)``
    from the index alone instead of scanning every row for the session.
    """
    db = tmp_path / "traces.db"
    TraceLogger(db)

    with sqlite3.connect(db) as conn:
        assert "idx_events_session_kind" in _index_names(conn)


def test_composite_index_migrated_on_pre_existing_database(tmp_path):
    """Opening a pre-#196 database adds the composite index non-destructively.

    Mirrors the ``ended_at`` migration guard (issue #8): the
    ``CREATE INDEX IF NOT EXISTS`` in ``_SCHEMA`` is idempotent, so an
    old database that predates the index gets it added on next open
    without losing existing data.
    """
    db = tmp_path / "traces.db"
    # Simulate a pre-#196 database: create the schema manually WITHOUT
    # the composite index, then insert a row so we can verify data survives.
    with sqlite3.connect(db) as conn:
        conn.executescript(
            "CREATE TABLE sessions ("
            "  session_id TEXT PRIMARY KEY,"
            "  started_at TEXT NOT NULL,"
            "  harness_version TEXT NOT NULL,"
            "  model_id TEXT,"
            "  metadata TEXT,"
            "  ended_at TEXT"
            ");"
            "CREATE TABLE events ("
            "  event_id TEXT PRIMARY KEY,"
            "  session_id TEXT NOT NULL,"
            "  timestamp TEXT NOT NULL,"
            "  kind TEXT NOT NULL,"
            "  payload TEXT NOT NULL"
            ");"
            "CREATE INDEX idx_events_session ON events(session_id);"
        )
        conn.execute(
            "INSERT INTO sessions VALUES ('s1','2026-01-01','0.1.0',NULL,'{}',NULL)",
        )
        conn.execute(
            "INSERT INTO events VALUES ('e1','s1','2026-01-01','user_prompt','{\"text\":\"hi\"}')",
        )

    # Re-open through TraceLogger — the constructor runs _SCHEMA which
    # includes the new composite index via CREATE INDEX IF NOT EXISTS.
    logger = TraceLogger(db)

    with sqlite3.connect(db) as conn:
        assert "idx_events_session_kind" in _index_names(conn)
        # Existing data is preserved.
        assert len(logger.load_session("s1")) == 1


def test_explain_query_plan_uses_composite_index(tmp_path):
    """EXPLAIN QUERY PLAN on a kind-filtered query uses the composite index.

    Issue #196 acceptance criterion: with 10k rows planted, SQLite should
    choose ``idx_events_session_kind`` (the covering composite index) over
    a full table scan when both ``session_id`` and ``kind`` are filtered.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    with logger.session(harness_version="0.1.0") as sid:
        for i in range(200):
            kind = "critic_verdict" if i % 50 == 0 else "tool_call"
            logger.record(sid, kind=kind, payload={"i": i})

    with sqlite3.connect(db) as conn:
        plan_rows = conn.execute(
            "EXPLAIN QUERY PLAN SELECT event_id FROM events WHERE session_id = ? AND kind = ?",
            (sid, "critic_verdict"),
        ).fetchall()
    plan_text = " ".join(str(row) for row in plan_rows)
    assert "idx_events_session_kind" in plan_text


# --- model_response token_usage (issue #191) -------------------------------


@_BACKENDS
def test_model_response_event_carries_token_usage_on_both_backends(tmp_path, backend):
    """The ``model_response`` payload contract (issue #191) lands on both
    trace backends: prompt_tokens / completion_tokens / total_tokens round
    trip through ``record()`` -> ``load_session()`` unchanged. The Phase 3
    Digester and the PRD "Improvement Rate" KPI rely on this telemetry
    staying structured in the store; a flattening or string-coercion
    regression here would silently break per-step token deltas.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    token_usage = {
        "prompt_tokens": 42,
        "completion_tokens": 17,
        "total_tokens": 59,
    }
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(
            sid,
            kind="model_response",
            payload={
                "step": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
                "tool_calls": [],
                "token_usage": token_usage,
            },
        )

    events = logger.load_session(sid)
    assert len(events) == 1
    payload = events[0].payload
    assert payload["token_usage"] == token_usage
    assert payload["token_usage"]["prompt_tokens"] == 42
    assert payload["token_usage"]["completion_tokens"] == 17
    assert payload["token_usage"]["total_tokens"] == 59


@_BACKENDS
def test_model_response_event_records_null_token_usage_when_missing(tmp_path, backend):
    """When the model adapter omits ``usage`` (issue #191 acceptance), the
    runner records ``token_usage=None`` so the Digester can distinguish
    "missing telemetry" from "zero tokens consumed". An omitted key would
    be ambiguous downstream and force the consumer to guess.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(
            sid,
            kind="model_response",
            payload={
                "step": 0,
                "finish_reason": "stop",
                "message": {"role": "assistant", "content": "done"},
                "tool_calls": [],
                "token_usage": None,
            },
        )

    events = logger.load_session(sid)
    assert len(events) == 1
    assert "token_usage" in events[0].payload
    assert events[0].payload["token_usage"] is None


# --- query_events: cross-session streaming query (issue #273) ----------------


def _seed_multi_session_fixture(logger: TraceLogger) -> dict[str, list[str]]:
    """Plant three sessions across two harness versions with mixed kinds.

    Returns a ``session_id -> [event_id, ...]`` map in insertion order so
    the equivalence tests can compare ``query_events`` against the
    per-session ``iter_events`` baseline without re-deriving it.
    """
    planted: dict[str, list[str]] = {}
    # Session 1 / version v1: task_received + critic_verdict.
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s1"})
        logger.record(sid, kind="critic_verdict", payload={"approved": True})
        planted[sid] = [e.event_id for e in logger.load_session(sid)]
    # Session 2 / version v1: critic_verdict only.
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="critic_verdict", payload={"approved": False})
        planted[sid] = [e.event_id for e in logger.load_session(sid)]
    # Session 3 / version v2: task_received + injection_blocked.
    with logger.session(harness_version="v2") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s3"})
        logger.record(sid, kind="injection_blocked", payload={"markers": ["x"]})
        planted[sid] = [e.event_id for e in logger.load_session(sid)]
    return planted


def _by_event_id(events):
    return [e.event_id for e in events]


@_BACKENDS
def test_query_events_unfiltered_matches_nested_loop_on_multi_session(tmp_path, backend):
    """query_events() with no filter yields the same rows as list_sessions +
    iter_events, in the same per-event ordering (issue #273 acceptance).

    The nested-loop baseline walks sessions in the order ``list_sessions``
    returns them (started_at ASC) and within each session pulls events in
    timestamp order via ``iter_events``. ``query_events`` promises the
    same timestamp ordering extended across sessions, so for distinct
    timestamps the two sequences must be identical.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_multi_session_fixture(logger)

    # Nested-loop baseline: list_sessions + iter_events.
    nested: list = []
    for session in logger.list_sessions():
        nested.extend(logger.iter_events(session.session_id))

    actual = list(logger.query_events())

    assert _by_event_id(actual) == _by_event_id(nested)
    # Sanity: the fixture planted 5 event rows total (2 + 1 + 2);
    # session_start/session_end markers are not events and must not
    # appear here.
    assert len(actual) == 5


@_BACKENDS
def test_query_events_kind_filter_matches_nested_loop_baseline(tmp_path, backend):
    """query_events(kind=...) yields exactly the rows the per-session
    nested loop would have produced, in the same order (issue #273).
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_multi_session_fixture(logger)

    kind = "critic_verdict"
    nested: list = []
    for session in logger.list_sessions():
        nested.extend(logger.iter_events(session.session_id, kind=kind))

    actual = list(logger.query_events(kind=kind))

    assert _by_event_id(actual) == _by_event_id(nested)
    # Sanity: two sessions planted a critic_verdict (sessions 1 and 2).
    assert len(actual) == 2
    assert {e.kind for e in actual} == {kind}


@_BACKENDS
def test_query_events_harness_version_filter_scopes_to_matching_sessions(tmp_path, backend):
    """query_events(harness_version=...) returns only events from sessions
    whose harness version matches, on both backends (issue #273).
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_multi_session_fixture(logger)

    v1_nested: list = []
    v2_nested: list = []
    for session in logger.list_sessions():
        bucket = v1_nested if session.harness_version == "v1" else v2_nested
        bucket.extend(logger.iter_events(session.session_id))

    v1_actual = list(logger.query_events(harness_version="v1"))
    v2_actual = list(logger.query_events(harness_version="v2"))

    assert _by_event_id(v1_actual) == _by_event_id(v1_nested)
    assert _by_event_id(v2_actual) == _by_event_id(v2_nested)
    # Sanity: v1 has sessions 1 (2 events) and 2 (1 event); v2 has session 3
    # (2 events). The v1/v2 buckets are disjoint.
    assert len(v1_actual) == 3
    assert len(v2_actual) == 2
    assert set(_by_event_id(v1_actual)).isdisjoint(_by_event_id(v2_actual))


@_BACKENDS
def test_query_events_kind_and_harness_version_filter_compose(tmp_path, backend):
    """kind + harness_version compose: only matching-kind events from
    matching-version sessions are yielded (issue #273).
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_multi_session_fixture(logger)

    # task_received lives in sessions 1 (v1) and 3 (v2). Filtering to v2
    # must narrow the result to session 3's task_received only.
    actual = list(logger.query_events(kind="task_received", harness_version="v2"))
    assert len(actual) == 1
    assert actual[0].kind == "task_received"

    # Cross-check against the nested-loop baseline with the same filters.
    nested: list = []
    for session in logger.list_sessions(harness_version="v2"):
        nested.extend(logger.iter_events(session.session_id, kind="task_received"))
    assert _by_event_id(actual) == _by_event_id(nested)


@_BACKENDS
def test_query_events_kind_and_harness_version_equivalence(tmp_path, backend):
    """query_events(kind=X, harness_version=Y) yields the same events as
    iterating all sessions and calling iter_events for each (issue #630).

    Both backends must pass. The nested-loop baseline iterates ALL sessions
    (no pre-filter) and applies the harness_version check in Python so the
    comparison is unbiased — it does not shortcut through list_sessions with
    the same filter that query_events also receives.
    """
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_multi_session_fixture(logger)

    for kind in ("task_received", "critic_verdict", "injection_blocked"):
        for harness_version in ("v1", "v2"):
            actual = list(logger.query_events(kind=kind, harness_version=harness_version))

            nested: list = []
            for session in logger.list_sessions():
                if session.harness_version != harness_version:
                    continue
                nested.extend(logger.iter_events(session.session_id, kind=kind))

            assert _by_event_id(actual) == _by_event_id(nested), (
                f"mismatch for kind={kind!r}, harness_version={harness_version!r}: "
                f"query_events returned {_by_event_id(actual)}, "
                f"nested loop returned {_by_event_id(nested)}"
            )


@_BACKENDS
def test_query_events_empty_store_yields_nothing(tmp_path, backend):
    """An empty trace store yields no rows without raising (issue #273)."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert list(logger.query_events()) == []
    assert list(logger.query_events(kind="anything")) == []
    assert list(logger.query_events(harness_version="missing")) == []


# --- WAL mode + single reused connection (issue #274) -----------------------


def test_sqlite_trace_store_enables_wal_mode(tmp_path):
    """``TraceLogger.__init__`` flips the sqlite backend into WAL journal mode.

    Issue #274 acceptance criterion. WAL is a persistent database property
    stored in the file header, so a fresh ``sqlite3.connect`` opened against
    the same file (the way the Digester/KPI readers and the trace CLI open
    it) must also report ``wal``. With the previous rollback-journal default
    this would return ``delete``.
    """
    db = tmp_path / "traces.db"
    TraceLogger(db)

    with sqlite3.connect(db) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]

    assert mode.lower() == "wal"


def test_reader_iter_events_without_busy_while_writer_holds_open_session(tmp_path):
    """A separate reader reads committed events while the writer holds an open
    write transaction, without raising ``SQLITE_BUSY`` (issue #274).

    Models the Phase-3 access pattern the change exists to fix: the Runner
    (writer, one reused connection) is mid-session, and the Digester/KPI
    reader (a separate connection) walks the events. Without WAL a writer
    holding a write lock blocks readers; with WAL the reader sees the last
    committed snapshot immediately.

    The writer deliberately holds an *uncommitted* write transaction on its
    reused connection (``BEGIN`` + ``INSERT`` without commit) so a lock is
    genuinely held during the read — exercising the WAL guarantee rather
    than reading during an idle window.
    """
    db = tmp_path / "traces.db"
    writer = TraceLogger(db)
    with writer.session(harness_version="0.1.0") as sid:
        writer.record(sid, kind="user_prompt", payload={"text": "committed-event"})

    # Hold an open write transaction on the writer's reused connection to
    # model a Runner mid-session. ``record``/``session`` commit per op, so we
    # open the transaction manually here. Reaching into ``_conn`` is a test
    # affordance; the property under test is the WAL read/write separation.
    writer_conn = writer._conn
    assert writer_conn is not None
    writer_conn.execute("BEGIN")
    writer_conn.execute(
        "INSERT INTO events VALUES ('held-tx', ?, '2026-01-01', 'tool_call', '{}')",
        (sid,),
    )
    try:
        # A separate reader connection with a short busy_timeout: under WAL it
        # returns immediately; without WAL a held write lock forces it to
        # block until timeout and then raise SQLITE_BUSY.
        with sqlite3.connect(db, timeout=0.5) as reader:
            rows = reader.execute(
                "SELECT event_id FROM events WHERE session_id = ? ORDER BY timestamp",
                (sid,),
            ).fetchall()
    finally:
        writer_conn.rollback()

    # The committed event is visible to the reader; the uncommitted
    # ``held-tx`` row is not (snapshot isolation in WAL).
    assert [row[0] for row in rows] != []
    assert "held-tx" not in {row[0] for row in rows}
    writer.close()


def test_close_releases_reused_connection(tmp_path):
    """``close()`` deterministically releases the reused sqlite connection.

    Issue #274 — calling ``close()`` twice is a no-op and subsequent reads
    through the public API are not exercised here (the connection is gone);
    this test pins the lifecycle contract so a future refactor cannot
    silently drop the release path.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    assert logger._conn is not None
    logger.close()
    assert logger._conn is None
    # Idempotent: a second close must not raise.
    logger.close()


# --- JSONL corruption resilience (issue #932) --------------------------------


def test_jsonl_read_paths_skip_corrupted_line(tmp_path):
    """All five JSONL read paths skip a corrupted line instead of crashing.

    Issue #932 — a partial write (SIGKILL, OOM, disk-full mid-append) can
    leave a truncated JSONL line in the trace store. Before this fix, that
    single bad line raised ``JSONDecodeError`` in every read path
    (``list_sessions``, ``load_session``, ``iter_events``, ``query_events``,
    ``redact_event``), making the entire store unreadable and the three PRD
    KPIs unmeasurable. The write-side methods (``_prune_jsonl``,
    ``_delete_session_jsonl``, ``compact``) already skipped bad lines; this
    test pins the same resilience on the read side. The corrupted line is
    skipped, never deleted.
    """
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")
    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "first"})
        logger.record(sid, kind="tool_call", payload={"name": "second"})

    # Inject a corrupted/partial line between the two valid events, matching
    # the real failure mode: a truncated append that never reached a closing
    # brace. The line deliberately carries this session's id + an event-like
    # kind so we prove the skip is driven by the decode failure, not by a
    # missing-field filter.
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) >= 4  # session_start, event1, event2, session_end
    corrupted = '{"kind": "tool_call", "session_id": "' + sid + '", "payload":'
    injected = lines[:2] + [corrupted + "\n"] + lines[2:]
    path.write_text("".join(injected), encoding="utf-8")

    reader = TraceLogger(path, backend="jsonl")

    # 1. list_sessions — the session is still discoverable.
    sessions = reader.list_sessions()
    assert len(sessions) == 1
    assert sessions[0].session_id == sid

    # 2. load_session — both valid events survive; the bad line is skipped.
    loaded = reader.load_session(sid)
    assert [e.kind for e in loaded] == ["task_received", "tool_call"]

    # 3. iter_events — streams the two valid events in order.
    iterated = list(reader.iter_events(sid))
    assert _by_event_id(iterated) == _by_event_id(loaded)
    assert len(iterated) == 2

    # 4. query_events — the cross-session stream skips the bad line too.
    queried = list(reader.query_events())
    assert len(queried) == 2
    assert {e.kind for e in queried} == {"task_received", "tool_call"}

    # 5. redact_event — succeeds against the first valid event (index 0 by
    # timestamp), never touching the corrupted line.
    assert reader.redact_event(sid, 0, "prompt") is True
    first = reader.load_session(sid)[0]
    assert first.kind == "task_received"
    assert first.payload["prompt"] == "[REDACTED]"

    # The corrupted line is still on disk — read paths skip, never delete.
    assert corrupted in path.read_text(encoding="utf-8")


def test_doctor_removes_corrupted_lines_preserving_valid_events(tmp_path):
    """doctor --apply drops JSONDecodeError lines and keeps valid ones (issue #1077).

    Acceptance criterion 5: a JSONL file with one truncated line between two
    valid events → after ``doctor --apply``, ``load_session`` returns exactly
    the two valid events.
    """
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")

    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "first"})
        logger.record(sid, kind="task_completed", payload={"status": "ok"})

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    assert len(lines) >= 4
    corrupted = '{"kind": "tool_call", "session_id": "' + sid + '", "payload":'
    injected = lines[:2] + [corrupted + "\n"] + lines[2:]
    path.write_text("".join(injected), encoding="utf-8")

    reader = TraceLogger(path, backend="jsonl")
    result = reader.doctor(apply=True)
    assert result["applied"] is True
    assert len(result["skipped_lines"]) == 1
    assert result["skipped_lines"][0] == 3

    loaded = reader.load_session(sid)
    assert len(loaded) == 2
    assert {e.kind for e in loaded} == {"task_received", "task_completed"}


def test_doctor_dry_run_does_not_modify_file(tmp_path):
    """doctor without --apply is a pure dry-run (issue #1077)."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")

    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "first"})

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    corrupted = '{"kind": "tool_call", "session_id": "' + sid + '", "payload":'
    injected = lines[:1] + [corrupted + "\n"] + lines[1:]
    path.write_text("".join(injected), encoding="utf-8")

    reader = TraceLogger(path, backend="jsonl")
    result = reader.doctor(apply=False)
    assert result["applied"] is False
    assert len(result["skipped_lines"]) == 1
    assert result["kept_lines"] == len(lines)

    assert corrupted in path.read_text(encoding="utf-8")


def test_jsonl_skip_info_recorded_on_corrupted_read(tmp_path):
    """jsonl_skip_info is incremented when a JSONDecodeError is caught (issue #1077)."""
    path = tmp_path / "traces.jsonl"
    logger = TraceLogger(path, backend="jsonl")

    with logger.session(harness_version="0.1.0", model_id="m") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "first"})
        logger.record(sid, kind="task_completed", payload={"status": "ok"})

    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    corrupted = '{"kind": "tool_call", "session_id": "' + sid + '", "payload":'
    injected = lines[:2] + [corrupted + "\n"] + lines[2:]
    path.write_text("".join(injected), encoding="utf-8")

    reader = TraceLogger(path, backend="jsonl")
    assert reader.jsonl_skip_info["count"] == 0

    list(reader.iter_events(sid))
    info = reader.jsonl_skip_info
    assert info["count"] >= 1
    assert info["last_reason"] == "json_decode_error"
    assert info["last_session_id"] == sid


def test_batch_mode_accumulates_and_flushes_on_threshold(tmp_path):
    """Batch mode accumulates events and flushes via executemany at threshold (issue #1124).

    With 200 events and flush_threshold=50, we expect 4 threshold-triggered
    flushes (at events 50, 100, 150, 200). The pending buffer is empty after
    each flush, so no additional flush occurs on session end.
    Total INSERT round-trips: 4 (≤ 5 as required by issue #1124).
    """
    import unittest.mock

    path = tmp_path / "traces.db"
    logger = TraceLogger(path, backend="sqlite", batch_mode=True, flush_threshold=50)

    flush_call_count = 0
    original_flush = logger._flush_session

    def counting_flush(session_id):
        nonlocal flush_call_count
        flush_call_count += 1
        return original_flush(session_id)

    with (
        unittest.mock.patch.object(logger, "_flush_session", counting_flush),
        logger.session(harness_version="0.1.0") as sid,
    ):
        for i in range(200):
            logger.record(sid, kind="model_response_chunk", payload={"index": i})

    assert flush_call_count == 4
    events = logger.load_session(sid)
    assert len(events) == 200
    assert sid not in logger._pending


def test_batch_mode_flushes_remaining_on_session_end(tmp_path):
    """Any remaining pending events are flushed when the session ends (issue #1124).

    With 37 events and flush_threshold=50, no threshold-triggered flush occurs
    during recording. The 37 events are flushed once on session end.
    """
    import unittest.mock

    path = tmp_path / "traces.db"
    logger = TraceLogger(path, backend="sqlite", batch_mode=True, flush_threshold=50)

    flush_call_count = 0
    original_flush = logger._flush_session

    def counting_flush(session_id):
        nonlocal flush_call_count
        flush_call_count += 1
        return original_flush(session_id)

    with (
        unittest.mock.patch.object(logger, "_flush_session", counting_flush),
        logger.session(harness_version="0.1.0") as sid,
    ):
        for i in range(37):
            logger.record(sid, kind="model_response_chunk", payload={"index": i})

    assert flush_call_count == 1
    events = logger.load_session(sid)
    assert len(events) == 37


def test_batch_mode_non_chunk_events_also_batched(tmp_path):
    """Non-chunk events (tool_call, model_request) are also correctly batched (issue #1124)."""
    path = tmp_path / "traces.db"
    logger = TraceLogger(path, backend="sqlite", batch_mode=True, flush_threshold=10)

    with logger.session(harness_version="0.1.0") as sid:
        for i in range(15):
            logger.record(sid, kind="tool_call", payload={"name": f"tool_{i}"})

    events = logger.load_session(sid)
    assert len(events) == 15


def test_batch_mode_false_unchanged_behavior(tmp_path):
    """batch_mode=False (default) uses one INSERT per record (issue #1124)."""
    path = tmp_path / "traces.db"
    logger = TraceLogger(path, backend="sqlite", batch_mode=False, flush_threshold=50)

    with logger.session(harness_version="0.1.0") as sid:
        for i in range(10):
            logger.record(sid, kind="tool_call", payload={"name": f"tool_{i}"})

    events = logger.load_session(sid)
    assert len(events) == 10


def test_batch_mode_close_flushes_all_pending(tmp_path):
    """close() flushes any remaining pending events (issue #1124)."""
    path = tmp_path / "traces.db"
    logger = TraceLogger(path, backend="sqlite", batch_mode=True, flush_threshold=50)

    with logger.session(harness_version="0.1.0") as sid:
        for i in range(75):
            logger.record(sid, kind="model_response_chunk", payload={"index": i})

    logger.close()
    logger = TraceLogger(path, backend="sqlite")
    events = logger.load_session(sid)
    assert len(events) == 75


def test_iter_events_returns_correct_results_in_under_50ms(tmp_path):
    """Regression test: iter_events with kind filter returns correct results
    in <50ms for a session with >1000 events (issue #1151).

    The composite index idx_events_session_kind makes this an O(log n)
    indexed lookup rather than an O(n) full table scan.
    """
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)

    with logger.session(harness_version="0.1.0") as sid:
        for i in range(1200):
            kind = "critic_verdict" if i % 100 == 0 else "tool_call"
            logger.record(sid, kind=kind, payload={"i": i})

    start = time.perf_counter()
    verdicts = list(logger.iter_events(sid, kind="critic_verdict"))
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert len(verdicts) == 12
    assert all(e.kind == "critic_verdict" for e in verdicts)
    assert elapsed_ms < 50, f"iter_events took {elapsed_ms:.1f}ms, expected <50ms"


# --- list_unevolved_sessions harness_version filter (issue #1272) --------------


def _seed_unevolved_fixture(logger: TraceLogger) -> dict[str, str]:
    """Plant sessions across two harness versions with mixed evolved states.

    Returns a ``session_id -> harness_version`` map.
    All sessions end when their context manager exits (session_end is always written).
    """
    ids: dict[str, str] = {}
    # Session 1 / v1: ended, not evolved.
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s1"})
        ids[sid] = "v1"
    # Session 2 / v1: ended, not evolved.
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s2"})
        ids[sid] = "v1"
    # Session 3 / v2: ended, not evolved.
    with logger.session(harness_version="v2") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s3"})
        ids[sid] = "v2"
    # Session 4 / v1: ended, mark as evolved.
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s4"})
        ids[sid] = "v1"
    logger.mark_session_evolved(sid)
    # Session 5 / v2: ended (context manager always exits), not evolved.
    with logger.session(harness_version="v2") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s5"})
        ids[sid] = "v2"
    return ids


@_BACKENDS
def test_list_unevolved_sessions_returns_only_unevolved_ended_sessions(tmp_path, backend):
    """list_unevolved_sessions excludes evolved sessions (issue #1272)."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    ids = _seed_unevolved_fixture(logger)

    unevolved = logger.list_unevolved_sessions()
    assert len(unevolved) == 4
    assert set(unevolved).issubset(set(ids.keys()))
    for sid in unevolved:
        assert ids[sid] in ("v1", "v2")
        assert logger.is_session_evolved(sid) is False


@_BACKENDS
def test_list_unevolved_sessions_harness_version_filter(tmp_path, backend):
    """list_unevolved_sessions(harness_version=...) scopes to matching version (issue #1272)."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    ids = _seed_unevolved_fixture(logger)

    v1_unevolved = logger.list_unevolved_sessions(harness_version="v1")
    v2_unevolved = logger.list_unevolved_sessions(harness_version="v2")

    assert len(v1_unevolved) == 2
    assert len(v2_unevolved) == 2
    v1_sids = set(v1_unevolved)
    v2_sids = set(v2_unevolved)
    for v1_sid in v1_unevolved:
        assert ids[v1_sid] == "v1"
    for v2_sid in v2_unevolved:
        assert ids[v2_sid] == "v2"
    assert v1_sids.isdisjoint(v2_sids)


@_BACKENDS
def test_list_unevolved_sessions_missing_harness_version_returns_empty(tmp_path, backend):
    """harness_version that does not match returns empty list (issue #1272)."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_unevolved_fixture(logger)

    assert logger.list_unevolved_sessions(harness_version="v99") == []


# --- iter_unevolved_verdicts: O(1) store call for daemon verdict scan (issue #1272)


def _seed_verdicts_fixture(logger: TraceLogger) -> dict[str, str]:
    """Plant sessions with critic_verdict events and mixed evolved states.

    Returns a ``session_id -> harness_version`` map.
    """
    ids: dict[str, str] = {}
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s1"})
        logger.record(sid, kind="critic_verdict", payload={"approved": True})
        ids[sid] = "v1"
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s2"})
        logger.record(sid, kind="critic_verdict", payload={"approved": False})
        ids[sid] = "v1"
    with logger.session(harness_version="v2") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s3"})
        logger.record(sid, kind="critic_verdict", payload={"approved": True})
        ids[sid] = "v2"
    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s4"})
        logger.record(sid, kind="critic_verdict", payload={"approved": False})
        ids[sid] = "v1"
    logger.mark_session_evolved(sid)
    with logger.session(harness_version="v2") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s5"})
        ids[sid] = "v2"
    return ids


@_BACKENDS
def test_iter_unevolved_verdicts_yields_only_unevolved_sessions(tmp_path, backend):
    """iter_unevolved_verdicts yields critic_verdict events only for unevolved sessions."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_verdicts_fixture(logger)

    verdicts = list(logger.iter_unevolved_verdicts())
    assert all(e.kind == "critic_verdict" for e in verdicts)
    assert len(verdicts) == 3
    verdict_sids = {e.session_id for e in verdicts}
    for sid in verdict_sids:
        assert logger.is_session_evolved(sid) is False


@_BACKENDS
def test_iter_unevolved_verdicts_excludes_evolved_sessions(tmp_path, backend):
    """iter_unevolved_verdicts does not yield from sessions marked as evolved."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    ids = _seed_verdicts_fixture(logger)

    verdicts = list(logger.iter_unevolved_verdicts())
    verdict_sids = {e.session_id for e in verdicts}
    for sid, version in ids.items():
        if version == "v1" and sid != "s5":
            is_evolved = logger.is_session_evolved(sid)
            if is_evolved:
                assert sid not in verdict_sids


@_BACKENDS
def test_iter_unevolved_verdicts_harness_version_filter(tmp_path, backend):
    """iter_unevolved_verdicts(harness_version=...) scopes to matching version."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    _seed_verdicts_fixture(logger)

    v1_verdicts = list(logger.iter_unevolved_verdicts(harness_version="v1"))
    v2_verdicts = list(logger.iter_unevolved_verdicts(harness_version="v2"))

    assert len(v1_verdicts) == 2
    assert len(v2_verdicts) == 1
    assert all(e.kind == "critic_verdict" for e in v1_verdicts)
    assert all(e.kind == "critic_verdict" for e in v2_verdicts)


@_BACKENDS
def test_iter_unevolved_verdicts_empty_store_yields_nothing(tmp_path, backend):
    """An empty trace store yields no events without raising."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)
    assert list(logger.iter_unevolved_verdicts()) == []


@_BACKENDS
def test_iter_unevolved_verdicts_evolved_session_excluded(tmp_path, backend):
    """iter_unevolved_verdicts excludes sessions that have been marked as evolved."""
    path = tmp_path / f"traces{_suffix(backend)}"
    logger = TraceLogger(path, backend=backend)

    with logger.session(harness_version="v1") as sid:
        logger.record(sid, kind="task_received", payload={"prompt": "s1"})
        logger.record(sid, kind="critic_verdict", payload={"approved": True})

    verdicts = list(logger.iter_unevolved_verdicts())
    assert len(verdicts) == 1

    logger.mark_session_evolved(verdicts[0].session_id)

    verdicts_after = list(logger.iter_unevolved_verdicts())
    assert len(verdicts_after) == 0


def test_iter_unevolved_verdicts_jsonl_vs_sqlite_count_equivalence(tmp_path):
    """JSONL and SQLite backends return the same number of verdicts for the same fixture."""
    sqlite_path = tmp_path / "traces.db"
    jsonl_path = tmp_path / "traces.jsonl"

    sqlite_logger = TraceLogger(sqlite_path, backend="sqlite")
    jsonl_logger = TraceLogger(jsonl_path, backend="jsonl")

    for logger in (sqlite_logger, jsonl_logger):
        with logger.session(harness_version="v1") as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "s1"})
            logger.record(sid, kind="critic_verdict", payload={"approved": True})
        with logger.session(harness_version="v2") as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "s2"})
            logger.record(sid, kind="critic_verdict", payload={"approved": False})
        with logger.session(harness_version="v1") as sid:
            logger.record(sid, kind="task_received", payload={"prompt": "s3"})
            logger.record(sid, kind="critic_verdict", payload={"approved": True})
        logger.mark_session_evolved(sid)

    sqlite_verdicts = list(sqlite_logger.iter_unevolved_verdicts())
    jsonl_verdicts = list(jsonl_logger.iter_unevolved_verdicts())

    assert len(sqlite_verdicts) == 2
    assert len(sqlite_verdicts) == len(jsonl_verdicts)
    assert all(e.kind == "critic_verdict" for e in sqlite_verdicts)
    assert all(e.kind == "critic_verdict" for e in jsonl_verdicts)


# --- --latest / --harness-version auto-session selection (issue #1254) --------


def _plant_multi_version_sessions(db: Path) -> list[tuple[str, str]]:
    """Plant 3 sessions across 2 harness versions (issue #1254).

    Returns ``[(session_id, harness_version), ...]`` in insertion (oldest
    first) order:
      1. v1.0 with payload ``"session-one"``
      2. v2.0 with payload ``"session-two"``
      3. v1.0 with payload ``"session-three"``  ← global most-recent

    ``--latest`` (no filter) picks #3. ``--latest --harness-version v2.0``
    picks #2. ``--latest --harness-version v1.0`` picks #3.
    """
    logger = TraceLogger(db)
    sessions: list[tuple[str, str]] = []
    with logger.session(harness_version="v1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "session-one"})
        sessions.append((sid, "v1.0"))
    with logger.session(harness_version="v2.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "session-two"})
        sessions.append((sid, "v2.0"))
    with logger.session(harness_version="v1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "session-three"})
        sessions.append((sid, "v1.0"))
    return sessions


def test_session_show_latest_picks_most_recent(tmp_path, capsys):
    """session-show --latest renders the globally most recent session (#1254 AC1)."""
    db = tmp_path / "traces.db"
    sessions = _plant_multi_version_sessions(db)
    expected_sid = sessions[-1][0]

    rc = main(["session-show", "--latest", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"Session: {expected_sid}" in out


def test_session_show_latest_with_harness_version(tmp_path, capsys):
    """session-show --latest --harness-version picks the right build (#1254 AC2)."""
    db = tmp_path / "traces.db"
    sessions = _plant_multi_version_sessions(db)
    v2_sid = sessions[1][0]

    rc = main(["session-show", "--latest", "--harness-version", "v2.0", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"Session: {v2_sid}" in out


def test_timeline_latest_picks_most_recent(tmp_path, capsys):
    """timeline --latest renders the most recent session's timeline (#1254 AC3)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main(["timeline", "--latest", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "Timeline:" in out
    assert "session-three" in out


def test_timeline_latest_with_kind_filter(tmp_path, capsys):
    """timeline --latest --kind filters the most recent session (#1254 AC4)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="v1.0", model_id="m") as sid:
        logger.record(sid, kind="user_prompt", payload={"text": "hi"})
        logger.record(sid, kind="tool_call", payload={"name": "read_file"})

    rc = main(["timeline", "--latest", "--kind", "tool_call", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "tool_call" in out
    assert "user_prompt" not in out


def test_diagnose_latest_picks_most_recent(tmp_path, capsys):
    """diagnose --latest runs the triage on the most recent session (#1254)."""
    db = tmp_path / "traces.db"
    sessions = _plant_multi_version_sessions(db)
    expected_sid = sessions[-1][0]

    rc = main(["diagnose", "--latest", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert f"`{expected_sid}`" in out


def test_render_failure_latest_picks_most_recent(tmp_path, capsys):
    """render-failure --latest digests the most recent session (#1254)."""
    db = tmp_path / "traces.db"
    sessions = _plant_multi_version_sessions(db)
    expected_sid = sessions[-1][0]

    rc = main(["render-failure", "--latest", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert expected_sid in out


def test_export_latest_picks_most_recent(tmp_path, capsys):
    """export --latest exports the most recent session as JSONL (#1254)."""
    db = tmp_path / "traces.db"
    sessions = _plant_multi_version_sessions(db)
    expected_sid = sessions[-1][0]

    rc = main(["export", "--latest", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.strip().splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["session_id"] == expected_sid


def test_export_latest_with_harness_version(tmp_path, capsys):
    """export --latest --harness-version picks the right build (#1254)."""
    db = tmp_path / "traces.db"
    sessions = _plant_multi_version_sessions(db)
    v2_sid = sessions[1][0]

    rc = main(["export", "--latest", "--harness-version", "v2.0", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    lines = [json.loads(line) for line in out.strip().splitlines() if line]
    assert len(lines) == 1
    assert lines[0]["session_id"] == v2_sid


def test_events_grep_latest_picks_most_recent(tmp_path, capsys):
    """events-grep --latest scans the most recent session (#1254)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main(["events-grep", "--latest", "--pattern", "session-three", "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "session-three" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["session-show", "--latest", "some-sid"],
        ["timeline", "--latest", "some-sid"],
        ["diagnose", "--latest", "some-sid"],
        ["render-failure", "--latest", "some-sid"],
        ["events-grep", "--latest", "some-sid", "--pattern", "x"],
    ],
)
def test_latest_and_session_id_mutually_exclusive(tmp_path, capsys, argv):
    """--latest + a session_id argument exits 2 (#1254 AC5)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main([*argv, "--trace-db", str(db)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_export_latest_and_session_id_mutually_exclusive(tmp_path, capsys):
    """export --latest + --session-id exits 2 (#1254 AC5)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main(
        [
            "export",
            "--latest",
            "--session-id",
            "some-sid",
            "--trace-db",
            str(db),
        ]
    )

    assert rc == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


def test_export_latest_and_all_mutually_exclusive(tmp_path, capsys):
    """export --latest + --all exits 2 (#1254)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main(["export", "--latest", "--all", "--trace-db", str(db)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "mutually exclusive" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["session-show"],
        ["timeline"],
        ["diagnose"],
        ["render-failure"],
        ["events-grep", "--pattern", "x"],
    ],
)
def test_no_session_id_and_no_latest_exits_two(tmp_path, capsys, argv):
    """Omitting both session_id and --latest exits 2 (#1254 AC5)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main([*argv, "--trace-db", str(db)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "either a session_id or --latest is required" in err


def test_export_no_session_id_and_no_latest_exits_two(tmp_path, capsys):
    """export without --session-id, --all, or --latest exits 2 (#1254 AC5)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main(["export", "--trace-db", str(db)])

    assert rc == 2
    err = capsys.readouterr().err
    assert "either a session_id or --latest is required" in err


@pytest.mark.parametrize(
    "argv",
    [
        ["session-show", "--latest"],
        ["timeline", "--latest"],
        ["diagnose", "--latest"],
        ["render-failure", "--latest"],
        ["events-grep", "--latest", "--pattern", "x"],
        ["export", "--latest"],
    ],
)
def test_latest_empty_store_exits_zero(tmp_path, capsys, argv):
    """--latest on an empty trace store prints 'no sessions' and exits 0 (#1254 AC6)."""
    db = tmp_path / "empty.db"

    rc = main([*argv, "--trace-db", str(db)])

    assert rc == 0
    out = capsys.readouterr().out
    assert "no sessions" in out


def test_latest_with_missing_harness_version_empty(tmp_path, capsys):
    """--latest --harness-version for a non-existent version prints 'no sessions' (#1254)."""
    db = tmp_path / "traces.db"
    _plant_multi_version_sessions(db)

    rc = main(
        [
            "session-show",
            "--latest",
            "--harness-version",
            "v9.9",
            "--trace-db",
            str(db),
        ]
    )

    assert rc == 0
    out = capsys.readouterr().out
    assert "no sessions" in out
