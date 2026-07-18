from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from foundry_x.evolution.digester import Digester
from foundry_x.observability.render import render_failure_report
from foundry_x.observability.timeline import format_timeline
from foundry_x.trace.logger import TraceEvent, TraceLogger, TraceSession


def _render_failure(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.trace_path)
    events = logger.load_session(args.session_id)
    report = Digester().digest(args.session_id, events)
    markdown = render_failure_report(report)
    if args.out:
        Path(args.out).write_text(markdown, encoding="utf-8")
    else:
        sys.stdout.write(markdown + "\n")
    return 0


def _format_session_row(session: TraceSession) -> str:
    fields = [
        session.session_id,
        session.started_at,
        session.harness_version,
        session.model_id or "-",
    ]
    if session.ended_at:
        fields.append(session.ended_at)
    return "  ".join(fields)


def _sessions(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.db)
    sessions = logger.list_sessions()
    if not sessions:
        sys.stdout.write("No sessions found.\n")
        return 0
    sys.stdout.write("session_id  started_at  harness_version  model_id\n")
    for session in sessions:
        sys.stdout.write(_format_session_row(session) + "\n")
    return 0


def _show(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"No events found for session {args.session_id}.\n")
        return 1
    sys.stdout.write(f"Session: {args.session_id}\n")
    sys.stdout.write(f"Events: {len(events)}\n\n")
    sys.stdout.write(format_timeline(events) + "\n")
    return 0


def _serialize_event(event: TraceEvent) -> dict[str, object]:
    return {
        "event_id": event.event_id,
        "session_id": event.session_id,
        "timestamp": event.timestamp,
        "kind": event.kind,
        "payload": event.payload,
    }


def _export(args: argparse.Namespace) -> int:
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    lines = [json.dumps(_serialize_event(event)) for event in events]
    output = "\n".join(lines)
    if lines:
        output += "\n"
    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        sys.stdout.write(output)
    return 0


# --- Issue #83: session-list / session-show / events-grep --------------------
# The three subcommands here are the named-developer-surface promised by
# ADR-0007 §Consequences ('Traces must be inspectable'). They co-exist with
# the ``sessions``/``show``/``export`` commands above; consolidating them is
# out of scope for #83 and tracked separately.


def _format_session_list_row(session: TraceSession) -> str:
    """Render one row of the ``session-list`` table.

    Column order matches the issue #83 acceptance criteria:
    ``session_id  started_at  ended_at  harness_version``. ``ended_at``
    may be empty while a session is still open, so the field is always
    emitted as a placeholder column rather than conditionally appended.
    """
    ended = session.ended_at if session.ended_at is not None else "-"
    return "  ".join(
        [
            session.session_id,
            session.started_at,
            ended,
            session.harness_version,
        ]
    )


def _session_list(args: argparse.Namespace) -> int:
    """Implement ``session-list`` (issue #83).

    Lists sessions ordered as ``TraceLogger.list_sessions`` returns them.
    ``--harness-version`` narrows to sessions recorded against a specific
    harness build; ``--limit`` truncates after N rows. The command exits 0
    even when the database is empty so it composes cleanly in shell pipes.
    """
    logger = TraceLogger(args.db)
    sessions = logger.list_sessions()
    if args.harness_version is not None:
        sessions = [s for s in sessions if s.harness_version == args.harness_version]
    if args.limit is not None:
        sessions = sessions[: args.limit]
    sys.stdout.write("session_id  started_at  ended_at  harness_version\n")
    for session in sessions:
        sys.stdout.write(_format_session_list_row(session) + "\n")
    return 0


def _session_show(args: argparse.Namespace) -> int:
    """Implement ``session-show`` (issue #83).

    Reuses ``observability.timeline.format_timeline`` so the rendered
    timeline stays consistent with ``show``/``render-failure``. An
    unknown session returns exit code 1 with a message on stderr,
    mirroring ``_show`` and the grep convention.
    """
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"No events found for session {args.session_id}.\n")
        return 1
    sys.stdout.write(f"Session: {args.session_id}\n")
    sys.stdout.write(f"Events: {len(events)}\n\n")
    sys.stdout.write(format_timeline(events) + "\n")
    return 0


def _events_grep(args: argparse.Namespace) -> int:
    """Implement ``events-grep`` (issue #83).

    Scans every event payload in ``session_id`` for a regex match against
    the serialized JSON. Matching events are printed as one line each
    ``"<timestamp>  <kind>  <payload_json>"`` so the matched substring is
    visible in context and the output remains greppable. Exit codes
    follow the conventional grep semantics: 0 when at least one event
    matched, 1 when none matched. An invalid regex is re-raised after
    logging the parse error to stderr per the 'never silently swallow
    exceptions' rule in AGENTS.md.
    """
    logger = TraceLogger(args.db)
    events = logger.load_session(args.session_id)
    if not events:
        sys.stderr.write(f"No events found for session {args.session_id}.\n")
        return 1
    try:
        pattern = re.compile(args.pattern)
    except re.error as exc:
        sys.stderr.write(f"Invalid --pattern regex: {exc}\n")
        raise
    matches = 0
    for event in events:
        payload_text = json.dumps(event.payload, sort_keys=True)
        if pattern.search(payload_text):
            sys.stdout.write(f"{event.timestamp}  {event.kind}  {payload_text}\n")
            matches += 1
    return 0 if matches else 1


# --- Issue #192: redact-session / redact-key ---------------------------------
# These subcommands expose the post-write correction API (``delete_session``
# and ``redact_event`` from logger.py) so an on-call operator can respond to
# a secret leak within the SECURITY.md §Secrets response window without
# dropping into a Python REPL. Both accept ``--out`` to append a JSONL audit
# record so the remediation is traceable after the fact.


def _logger_for(db_path: str) -> TraceLogger:
    """Construct a :class:`TraceLogger` auto-detecting the backend.

    A ``--db`` path ending in ``.jsonl`` selects the JSONL backend; every
    other path (including the default ``logs/traces.db``) uses sqlite.
    Issue #192 requires both subcommands to work on either backend.
    """
    backend = "jsonl" if db_path.endswith(".jsonl") else "sqlite"
    return TraceLogger(db_path, backend=backend)


def _now_ts() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_audit(args: argparse.Namespace, record: dict[str, Any]) -> None:
    """Append a JSONL audit line to ``args.out`` if it was provided."""
    if not args.out:
        return
    with Path(args.out).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record) + "\n")


def _redact_session(args: argparse.Namespace) -> int:
    """Implement ``redact-session`` (issue #192).

    Counts the session's events (so the operator knows what was removed),
    deletes the session row and every event via ``TraceLogger.delete_session``
    (idempotent), prints the count, and optionally appends an audit record
    to ``--out``. Exits 0 even when the session did not exist, mirroring
    the idempotent contract of ``delete_session``.
    """
    logger = _logger_for(args.db)
    count = len(logger.load_session(args.session_id))
    logger.delete_session(args.session_id)
    sys.stdout.write(f"Deleted session {args.session_id}: {count} event(s) removed.\n")
    _write_audit(
        args,
        {
            "action": "redact-session",
            "session_id": args.session_id,
            "events_deleted": count,
            "timestamp": _now_ts(),
        },
    )
    return 0


def _redact_key(args: argparse.Namespace) -> int:
    """Implement ``redact-key`` (issue #192).

    Rewrites ``payload[key]`` to ``"[REDACTED]"`` on the timestamp-ordered
    event at ``event_index`` via ``TraceLogger.redact_event``. An
    out-of-range index returns exit code 1 immediately so a stale index
    never silently rewrites the wrong row.
    """
    logger = _logger_for(args.db)
    ok = logger.redact_event(args.session_id, args.event_index, args.key)
    if not ok:
        sys.stderr.write(
            f"redact-key: event_index {args.event_index} is out of range "
            f"for session {args.session_id}.\n"
        )
        return 1
    sys.stdout.write(
        f"Redacted key '{args.key}' on event {args.event_index} of session {args.session_id}.\n"
    )
    _write_audit(
        args,
        {
            "action": "redact-key",
            "session_id": args.session_id,
            "event_index": args.event_index,
            "key": args.key,
            "timestamp": _now_ts(),
        },
    )
    return 0


# --- Issue #275: delete-session / prune --------------------------------------
# These subcommands expose retention management for the trace store.  ADR-0003
# flags unbounded growth as the trigger to "revisit" the store; Phase-3's
# mandate of many real benchmark runs/day makes a prune CLI essential for
# keeping ``logs/`` under control and Digester/KPI queries fast.


def _delete_session(args: argparse.Namespace) -> int:
    """Implement ``delete-session`` (issue #275).

    Removes one session and all its events via ``TraceLogger.delete_session``.
    Idempotent: exits 0 whether or not the session existed, mirroring the
    contract of the underlying primitive.
    """
    logger = _logger_for(args.db)
    count = len(logger.load_session(args.session_id))
    logger.delete_session(args.session_id)
    sys.stdout.write(f"Deleted session {args.session_id}: {count} event(s) removed.\n")
    return 0


def _prune(args: argparse.Namespace) -> int:
    """Implement ``prune`` (issue #275).

    Two modes:
    * ``--keep-last N`` — retain only the N most recent sessions (by
      ``started_at``) and remove the rest.
    * ``--older-than DAYS`` — remove sessions whose ``started_at`` is
      older than the given number of days.

    ``--dry-run`` reports what *would* be removed without touching the
    store. Both modes work on sqlite and jsonl backends. Exits 0 on
    success, 1 on argument errors (neither / both flags given, or DAYS
    not a positive integer).
    """
    logger = _logger_for(args.db)
    sessions = list(logger.list_sessions())

    if args.keep_last is not None and args.older_than is not None:
        sys.stderr.write("prune: use --keep-last OR --older-than, not both.\n")
        return 1
    if args.keep_last is None and args.older_than is None:
        sys.stderr.write("prune: specify --keep-last or --older-than.\n")
        return 1

    if args.keep_last is not None:
        if args.keep_last < 0:
            sys.stderr.write("prune: --keep-last must be >= 0.\n")
            return 1
        to_delete = sessions[: max(0, len(sessions) - args.keep_last)]
    else:
        assert args.older_than is not None
        if args.older_than <= 0:
            sys.stderr.write("prune: --older-than must be a positive integer.\n")
            return 1
        cutoff = datetime.now(timezone.utc) - timedelta(days=args.older_than)
        to_delete = [s for s in sessions if datetime.fromisoformat(s.started_at) < cutoff]

    if not to_delete:
        sys.stdout.write("Nothing to prune.\n")
        return 0

    count = len(to_delete)
    if args.dry_run:
        for session in to_delete:
            sys.stdout.write(
                f"  would delete {session.session_id}  started_at={session.started_at}\n"
            )
        sys.stdout.write(f"Dry run: {count} session(s) would be deleted.\n")
        return 0

    logger.prune_sessions([session.session_id for session in to_delete])
    sys.stdout.write(f"Deleted {count} session(s).\n")
    return 0


def _compact(args: argparse.Namespace) -> int:
    """Implement ``compact`` (issue #632).

    Rewrites the JSONL file removing orphaned ``session_end`` markers — those
    whose ``session_id`` has no corresponding ``session_start`` marker.
    This can happen after a session is deleted and a stale ``session_end``
    marker is written afterward.

    ``--dry-run`` reports what would be removed without touching the file.
    Only works on the JSONL backend; exits 0 on sqlite with a no-op message.
    """
    logger = _logger_for(args.db)

    if logger.backend != "jsonl":
        sys.stdout.write("compact: jsonl backend required; sqlite VACUUM is automatic.\n")
        return 0

    if not logger.path.exists():
        sys.stdout.write("No orphaned markers found.\n")
        return 0

    kept: list[str] = []
    seen_starts: set[str] = set()
    orphaned: list[str] = []

    with logger.path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                kept.append(line)
                continue
            try:
                record = json.loads(stripped)
            except json.JSONDecodeError:
                kept.append(line)
                continue

            kind = record.get("kind")
            session_id = record.get("session_id")

            if kind == "session_start":
                if session_id not in seen_starts:
                    seen_starts.add(session_id)
                kept.append(line)
            elif kind == "session_end":
                if session_id not in seen_starts:
                    orphaned.append(line)
                else:
                    kept.append(line)
            else:
                kept.append(line)

    orphaned_count = len(orphaned)

    if args.dry_run:
        if orphaned:
            for line in orphaned:
                sys.stdout.write(f"  would remove: {line.strip()}\n")
        sys.stdout.write(f"Dry run: {orphaned_count} orphaned marker(s) would be removed.\n")
        return 0

    if orphaned_count == 0:
        sys.stdout.write("No orphaned markers found.\n")
        return 0

    with logger.path.open("w", encoding="utf-8") as fh:
        fh.writelines(kept)
    sys.stdout.write(f"Removed {orphaned_count} orphaned marker(s).\n")
    return 0


# --- Issue #195: seed-sample-trace -------------------------------------------
# Plant a deterministic, secret-free session so the KPI / regression-report
# CLIs have something to chew on without standing up llama-server. Mirrors
# the event vocabulary the runner actually emits (issue #89, #91), so the
# Digester sees a faithful shape.


# Hard-coded payload constants. Every value is a literal placeholder — no
# real tokens, PEM blocks, or other secret-shaped substrings (docs/SECURITY.md
# §Secrets). Issue #195 acceptance criterion: "planted payloads contain no
# real secrets".
_SEED_PROMPT = "Refactor the auth helper to drop the legacy token cache."
_SEED_TOOL_NAME = "read_file"
_SEED_TOOL_ARGUMENTS: dict[str, str] = {"path": "src/foundry_x/auth.py"}
_SEED_TOOL_OUTPUT = "def authenticate(user, password):\n    return False\n"
_SEED_TOOL_CALL_ID = "call-seed-0001"
_SEED_DURATION_MS = 12
_SEED_MODEL_ID = "seeded-llama-sample"
# Issue #271: deterministic token-usage figures planted on the
# ``model_response`` event so the KPI summary and timeline CLIs have token
# data to surface without a live llama-server. All literal placeholders.
_SEED_USAGE = {"prompt_tokens": 42, "completion_tokens": 18, "total_tokens": 60}
# Default harness version used when ``--harness-version`` is omitted. Picked
# to be obviously a synthetic seed (``seed-sample``) so real-run sessions
# don't collide with the planted one in regression reports.
_SEED_DEFAULT_HARNESS_VERSION = "seed-sample"
# Backend inferred from the ``--db`` suffix, mirroring ``observability/cli.py``.
_SQLITE_SUFFIX = ".db"


def _seed_sample_trace(args: argparse.Namespace) -> int:
    """Implement ``seed-sample-trace`` (issue #195).

    Plants one session in the trace store at ``args.db`` containing every
    event kind the :mod:`foundry_x.execution.runner` emits, so the Digester,
    KPI, and regression-report CLIs have realistic input without a live
    ``llama-server``. Returns 0 on success; emits the planted ``session_id``
    on stdout so a follow-up ``session-show`` or ``render-failure`` can pick
    it up. All planted payloads are literal placeholder text — never
    tokens, keys, or PEM blocks — per the redaction contract in
    ``docs/SECURITY.md`` §Secrets.
    """
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    backend = "jsonl" if db_path.suffix.lower() != _SQLITE_SUFFIX else "sqlite"
    logger = TraceLogger(db_path, backend=backend)
    harness_version = args.harness_version

    with logger.session(
        harness_version=harness_version,
        model_id=_SEED_MODEL_ID,
        metadata={"seed": "seed-sample-trace", "issue": 195},
    ) as session_id:
        logger.record(
            session_id,
            kind="task_received",
            payload={"prompt": _SEED_PROMPT},
        )
        logger.record(
            session_id,
            kind="user_prompt",
            payload={"content": _SEED_PROMPT, "tool_count": 1},
        )
        logger.record(
            session_id,
            kind="model_request",
            payload={"step": 0, "message_count": 1, "tool_count": 1},
        )
        logger.record(
            session_id,
            kind="model_response",
            payload={
                "step": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Reading auth.py first to understand the legacy cache.",
                },
                "tool_calls": [
                    {
                        "id": _SEED_TOOL_CALL_ID,
                        "function": {
                            "name": _SEED_TOOL_NAME,
                            "arguments": json.dumps(_SEED_TOOL_ARGUMENTS),
                        },
                    }
                ],
                # Issue #271: token-usage accounting so the KPI summary and
                # timeline renderers have a token surface to exercise.
                "usage": _SEED_USAGE,
                "tokens_used": _SEED_USAGE["total_tokens"],
            },
        )
        logger.record(
            session_id,
            kind="tool_call",
            payload={
                "step": 0,
                "call_id": _SEED_TOOL_CALL_ID,
                "name": _SEED_TOOL_NAME,
                "arguments": _SEED_TOOL_ARGUMENTS,
                "duration_ms": _SEED_DURATION_MS,
            },
        )
        logger.record(
            session_id,
            kind="tool_result",
            payload={
                "step": 0,
                "call_id": _SEED_TOOL_CALL_ID,
                "name": _SEED_TOOL_NAME,
                "duration_ms": _SEED_DURATION_MS,
                "output": _SEED_TOOL_OUTPUT,
                "error": None,
            },
        )
        logger.record(
            session_id,
            kind="outcome",
            payload={"status": "success", "reason": "final_answer", "steps": 1},
        )

    sys.stdout.write(f"seeded session_id={session_id}\n")
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundry-trace",
        description="Inspect and render trace data (ADR-0007).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    render_parser = sub.add_parser(
        "render-failure",
        help="Render a Digester FailureReport as Markdown.",
    )
    render_parser.add_argument("session_id", help="Trace session to digest.")
    render_parser.add_argument(
        "--trace-path",
        default="logs/traces.db",
        help="Path to the trace SQLite database.",
    )
    render_parser.add_argument(
        "--out",
        default=None,
        help="Write Markdown to this path instead of stdout.",
    )
    render_parser.set_defaults(func=_render_failure)

    sessions_parser = sub.add_parser(
        "sessions",
        help="List recorded trace sessions.",
    )
    sessions_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    sessions_parser.set_defaults(func=_sessions)

    show_parser = sub.add_parser(
        "show",
        help="Print the ordered events of one session.",
    )
    show_parser.add_argument("session_id", help="Session to display.")
    show_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    show_parser.set_defaults(func=_show)

    export_parser = sub.add_parser(
        "export",
        help="Export a session as newline-delimited JSON (ADR-0003 JSONL).",
    )
    export_parser.add_argument("session_id", help="Session to export.")
    export_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    export_parser.add_argument(
        "--out",
        default=None,
        help="Write JSONL to this path instead of stdout.",
    )
    export_parser.set_defaults(func=_export)

    # Issue #83 subcommands: session-list / session-show / events-grep.
    session_list_parser = sub.add_parser(
        "session-list",
        help="List trace sessions (session_id, started_at, ended_at, harness_version).",
    )
    session_list_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    session_list_parser.add_argument(
        "--harness-version",
        default=None,
        help="Filter to sessions recorded with this harness version.",
    )
    session_list_parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Print at most N sessions after filtering.",
    )
    session_list_parser.set_defaults(func=_session_list)

    session_show_parser = sub.add_parser(
        "session-show",
        help="Print every event of a session via the timeline renderer.",
    )
    session_show_parser.add_argument("session_id", help="Session to display.")
    session_show_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    session_show_parser.set_defaults(func=_session_show)

    events_grep_parser = sub.add_parser(
        "events-grep",
        help="Print events whose payload JSON matches a regex.",
    )
    events_grep_parser.add_argument("session_id", help="Session to scan.")
    events_grep_parser.add_argument(
        "--pattern",
        required=True,
        help="Python regex applied to each event's serialized payload.",
    )
    events_grep_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    events_grep_parser.set_defaults(func=_events_grep)

    # Issue #192 subcommands: redact-session / redact-key.
    redact_session_parser = sub.add_parser(
        "redact-session",
        help="Delete a session and all its events (SECURITY.md \u00a7Secrets).",
    )
    redact_session_parser.add_argument("session_id", help="Session to delete.")
    redact_session_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    redact_session_parser.add_argument(
        "--out",
        default=None,
        help="Append a JSONL audit-log record to this path.",
    )
    redact_session_parser.set_defaults(func=_redact_session)

    redact_key_parser = sub.add_parser(
        "redact-key",
        help="Overwrite a single payload field with [REDACTED] (SECURITY.md \u00a7Secrets).",
    )
    redact_key_parser.add_argument("session_id", help="Session containing the event.")
    redact_key_parser.add_argument(
        "event_index",
        type=int,
        help="Zero-based index of the event in timestamp order.",
    )
    redact_key_parser.add_argument("key", help="Payload key to overwrite.")
    redact_key_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    redact_key_parser.add_argument(
        "--out",
        default=None,
        help="Append a JSONL audit-log record to this path.",
    )
    redact_key_parser.set_defaults(func=_redact_key)

    # Issue #275: delete-session / prune — retention management.
    delete_session_parser = sub.add_parser(
        "delete-session",
        help="Remove a session and all its events (idempotent).",
    )
    delete_session_parser.add_argument("session_id", help="Session to delete.")
    delete_session_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    delete_session_parser.set_defaults(func=_delete_session)

    # Issue #195: offline smoke subcommand. Plants a deterministic session
    # so the Digester, KPI, and regression-report CLIs have realistic input
    # without standing up llama-server. ``--harness-version`` lets the
    # caller simulate a candidate harness version for the compare-kpis
    # baseline-vs-candidate path (issue #100).
    seed_parser = sub.add_parser(
        "seed-sample-trace",
        help="Plant a deterministic sample session for offline smoke testing.",
    )
    seed_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database (default: logs/traces.db).",
    )
    seed_parser.add_argument(
        "--harness-version",
        default=_SEED_DEFAULT_HARNESS_VERSION,
        help=(
            "Harness version to stamp on the seeded session "
            "(default: %(default)s). Set to a candidate version (e.g. "
            "1.0.0) so compare-kpis can treat the seed as the candidate."
        ),
    )
    seed_parser.set_defaults(func=_seed_sample_trace)

    prune_parser = sub.add_parser(
        "prune",
        help="Remove old sessions per retention policy (--keep-last or --older-than).",
    )
    prune_parser.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    prune_parser.add_argument(
        "--keep-last",
        type=int,
        metavar="N",
        help="Keep only the N most recent sessions (by started_at).",
    )
    prune_parser.add_argument(
        "--older-than",
        type=int,
        metavar="DAYS",
        help="Remove sessions older than DAYS days (by started_at).",
    )
    prune_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List sessions that would be removed without deleting them.",
    )
    prune_parser.set_defaults(func=_prune)

    compact_parser = sub.add_parser(
        "compact",
        help="Remove orphaned session_end markers from a JSONL trace file (issue #632).",
    )
    compact_parser.add_argument(
        "--db",
        default="logs/traces.jsonl",
        help="Path to the JSONL trace file (default: logs/traces.jsonl).",
    )
    compact_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print orphaned markers without modifying the file.",
    )
    compact_parser.set_defaults(func=_compact)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
