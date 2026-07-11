from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
