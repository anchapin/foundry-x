from __future__ import annotations

import argparse
import json
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
