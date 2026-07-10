from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foundry_x.evolution.digester import Digester
from foundry_x.observability.render import render_failure_report
from foundry_x.trace.logger import TraceLogger


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foundry-x-trace",
        description="Inspect and render trace data.",
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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
