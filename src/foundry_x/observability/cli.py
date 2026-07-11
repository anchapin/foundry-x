from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foundry_x.observability.regression_report import analyze_regressions
from foundry_x.observability.session_card import format_session_card
from foundry_x.observability.session_summary import (
    build_session_summary,
    render_session_summary,
)
from foundry_x.observability.timeline import format_timeline
from foundry_x.trace.logger import TraceLogger


def _infer_backend(path: str | Path) -> str:
    """Return ``"jsonl"`` for ``.jsonl`` paths, ``"sqlite"`` otherwise."""
    suffix = Path(path).suffix.lower()
    return "jsonl" if suffix == ".jsonl" else "sqlite"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fx-trace",
        description="FoundryX trace inspection tooling.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    regression = sub.add_parser(
        "regression-report",
        help="Aggregate Critic verdicts into a regression report.",
    )
    regression.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace sqlite database (default: logs/traces.db).",
    )
    regression.add_argument(
        "--since",
        default=None,
        help="ISO-8601 timestamp; only consider verdicts at or after this time.",
    )
    regression.add_argument(
        "--out",
        default=None,
        help="Write the report to this path instead of stdout.",
    )
    regression.add_argument(
        "--fail-on-regression",
        action="store_true",
        default=False,
        help=(
            "Exit non-zero if any regressed task is detected (CI gate). "
            "The Markdown artifact is still written to --out before the exit code."
        ),
    )
    regression.add_argument(
        "--task",
        default=None,
        help=(
            "Only show regressions / new passes for this task name "
            "(issue #182). Other rows are excluded from the table; "
            "summary counts remain the full population."
        ),
    )
    regression.add_argument(
        "--format",
        default="markdown",
        choices=("markdown", "json"),
        help=(
            "Output format. ``json`` emits the structured RegressionAnalysis "
            "(grep-friendly, includes the filtered rows)."
        ),
    )

    timeline = sub.add_parser(
        "timeline",
        help="Print the formatted timeline of a session from the trace store.",
    )
    timeline.add_argument(
        "--db",
        required=True,
        help="Path to the trace store (sqlite .db or jsonl).",
    )
    timeline.add_argument(
        "--session-id",
        required=True,
        help="Session UUID whose events should be rendered.",
    )

    # Issue #184: cross-session outcome roll-up. Lets an Operator read
    # a single table over every (or filtered) session before opening
    # any one of them in detail. Does not call the Digester or Critic.
    session_summary = sub.add_parser(
        "session-summary",
        help="Render a one-row-per-session roll-up of recorded outcomes.",
    )
    session_summary.add_argument(
        "--db",
        default="logs/traces.db",
        help="Path to the trace store (sqlite .db or jsonl).",
    )
    session_summary.add_argument(
        "--harness-version",
        default=None,
        help="Only include sessions recorded with this harness version.",
    )
    session_summary.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Show at most N sessions after newest-first ordering.",
    )

    # Issue #180: ``fx-trace session-card <sid>`` prints a one-screen
    # triage summary for a session — the human counterpart to the verbose
    # ``timeline`` view. Exits non-zero when the session id is unknown so
    # CI gates can depend on it.
    card = sub.add_parser(
        "session-card",
        help="Print a one-screen triage card for a session from the trace store.",
    )
    card.add_argument(
        "--db",
        required=True,
        help="Path to the trace store (sqlite .db or jsonl).",
    )
    card.add_argument(
        "--session-id",
        required=True,
        help="Session UUID whose triage card should be rendered.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "regression-report":
        logger = TraceLogger(args.db)
        analysis = analyze_regressions(logger, since=args.since, task=args.task)
        if args.format == "json":
            rendered = analysis.model_dump_json(indent=2) + "\n"
        else:
            rendered = analysis.report
        if args.out:
            Path(args.out).write_text(rendered, encoding="utf-8")
        else:
            sys.stdout.write(rendered)
        if args.fail_on_regression and analysis.regressions:
            return 1
        return 0

    if args.command == "timeline":
        backend = _infer_backend(args.db)
        logger = TraceLogger(args.db, backend=backend)
        events = logger.load_session(args.session_id)
        if not events:
            sys.stderr.write(f"session {args.session_id} not found or empty\n")
            return 2
        sys.stdout.write(format_timeline(events))
        sys.stdout.write("\n")
        return 0

    if args.command == "session-summary":
        backend = _infer_backend(args.db)
        logger = TraceLogger(args.db, backend=backend)
        rows = build_session_summary(logger, harness_version=args.harness_version)
        sys.stdout.write(render_session_summary(rows, limit=args.limit))
        sys.stdout.write("\n")
        return 0

    if args.command == "session-card":
        backend = _infer_backend(args.db)
        logger = TraceLogger(args.db, backend=backend)
        # Match by session_id rather than by events alone so an empty
        # in-flight session still renders its roll-up (graceful degrade,
        # issue #180) instead of being reported as missing.
        match = next(
            (s for s in logger.list_sessions() if s.session_id == args.session_id),
            None,
        )
        if match is None:
            sys.stderr.write(f"session {args.session_id} not found\n")
            return 2
        events = logger.load_session(args.session_id)
        sys.stdout.write(format_session_card(match, events) + "\n")
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
