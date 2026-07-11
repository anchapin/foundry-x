from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foundry_x.observability.regression_report import analyze_regressions
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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "regression-report":
        logger = TraceLogger(args.db)
        analysis = analyze_regressions(logger, since=args.since)
        if args.out:
            Path(args.out).write_text(analysis.report, encoding="utf-8")
        else:
            sys.stdout.write(analysis.report)
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

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
