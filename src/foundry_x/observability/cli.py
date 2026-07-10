from __future__ import annotations

import argparse
import sys
from pathlib import Path

from foundry_x.observability.regression_report import generate_regression_report
from foundry_x.trace.logger import TraceLogger


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
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if args.command == "regression-report":
        logger = TraceLogger(args.db)
        report = generate_regression_report(logger, since=args.since)
        if args.out:
            Path(args.out).write_text(report, encoding="utf-8")
        else:
            sys.stdout.write(report)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
