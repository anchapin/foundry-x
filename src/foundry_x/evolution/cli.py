"""foundry-evolve CLI — one-shot evolution loop execution (issue #256).

Orchestrates the evolution loop as a standalone command so an operator can
run one evolution step without writing Python code::

    foundry-evolve --session-id <id> --trace-db <path> --harness-dir <dir>

The loop is: TraceLogger -> Digester -> Evolver -> Critic.

Exit codes:
    0  Critic approved the edit (or no failure was detected)
    1  Critic rejected the edit
    2  Digester produced no session events, or other usage error
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from foundry_x.evolution.critic import Critic, CriticVerdict
from foundry_x.evolution.digester import Digester, FailureReport
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.execution.runner import resolve_harness_version
from foundry_x.trace.logger import TraceLogger


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with offset suffix."""
    return datetime.now(timezone.utc).isoformat()


def _infer_backend(trace_db: str) -> str:
    """Return ``"jsonl"`` for ``.jsonl`` paths, ``"sqlite"`` otherwise."""
    return "jsonl" if trace_db.endswith(".jsonl") else "sqlite"


def _render_failure_report(report: FailureReport) -> str:
    """Render a FailureReport as a compact plain-text summary."""
    lines = [
        f"Failure Report — session `{report.session_id}`",
        f"  Summary: {report.summary}",
        f"  Proposed class: {report.proposed_class}",
    ]
    if report.suspected_causes:
        lines.append("  Suspected causes:")
        for cause in report.suspected_causes:
            lines.append(f"    - {cause}")
    else:
        lines.append("  Suspected causes: none")
    if report.failed_steps:
        lines.append(f"  Failed steps: {len(report.failed_steps)}")
    else:
        lines.append("  Failed steps: none")
    return "\n".join(lines)


def _render_proposed_edit(edit: ProposedEdit, verbose: bool = False) -> str:
    """Render a ProposedEdit as a compact summary."""
    lines = [
        "Proposed Edit:",
        f"  Target: {edit.target_file}",
        f"  Rationale: {edit.rationale}",
    ]
    if verbose:
        lines.append(f"  Unified diff:\n{edit.unified_diff}")
    else:
        diff_lines = edit.unified_diff.splitlines()
        lines.append(f"  Unified diff: {len(diff_lines)} line(s) (use --verbose to print)")
    return "\n".join(lines)


def _render_critic_verdict(verdict: CriticVerdict) -> str:
    """Render a CriticVerdict as a compact plain-text summary."""
    status = "APPROVED" if verdict.verdict else "REJECTED"
    lines = [f"Critic Verdict: {status}"]
    if verdict.passed_checks:
        lines.append("  Passed checks:")
        for check in verdict.passed_checks:
            lines.append(f"    + {check}")
    if verdict.failed_checks:
        lines.append("  Failed checks:")
        for check in verdict.failed_checks:
            lines.append(f"    - {check}")
    if verdict.notes:
        notes_preview = verdict.notes[:500]
        if len(verdict.notes) > 500:
            notes_preview += " [...truncated]"
        lines.append(f"  Notes: {notes_preview}")
    return "\n".join(lines)


def _run_loop(
    session_id: str,
    trace_db: str,
    harness_dir: Path,
    verbose: bool = False,
) -> tuple[FailureReport, ProposedEdit | None, CriticVerdict | None, int, str]:
    """Execute the evolution loop: Digester -> Evolver -> Critic.

    Returns (failure_report, proposed_edit, verdict, exit_code, harness_version).
    proposed_edit may be None if no failure was detected or Evolver is not yet
    implemented. verdict is None if no proposed_edit was produced.
    Exit code 0 = approved / no failure, 1 = rejected, 2 = error.
    """
    harness_version = resolve_harness_version(harness_dir)
    started_at = _now_iso()
    backend = _infer_backend(trace_db)
    logger = TraceLogger(trace_db, backend=backend)
    events = logger.load_session(session_id)
    if not events:
        sys.stderr.write(f"No events found for session {session_id}.\n")
        return None, None, None, 2, harness_version

    report = Digester().digest(session_id, events)
    print(_render_failure_report(report))
    print()

    if report.proposed_class == "clean":
        completed_at = _now_iso()
        print(f"Started: {started_at} | Completed: {completed_at}")
        print()
        print("No failure detected — evolution loop complete.")
        return report, None, None, 0, harness_version

    evolver = Evolver()
    try:
        edits = evolver.propose(harness_dir=harness_dir, failure=report)
    except NotImplementedError:
        completed_at = _now_iso()
        print(f"Started: {started_at} | Completed: {completed_at}")
        print()
        sys.stderr.write(
            "Evolver.propose() is not yet implemented (Phase 2). "
            "The evolution loop cannot produce a ProposedEdit yet.\n"
        )
        return report, None, None, 2, harness_version

    if not edits:
        completed_at = _now_iso()
        print(f"Started: {started_at} | Completed: {completed_at}")
        print()
        print("Evolver returned no ProposedEdit objects.")
        return report, None, None, 0, harness_version

    edit = edits[0]
    print(_render_proposed_edit(edit, verbose=verbose))
    print()

    critic = Critic(harness_dir=harness_dir)
    verdict = critic.evaluate(edit.unified_diff)
    print(_render_critic_verdict(verdict))
    print()

    exit_code = 0 if verdict.verdict else 1
    return report, edit, verdict, exit_code, harness_version


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundry-evolve",
        description=(
            "Run one evolution loop step: TraceLogger -> Digester -> "
            "Evolver -> Critic.  Prints FailureReport, ProposedEdit "
            "details, and CriticVerdict.  Exit 0 when Critic approves "
            "(or no failure found), 1 when rejected."
        ),
    )
    parser.add_argument(
        "--session-id",
        required=True,
        help="Trace session UUID to analyse.",
    )
    parser.add_argument(
        "--trace-db",
        default="logs/traces.db",
        help="Path to the trace SQLite database or JSONL file (default: logs/traces.db).",
    )
    parser.add_argument(
        "--harness-dir",
        required=True,
        type=Path,
        help="Path to the harness directory to be evolved and evaluated.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full unified_diff of each ProposedEdit.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
        session_id=args.session_id,
        trace_db=args.trace_db,
        harness_dir=args.harness_dir,
        verbose=args.verbose,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
