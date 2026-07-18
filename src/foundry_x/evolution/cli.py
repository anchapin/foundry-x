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
import asyncio
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from foundry_x.evolution.critic import (
    Critic,
    CriticVerdict,
    QuantizationResult,
    QuantizationVerdict,
)
from foundry_x.evolution.digester import Digester, FailureReport
from foundry_x.evolution.evolver import Evolver, ProposedEdit
from foundry_x.evolution.loop import run_evolution_step_async
from foundry_x.execution.runner import resolve_harness_version
from foundry_x.evolution.store import ProposedEditStore, TrackedProposedEdit, ProposedEditStatus
from foundry_x.observability.regression_report import record_verdict
from foundry_x.trace.logger import TraceLogger


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with offset suffix."""
    return datetime.now(timezone.utc).isoformat()


def _infer_backend(trace_db: str) -> str:
    """Return ``"jsonl"`` for ``.jsonl`` paths, ``"sqlite"`` otherwise."""
    return "jsonl" if trace_db.endswith(".jsonl") else "sqlite"


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with offset suffix."""
    return datetime.now(timezone.utc).isoformat()


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


def _render_tracked_edit(edit: TrackedProposedEdit, verbose: bool = False) -> str:
    """Render a TrackedProposedEdit as a compact summary."""
    lines = [
        f"ID: {edit.id}",
        f"  Status: {edit.status.value.upper()}",
        f"  Target: {edit.target_file}",
        f"  Rationale: {edit.rationale}",
    ]
    if edit.review_reason:
        lines.append(f"  Review reason: {edit.review_reason}")
    if edit.reviewed_at:
        lines.append(f"  Reviewed at: {edit.reviewed_at}")
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


def _render_quantization_result(result: QuantizationResult) -> str:
    """Render a single QuantizationResult as a table row."""
    pass_rate_pct = result.pass_rate * 100
    avg_time = f"{result.avg_cycle_time_s:.1f}s" if result.avg_cycle_time_s else "N/A"
    token_eff = f"{result.token_efficiency:.1f}" if result.token_efficiency else "N/A"
    cost = f"${result.cost_per_task:.4f}" if result.cost_per_task else "N/A"
    return (
        f"  {result.quantization:<15} | {pass_rate_pct:>6.1f}% | "
        f"{avg_time:>8} | {result.total_tokens:>10} | "
        f"{token_eff:>8} | {cost:>10} | {result.model_id}"
    )


def _render_quantization_verdict(verdict: QuantizationVerdict) -> str:
    """Render a QuantizationVerdict as a comparison table."""
    lines = ["Quantization Sweep Results", "=" * 90]
    header = (
        f"  {'Quantization':<15} | {'Pass Rate':>9} | {'Avg Cycle':>10} | "
        f"{'Tokens':>10} | {'Tok/s':>8} | {'Cost/Task':>10} | Model ID"
    )
    lines.append(header)
    lines.append("-" * 90)
    for result in verdict.quantizations:
        lines.append(_render_quantization_result(result))
    lines.append("=" * 90)
    reg_status = "REGRESSION DETECTED" if verdict.regression else "No regression"
    lines.append(f"Recommended: {verdict.recommended}  [{reg_status}]")
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
    verdict_with_class = CriticVerdict(
        verdict=verdict.verdict,
        passed_checks=list(verdict.passed_checks),
        failed_checks=list(verdict.failed_checks),
        notes=verdict.notes,
        failure_class=report.proposed_class,
    )
    record_verdict(logger, session_id, verdict_with_class)
    print(_render_critic_verdict(verdict))
    print()

    completed_at = _now_iso()
    print(f"Started: {started_at} | Completed: {completed_at}")
    print()

    exit_code = 0 if verdict.verdict else 1
    return report, edit, verdict, exit_code, harness_version


async def _run_loop_async(
    session_id: str,
    trace_db: str,
    harness_dir: Path,
    verbose: bool = False,
) -> tuple[FailureReport, ProposedEdit | None, CriticVerdict | None, int]:
    """Async variant of _run_loop.

    Phase 1: awaits run_evolution_step_async (no-op, Critic is still sync).
    """
    harness_version = resolve_harness_version(harness_dir)
    started_at = _now_iso()
    backend = _infer_backend(trace_db)
    logger = TraceLogger(trace_db, backend=backend)
    events = logger.load_session(session_id)
    if not events:
        sys.stderr.write(f"No events found for session {session_id}.\n")
        return None, None, None, 2, harness_version

    result = await run_evolution_step_async(
        session_id=session_id,
        events=events,
        harness_dir=harness_dir,
    )

    report = result.failure_report
    print(_render_failure_report(report))
    print()

    if report.proposed_class == "clean":
        completed_at = _now_iso()
        print(f"Started: {started_at} | Completed: {completed_at}")
        print()
        print("No failure detected — evolution loop complete.")
        return report, None, None, 0, harness_version

    if not result.proposed_edits:
        print("Evolver returned no ProposedEdit objects.")
        return report, None, None, 0, harness_version

    edit = result.proposed_edits[0]
    print(_render_proposed_edit(edit, verbose=verbose))
    print()

    if result.verdict is not None:
        print(_render_critic_verdict(result.verdict))
        print()

    exit_code = 0 if (result.verdict and result.verdict.verdict) else 1
    return report, edit, result.verdict, exit_code

def _list_pending(args: argparse.Namespace) -> int:
    """Implement ``foundry-evolve list-pending`` (issue #498)."""
    store = ProposedEditStore(args.store)
    pending = store.list_pending()
    if not pending:
        sys.stdout.write("No pending ProposedEdits.\n")
        store.close()
        return 0
    sys.stdout.write(f"{len(pending)} pending ProposedEdit(s):\n\n")
    for edit in pending:
        sys.stdout.write(_render_tracked_edit(edit, verbose=args.verbose) + "\n\n")
    store.close()
    return 0


def _approve(args: argparse.Namespace) -> int:
    """Implement ``foundry-evolve approve <edit_id>`` (issue #498)."""
    store = ProposedEditStore(args.store)
    edit = store.approve(args.edit_id, reason=args.reason or "")
    store.close()
    if edit is None:
        return 1
    sys.stdout.write(f"Approved edit {args.edit_id}.\n")
    sys.stdout.write(_render_tracked_edit(edit, verbose=True) + "\n")
    return 0


def _reject(args: argparse.Namespace) -> int:
    """Implement ``foundry-evolve reject <edit_id>`` (issue #498)."""
    store = ProposedEditStore(args.store)
    edit = store.reject(args.edit_id, reason=args.reason or "")
    store.close()
    if edit is None:
        return 1
    sys.stdout.write(f"Rejected edit {args.edit_id}.\n")
    sys.stdout.write(_render_tracked_edit(edit, verbose=True) + "\n")
    return 0


def _apply(args: argparse.Namespace) -> int:
    """Implement ``foundry-evolve apply <edit_id>`` (issue #498).

    Transitions the edit to APPLIED and applies the unified_diff to the harness
    directory via git apply. Runs git apply from the parent of harness_dir so
    that diff paths (e.g. ``harness/system_prompt.txt``) resolve correctly.
    """
    store = ProposedEditStore(args.store)
    edit = store.get(args.edit_id)
    if edit is None:
        sys.stderr.write(f"apply: edit {args.edit_id} not found.\n")
        store.close()
        return 2
    if edit.status != ProposedEditStatus.APPROVED:
        sys.stderr.write(
            f"apply: edit {args.edit_id} is {edit.status.value}; must be approved first.\n"
        )
        store.close()
        return 1
    harness_dir = Path(args.harness_dir)
    if not harness_dir.exists():
        sys.stderr.write(f"apply: harness directory {args.harness_dir} does not exist.\n")
        store.close()
        return 2
    git_apply_dir = harness_dir.parent
    result = subprocess.run(
        ["git", "apply", "--whitespace=nowarn"],
        input=edit.unified_diff,
        cwd=git_apply_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        sys.stderr.write(f"apply: git apply failed:\n{result.stderr}\n")
        store.close()
        return 1
    applied = store.mark_applied(args.edit_id)
    store.close()
    sys.stdout.write(f"Applied edit {args.edit_id} to harness.\n")
    if applied:
        sys.stdout.write(_render_tracked_edit(applied, verbose=False) + "\n")
    return 0


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
    parser.add_argument(
        "--async",
        dest="use_async",
        action="store_true",
        help="Run the evolution loop asynchronously (Phase 1: no-op, awaits the call).",
    )
    return parser


def _build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="foundry-sweep",
        description=(
            "Run a quantization sweep: execute the benchmark suite against "
            "each listed quantization and produce a comparison table. "
            "FOUNDRY_MODEL_PATH must point to a directory containing model files. "
            "Each model file is matched via a glob pattern (default: *.<quant>.gguf). "
            "Exit 0 on success, non-zero if any quantization fails all benchmarks."
        ),
    )
    parser.add_argument(
        "--quantizations",
        required=True,
        help=(
            "Comma-separated list of quantization labels to sweep (e.g. Q4_K_S,Q5_K_M,Q6_K,Q8_0)."
        ),
    )
    parser.add_argument(
        "--harness-dir",
        required=True,
        type=Path,
        help="Path to the harness directory to be evaluated.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Baseline quantization to compare against. "
            "Defaults to the first quantization in --quantizations."
        ),
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=2.0,
        help=(
            "Regression threshold in percentage points. A candidate's pass "
            "rate must be within this many pp of the baseline to be "
            "considered non-regressing (default: 2.0)."
        ),
    )
    parser.add_argument(
        "--cost-per-token",
        type=float,
        default=None,
        help=(
            "Cost per token in USD for cost-per-task computation. "
            "Can also be set via FOUNDRY_COST_PER_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        dest="output",
        help=(
            "Path to write the quantization verdict as JSON. "
            "If not specified, results are not persisted to disk."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    if argv and argv[0] in ("evolve", "sweep", "list-pending", "approve", "reject", "apply"):
        parser = argparse.ArgumentParser(
            prog="foundry-evolve",
            description="foundry-evolve and foundry-sweep commands.",
        )
        sub = parser.add_subparsers(dest="command", required=True)

        evolve_parser = sub.add_parser("evolve", help="Run one evolution loop step.")
        _build_evolve_subparser(evolve_parser)

        sweep_parser = sub.add_parser("sweep", help="Run a quantization sweep.")
        _build_sweep_subparser(sweep_parser)

        list_parser = sub.add_parser(
            "list-pending",
            help="List all pending ProposedEdits awaiting review (issue #498).",
        )
        list_parser.add_argument(
            "--store",
            default="logs/proposed_edits.db",
            help="Path to the ProposedEdit SQLite store (default: logs/proposed_edits.db).",
        )
        list_parser.add_argument(
            "--verbose",
            action="store_true",
            help="Print the full unified_diff for each edit.",
        )
        list_parser.set_defaults(func=_list_pending)

        approve_parser = sub.add_parser(
            "approve",
            help="Approve a pending ProposedEdit (issue #498).",
        )
        approve_parser.add_argument(
            "edit_id",
            help="UUID of the ProposedEdit to approve.",
        )
        approve_parser.add_argument(
            "--store",
            default="logs/proposed_edits.db",
            help="Path to the ProposedEdit SQLite store (default: logs/proposed_edits.db).",
        )
        approve_parser.add_argument(
            "--reason",
            default="",
            help="Optional reason for the approval.",
        )
        approve_parser.set_defaults(func=_approve)

        reject_parser = sub.add_parser(
            "reject",
            help="Reject a pending ProposedEdit (issue #498).",
        )
        reject_parser.add_argument(
            "edit_id",
            help="UUID of the ProposedEdit to reject.",
        )
        reject_parser.add_argument(
            "--store",
            default="logs/proposed_edits.db",
            help="Path to the ProposedEdit SQLite store (default: logs/proposed_edits.db).",
        )
        reject_parser.add_argument(
            "--reason",
            default="",
            help="Reason for the rejection.",
        )
        reject_parser.set_defaults(func=_reject)

        apply_parser = sub.add_parser(
            "apply",
            help="Apply an approved ProposedEdit to the harness (issue #498).",
        )
        apply_parser.add_argument(
            "edit_id",
            help="UUID of the ProposedEdit to apply.",
        )
        apply_parser.add_argument(
            "--store",
            default="logs/proposed_edits.db",
            help="Path to the ProposedEdit SQLite store (default: logs/proposed_edits.db).",
        )
        apply_parser.add_argument(
            "--harness-dir",
            default="harness",
            help="Path to the harness directory (default: harness).",
        )
        apply_parser.set_defaults(func=_apply)

        args = parser.parse_args(argv)

        if args.command == "evolve":
            return _main_evolve(args)
        elif args.command == "sweep":
            return _main_sweep(args)
        elif args.command in ("list-pending", "approve", "reject", "apply"):
            return args.func(args)
        else:
            return 2
    else:
        return _main_evolve_legacy(argv)


def _build_evolve_subparser(parser: argparse.ArgumentParser) -> None:
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


def _build_sweep_subparser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--quantizations",
        required=True,
        help=(
            "Comma-separated list of quantization labels to sweep (e.g. Q4_K_S,Q5_K_M,Q6_K,Q8_0)."
        ),
    )
    parser.add_argument(
        "--harness-dir",
        required=True,
        type=Path,
        help="Path to the harness directory to be evaluated.",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help=(
            "Baseline quantization to compare against. "
            "Defaults to the first quantization in --quantizations."
        ),
    )
    parser.add_argument(
        "--regression-threshold",
        type=float,
        default=2.0,
        help=("Regression threshold in percentage points (default: 2.0)."),
    )
    parser.add_argument(
        "--cost-per-token",
        type=float,
        default=None,
        help=(
            "Cost per token in USD for cost-per-task computation. "
            "Can also be set via FOUNDRY_COST_PER_TOKEN environment variable."
        ),
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        dest="output",
        help=(
            "Path to write the quantization verdict as JSON. "
            "If not specified, results are not persisted to disk."
        ),
    )


def _main_evolve(args: argparse.Namespace) -> int:
    _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
        session_id=args.session_id,
        trace_db=args.trace_db,
        harness_dir=args.harness_dir,
        verbose=args.verbose,
    )
    return exit_code


def _main_evolve_legacy(argv: list[str] | None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.use_async:
        _report, _edit, _verdict, exit_code = asyncio.run(
            _run_loop_async(
                session_id=args.session_id,
                trace_db=args.trace_db,
                harness_dir=args.harness_dir,
                verbose=args.verbose,
            )
        )
    else:
        _report, _edit, _verdict, exit_code, _harness_version = _run_loop(
            session_id=args.session_id,
            trace_db=args.trace_db,
            harness_dir=args.harness_dir,
            verbose=args.verbose,
        )
    return exit_code


def _main_sweep(args: argparse.Namespace) -> int:
    quantizations = [q.strip() for q in args.quantizations.split(",") if q.strip()]
    if not quantizations:
        sys.stderr.write("--quantizations must specify at least one quantization label.\n")
        return 2

    critic = Critic(harness_dir=args.harness_dir)
    try:
        verdict = critic.quantization_sweep(
            quantizations=quantizations,
            baseline_quantization=args.baseline,
            regression_threshold_pp=args.regression_threshold,
            cost_per_token=args.cost_per_token,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(verdict.model_dump(mode="json"), indent=2))
        print(f"Results written to {output_path}", file=sys.stderr)

    print(_render_quantization_verdict(verdict))
    return 0 if not verdict.regression else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))


def sweep_main(argv: list[str] | None = None) -> int:
    """Entry point for the ``foundry-sweep`` CLI (issue #464).

    Standalone sweep invocation that does not go through the evolution loop.
    Runs the benchmark suite against each listed quantization and prints a
    comparison table.

    Exit codes:
        0  Sweep completed with no regression detected
        1  Sweep completed but regression detected
        2  Usage error or model path not found
    """
    parser = _build_sweep_parser()
    args = parser.parse_args(argv)

    quantizations = [q.strip() for q in args.quantizations.split(",") if q.strip()]
    if not quantizations:
        sys.stderr.write("--quantizations must specify at least one quantization label.\n")
        return 2

    critic = Critic(harness_dir=args.harness_dir)
    try:
        verdict = critic.quantization_sweep(
            quantizations=quantizations,
            baseline_quantization=args.baseline,
            regression_threshold_pp=args.regression_threshold,
            cost_per_token=args.cost_per_token,
        )
    except (ValueError, FileNotFoundError) as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(verdict.model_dump(mode="json"), indent=2))
        print(f"Results written to {output_path}", file=sys.stderr)

    print(_render_quantization_verdict(verdict))
    return 0 if not verdict.regression else 1
