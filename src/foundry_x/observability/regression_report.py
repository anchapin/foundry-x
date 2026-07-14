from __future__ import annotations

from dataclasses import asdict, dataclass

from pydantic import BaseModel, Field

from foundry_x.evolution.critic import CriticVerdict, QuantizationVerdict
from foundry_x.trace.logger import TraceLogger

VERDICT_KIND = "critic_verdict"
TASK_ABORTED_KIND = "task_aborted"
TOKEN_BUDGET_REASON = "token_budget"


class VerdictRecord(BaseModel):
    """Structured payload persisted for every Critic verdict (ADR-0006 boundary model)."""

    verdict: bool = False
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    notes: str = ""


@dataclass
class _Regression:
    task: str
    was_passing_session: str
    now_failing_session: str
    now_failing_version: str


@dataclass
class _NewPass:
    task: str
    was_failing_session: str
    now_passing_session: str
    now_passing_version: str


class RegressionRow(BaseModel):
    """One regressed task observed by the regression report (ADR-0006 boundary model)."""

    task: str
    was_passing_session: str
    now_failing_session: str
    now_failing_version: str


class NewPassRow(BaseModel):
    """One task that began passing in the latest window (ADR-0006 boundary model)."""

    task: str
    was_failing_session: str
    now_passing_session: str
    now_passing_version: str


class RegressionAnalysis(BaseModel):
    """Full result of a regression analysis pass.

    Carries the rendered Markdown report alongside the structured regressions
    and new passes so callers (e.g. ``fx-trace regression-report
    --fail-on-regression``) can both persist the artifact and gate CI off the
    same observation (issue #99).

    Issue #466 adds ``token_budget_abort_count``: the number of sessions
    that recorded at least one ``task_aborted(reason="token_budget")``
    event during the analysis window. This is a separate signal from
    ``regressions`` and ``new_passes`` because token budget aborts are
    task-shaped failures, not harness regressions.
    """

    report: str
    total: int
    approvals: int
    rejections: int
    regressions: list[RegressionRow] = Field(default_factory=list)
    new_passes: list[NewPassRow] = Field(default_factory=list)
    token_budget_abort_count: int = 0


def record_verdict(logger: TraceLogger, session_id: str, verdict: CriticVerdict) -> None:
    """Persist a CriticVerdict as a ``critic_verdict`` trace event."""
    record = VerdictRecord(
        verdict=verdict.verdict,
        passed_checks=list(verdict.passed_checks),
        failed_checks=list(verdict.failed_checks),
        notes=verdict.notes,
    )
    logger.record(session_id=session_id, kind=VERDICT_KIND, payload=record.model_dump())


def _load_verdict_events(
    logger: TraceLogger,
    since: str | None,
) -> list[tuple[str, str, VerdictRecord]]:
    """Stream every ``critic_verdict`` event through :class:`TraceLogger`.

    Issue #273 — previously walked ``list_sessions()`` and called
    ``iter_events(sid)`` once per session, opening a fresh connection per
    session. :meth:`TraceLogger.query_events` collapses that nested loop
    into a single streaming cursor across all sessions; the ``since``
    filter is still applied after the fetch (the issue's
    ``query_events`` signature deliberately does not include a timestamp
    filter — keeping the surface narrow).
    """
    events: list[tuple[str, str, VerdictRecord]] = []
    for event in logger.query_events(kind=VERDICT_KIND):
        if since is not None and event.timestamp < since:
            continue
        events.append(
            (
                event.session_id,
                event.timestamp,
                VerdictRecord(**event.payload),
            )
        )
    # Preserve the previous ORDER BY timestamp ASC, rowid ASC ordering
    # (issue #82: deterministic ordering keeps the regression-pairing
    # logic stable across runs). ``query_events`` already returns rows in
    # timestamp order; this stable re-sort guarantees identical tie-
    # breaking to the prior per-session nested loop.
    events.sort(key=lambda row: row[1])
    return events


def _compute(
    events: list[tuple[str, str, VerdictRecord]],
    versions: dict[str, str],
) -> tuple[list[_Regression], list[_NewPass]]:
    prior_passed: dict[str, str] = {}
    prior_failed: dict[str, str] = {}
    regressions: list[_Regression] = []
    new_passes: list[_NewPass] = []
    for session_id, _timestamp, verdict in events:
        session_version = versions.get(session_id, "")
        for task in verdict.failed_checks:
            if task in prior_passed:
                regressions.append(
                    _Regression(
                        task=task,
                        was_passing_session=prior_passed[task],
                        now_failing_session=session_id,
                        now_failing_version=session_version,
                    )
                )
        for task in verdict.passed_checks:
            if task in prior_failed:
                new_passes.append(
                    _NewPass(
                        task=task,
                        was_failing_session=prior_failed[task],
                        now_passing_session=session_id,
                        now_passing_version=session_version,
                    )
                )
        for task in verdict.passed_checks:
            prior_passed[task] = session_id
        for task in verdict.failed_checks:
            prior_failed[task] = session_id
    return regressions, new_passes


def generate_regression_report(
    logger: TraceLogger,
    since: str | None = None,
    task: str | None = None,
) -> str:
    """Produce a Markdown regression report over all persisted Critic verdicts.

    When ``task`` is provided, only rows whose ``task`` column equals that
    name are included in the Regressed / New Passes sections (issue #182).
    The Regression Summary counts (total verdicts / approvals / rejections)
    remain the full population so the reviewer keeps context about the
    analysis pass. If the task filter eliminates every row, the rendered
    report collapses to a single ``no rows for task <name>`` line.
    """
    return analyze_regressions(logger, since=since, task=task).report


def analyze_regressions(
    logger: TraceLogger,
    since: str | None = None,
    task: str | None = None,
) -> RegressionAnalysis:
    """Run the regression analysis and return both the Markdown report and the
    structured rows.

    Issue #99: the regression-report CLI needs both the rendered artifact and
    the list of regressed tasks (to gate CI with ``--fail-on-regression``).
    Doing the analysis once here keeps the report and the gate consistent.

    Issue #182: ``task`` narrows the regressions / new passes lists to a
    single task name. The summary counts stay at full population so the
    filtered view does not silently hide regressions in unrelated tasks.

    Issue #466: ``token_budget_abort_count`` counts sessions that recorded
    at least one ``task_aborted(reason="token_budget")`` event. This is
    reported separately from task regressions because it is a task-sizing
    problem, not a harness defect.
    """
    events = _load_verdict_events(logger, since)
    total = len(events)
    approvals = sum(1 for _sid, _ts, v in events if v.verdict)
    rejections = total - approvals
    versions = _session_versions(logger)
    regressions, new_passes = _compute(events, versions)
    if task is not None:
        regressions = [r for r in regressions if r.task == task]
        new_passes = [p for p in new_passes if p.task == task]
    token_budget_abort_count = _count_token_budget_aborts(logger, since=since)
    report = _render(
        total,
        approvals,
        rejections,
        regressions,
        new_passes,
        token_budget_abort_count=token_budget_abort_count,
        task=task,
    )
    return RegressionAnalysis(
        report=report,
        total=total,
        approvals=approvals,
        rejections=rejections,
        regressions=[RegressionRow(**asdict(r)) for r in regressions],
        new_passes=[NewPassRow(**asdict(p)) for p in new_passes],
        token_budget_abort_count=token_budget_abort_count,
    )


def _session_versions(logger: TraceLogger) -> dict[str, str]:
    """Build a ``session_id -> harness_version`` map for every known session.

    The map is consumed by :func:`_compute` so each regression / new-pass row
    can surface the manifest version of its *current-state* session (issue
    #103: regression_report gains a column showing the manifest version of
    each verdict's source session). Sessions whose row is missing are
    rendered as an empty string rather than ``None`` so the Markdown table
    stays a 4-column shape.
    """
    return {s.session_id: s.harness_version for s in logger.list_sessions()}


def _render(
    total: int,
    approvals: int,
    rejections: int,
    regressions: list[_Regression],
    new_passes: list[_NewPass],
    token_budget_abort_count: int = 0,
    task: str | None = None,
) -> str:
    # Issue #182: when the task filter narrows both sections to zero rows,
    # collapse the report to a single-line message so the CLI's stdout is
    # grep-friendly without a dangling "_None._" table.
    if task is not None and not regressions and not new_passes:
        return f"no rows for task {task}\n"
    lines: list[str] = [
        "# Critic Regression Report",
        "",
        "## Regression Summary",
        "",
        f"- Total verdicts: {total}",
        f"- Approvals: {approvals}",
        f"- Rejections: {rejections}",
        "",
        "## Regressed Tasks",
        "",
    ]
    if regressions:
        lines.append("| Task | Was passing (session) | Now failing (session) | Manifest version |")
        lines.append("| --- | --- | --- | --- |")
        for reg in regressions:
            lines.append(
                f"| {reg.task} | {reg.was_passing_session} | "
                f"{reg.now_failing_session} | {reg.now_failing_version} |"
            )
    else:
        lines.append("_None._")
    lines += ["", "## New Passes", ""]
    if new_passes:
        lines.append("| Task | Was failing (session) | Now passing (session) | Manifest version |")
        lines.append("| --- | --- | --- | --- |")
        for pas in new_passes:
            lines.append(
                f"| {pas.task} | {pas.was_failing_session} | "
                f"{pas.now_passing_session} | {pas.now_passing_version} |"
            )
    else:
        lines.append("_None._")
    # Issue #466: token budget aborts are a distinct failure category,
    # reported separately from task regressions because they indicate a
    # task-sizing problem, not a harness defect.
    lines += ["", "## Token Budget Aborts", ""]
    lines.append(f"_Token budget aborts: {token_budget_abort_count} session(s)_")
    lines.append("")
    return "\n".join(lines)


 def _count_token_budget_aborts(
     logger: TraceLogger,
     since: str | None = None,
 ) -> int:
     """Count sessions with at least one ``task_aborted(reason="token_budget")`` event.

     Issue #466: token budget aborts are task-shaped failures (a task exceeded
     the model's context budget), not harness regressions. They are reported
     as a separate signal in the regression report.

     Sessions are counted once regardless of how many times the abort fires
     within them. Uses one :meth:`TraceLogger.query_events` cursor.
     """
     sessions_with_abort: set[str] = set()
     for event in logger.query_events(kind=TASK_ABORTED_KIND):
         if since is not None and event.timestamp < since:
             continue
         if event.payload.get("reason") == TOKEN_BUDGET_REASON:
             sessions_with_abort.add(event.session_id)
     return len(sessions_with_abort)


 class QuantizationComparisonReport(BaseModel):
     """Structured result of a quantization comparison sweep (ADR-0016).

     Carries the rendered Markdown report alongside the structured
     ``QuantizationVerdict`` so callers (e.g. CI gate) can both persist
     the artifact and gate off the same observation.
     """

     report: str
     verdict: QuantizationVerdict
     best_token_efficiency: float | None = Field(
         default=None,
         description=(
             "Highest token efficiency (tokens/sec) across all quantizations. "
             "None when no quantization provides token_efficiency data."
         ),
     )
     best_cost_per_task: float | None = Field(
         default=None,
         description=(
             "Lowest cost per task (USD) across all quantizations. "
             "None when no quantization provides cost_per_task data."
         ),
     )


 def generate_quantization_comparison_report(
     verdict: QuantizationVerdict,
 ) -> QuantizationComparisonReport:
     """Produce a Markdown comparison report from a ``QuantizationVerdict``.

     Computes ``best_token_efficiency`` and ``best_cost_per_task`` from the
     per-quantization results and renders a table comparing all quantizations.

     Args:
         verdict: The ``QuantizationVerdict`` produced by
             ``Critic.quantization_sweep``.

     Returns:
         A ``QuantizationComparisonReport`` with the rendered Markdown report
         and the structured verdict including best-cost and best-efficiency
         annotations.
     """
     best_eff = None
     best_cost = None

     for result in verdict.quantizations:
         if result.token_efficiency is not None:
             if best_eff is None or result.token_efficiency > best_eff:
                 best_eff = result.token_efficiency
         if result.cost_per_task is not None:
             if best_cost is None or result.cost_per_task < best_cost:
                 best_cost = result.cost_per_task

     report_lines: list[str] = [
         "# Quantization Comparison Report",
         "",
         "## Summary",
         "",
         f"- Quantizations compared: {len(verdict.quantizations)}",
         f"- Recommended: **{verdict.recommended}**",
         f"- Regression detected: {verdict.regression}",
     ]
     if best_eff is not None:
         report_lines.append(f"- Best token efficiency: {best_eff:.1f} tokens/s")
     if best_cost is not None:
         report_lines.append(f"- Best cost per task: ${best_cost:.6f}")
     report_lines.append("")
     report_lines.append("## Per-Quantization Results")
     report_lines.append("")
     report_lines.append("| Quantization | Pass Rate | Avg Cycle | Tokens | Tok/s | Cost/Task |")
     report_lines.append("|---|---|---|---|---|---|")
     for result in verdict.quantizations:
         pr = f"{result.pass_rate * 100:.1f}%"
         avg_t = f"{result.avg_cycle_time_s:.1f}s" if result.avg_cycle_time_s else "N/A"
         toks = f"{result.total_tokens:,}" if result.total_tokens else "N/A"
         eff = f"{result.token_efficiency:.1f}" if result.token_efficiency else "N/A"
         cost = f"${result.cost_per_task:.6f}" if result.cost_per_task else "N/A"
         rec_marker = " **(rec)**" if result.quantization == verdict.recommended else ""
         report_lines.append(
             f"| {result.quantization}{rec_marker} | {pr} | {avg_t} | {toks} | {eff} | {cost} |"
         )
     report_lines.append("")
     return QuantizationComparisonReport(
         report="\n".join(report_lines),
         verdict=verdict,
         best_token_efficiency=best_eff,
         best_cost_per_task=best_cost,
     )
