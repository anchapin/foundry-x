from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, Field

from foundry_x.evolution.critic import CriticVerdict, QuantizationVerdict
from foundry_x.trace.logger import TraceLogger

VERDICT_KIND = "critic_verdict"
TASK_ABORTED_KIND = "task_aborted"
TOKEN_BUDGET_REASON = "token_budget"
TOKEN_BUDGET_ABORTED_KIND = "token_budget_aborted"

GroupByDim = Literal["skill", "task_family", "difficulty_tier"]


class TaskKpiMetadata(BaseModel):
    """Grouping metadata for one benchmark task.

    Mirrors the boundary model from :mod:`foundry_x.observability.kpis`
    so the regression report can accept task-level groupings without
    importing the KPI module.
    """

    name: str
    skills: list[str] = Field(default_factory=list)
    task_families: list[str] = Field(default_factory=list)
    difficulty_tier: str | None = None


class SliceRegressions(BaseModel):
    """Per-dimension regression breakdown (issue #1114).

    Attributes
    ----------
    skill:
        Regression counts keyed by skill name.
    task_family:
        Regression counts keyed by task-family (tag) name.
    difficulty_tier:
        Regression counts keyed by difficulty tier name.
    """

    skill: dict[str, int] = Field(default_factory=dict)
    task_family: dict[str, int] = Field(default_factory=dict)
    difficulty_tier: dict[str, int] = Field(default_factory=dict)


class VerdictRecord(BaseModel):
    """Structured payload persisted for every Critic verdict (ADR-0006 boundary model).

    ``verdict`` mirrors :class:`foundry_x.evolution.critic.CriticVerdict.verdict`
    and is ``None`` when the gate was skipped via ``--no-verify`` (issue #888).
    The regression-pairing logic in :func:`_compute` treats ``None`` as a
    non-approval (falsy) so skipped gates do not seed ``prior_passed`` entries
    that would later flag unrelated failures as regressions.
    """

    verdict: bool | None = None
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    skipped_checks: list[str] = Field(default_factory=list)
    notes: str = ""
    failure_class: str | None = Field(default=None)
    target_file: str | None = Field(default=None)


@dataclass
class _Regression:
    task: str
    was_passing_session: str
    now_failing_session: str
    now_failing_version: str
    target_file: str | None = None


@dataclass
class _NewPass:
    task: str
    was_failing_session: str
    now_passing_session: str
    now_passing_version: str
    target_file: str | None = None


class RegressionRow(BaseModel):
    """One regressed task observed by the regression report (ADR-0006 boundary model)."""

    task: str
    was_passing_session: str
    now_failing_session: str
    now_failing_version: str
    target_file: str | None = None


class NewPassRow(BaseModel):
    """One task that began passing in the latest window (ADR-0006 boundary model)."""

    task: str
    was_failing_session: str
    now_passing_session: str
    now_passing_version: str
    target_file: str | None = None


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

    Issue #1114 adds ``slice_regressions``: per-dimension regression
    counts keyed by skill, task_family, or difficulty_tier. Only the
    dimension selected via ``group_by`` is populated; the others stay empty.
    """

    report: str
    total: int
    approvals: int
    rejections: int
    regressions: list[RegressionRow] = Field(default_factory=list)
    new_passes: list[NewPassRow] = Field(default_factory=list)
    token_budget_abort_count: int = 0
    slice_regressions: SliceRegressions = Field(default_factory=SliceRegressions)


def record_verdict(logger: TraceLogger, session_id: str, verdict: CriticVerdict) -> None:
    """Persist a CriticVerdict as a ``critic_verdict`` trace event."""
    record = VerdictRecord(
        verdict=verdict.verdict,
        passed_checks=list(verdict.passed_checks),
        failed_checks=list(verdict.failed_checks),
        skipped_checks=list(verdict.skipped_checks),
        notes=verdict.notes,
        failure_class=verdict.failure_class,
        target_file=verdict.target_file,
    )
    logger.record(session_id=session_id, kind=VERDICT_KIND, payload=record.model_dump())


def _load_verdict_events(
    logger: TraceLogger,
    since: str | None,
    harness_version: str | None = None,
) -> list[tuple[str, str, VerdictRecord]]:
    """Stream every ``critic_verdict`` event through :class:`TraceLogger`.

    Issue #273 — previously walked ``list_sessions()`` and called
    ``iter_events(sid)`` once per session, opening a fresh connection per
    session. :meth:`TraceLogger.query_events` collapses that nested loop
    into a single streaming cursor across all sessions.

    Issue #1270 — the ``since`` filter is pushed down to the store as a
    ``WHERE e.timestamp >= ?`` clause (sqlite) or an inline filter (jsonl)
    so time-bounded queries do not materialize events outside the window.
    """
    events: list[tuple[str, str, VerdictRecord]] = []
    for event in logger.query_events(
        kind=VERDICT_KIND, harness_version=harness_version, since=since
    ):
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
                        target_file=verdict.target_file,
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
                        target_file=verdict.target_file,
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
    harness_version: str | None = None,
    *,
    group_by: GroupByDim | None = None,
    task_metadata: dict[str, TaskKpiMetadata] | None = None,
) -> str:
    """Produce a Markdown regression report over all persisted Critic verdicts.

    When ``task`` is provided, only rows whose ``task`` column equals that
    name are included in the Regressed / New Passes sections (issue #182).
    The Regression Summary counts (total verdicts / approvals / rejections)
    remain the full population so the reviewer keeps context about the
    analysis pass. If the task filter eliminates every row, the rendered
    report collapses to a single ``no rows for task <name>`` line.

    Issue #1114: when ``group_by`` and ``task_metadata`` are provided,
    a ``Per-<Dim> Regressions`` section is appended to the report.
    """
    return analyze_regressions(
        logger,
        since=since,
        task=task,
        harness_version=harness_version,
        group_by=group_by,
        task_metadata=task_metadata,
    ).report


def analyze_regressions(
    logger: TraceLogger,
    since: str | None = None,
    task: str | None = None,
    harness_version: str | None = None,
    *,
    group_by: GroupByDim | None = None,
    task_metadata: dict[str, TaskKpiMetadata] | None = None,
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

    Issue #1114: ``group_by`` and ``task_metadata`` compute per-group
    regression counts and append a ``Per-<Dim> Regressions`` section to
    the Markdown report. The ``slice_regressions`` field on the returned
    ``RegressionAnalysis`` is populated with the matching dimension's dict.
    """
    events = _load_verdict_events(logger, since, harness_version=harness_version)
    total = len(events)
    approvals = sum(1 for _sid, _ts, v in events if v.verdict)
    rejections = total - approvals
    versions = _session_versions(logger)
    regressions, new_passes = _compute(events, versions)
    if task is not None:
        regressions = [r for r in regressions if r.task == task]
        new_passes = [p for p in new_passes if p.task == task]
    token_budget_abort_count = _count_token_budget_aborts(logger, since=since)

    slice_regressions = SliceRegressions()
    if group_by is not None and task_metadata:
        counts = _slice_regression_counts(
            logger,
            harness_version=harness_version,
            group_by=group_by,
            task_metadata=task_metadata,
        )
        if group_by == "skill":
            slice_regressions.skill = counts
        elif group_by == "task_family":
            slice_regressions.task_family = counts
        elif group_by == "difficulty_tier":
            slice_regressions.difficulty_tier = counts

    report = _render(
        total,
        approvals,
        rejections,
        regressions,
        new_passes,
        token_budget_abort_count=token_budget_abort_count,
        task=task,
        group_by=group_by,
        slice_regressions=slice_regressions,
    )
    return RegressionAnalysis(
        report=report,
        total=total,
        approvals=approvals,
        rejections=rejections,
        regressions=[RegressionRow(**asdict(r)) for r in regressions],
        new_passes=[NewPassRow(**asdict(p)) for p in new_passes],
        token_budget_abort_count=token_budget_abort_count,
        slice_regressions=slice_regressions,
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
    group_by: GroupByDim | None = None,
    slice_regressions: SliceRegressions | None = None,
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
        lines.append(
            "| Task | Was passing (session) | Now failing (session) | Manifest version | Target file |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for reg in regressions:
            target_file = reg.target_file if reg.target_file else ""
            lines.append(
                f"| {reg.task} | {reg.was_passing_session} | "
                f"{reg.now_failing_session} | {reg.now_failing_version} | {target_file} |"
            )
    else:
        lines.append("_None._")
    lines += ["", "## New Passes", ""]
    if new_passes:
        lines.append(
            "| Task | Was failing (session) | Now passing (session) | Manifest version | Target file |"
        )
        lines.append("| --- | --- | --- | --- | --- |")
        for pas in new_passes:
            target_file = pas.target_file if pas.target_file else ""
            lines.append(
                f"| {pas.task} | {pas.was_failing_session} | "
                f"{pas.now_passing_session} | {pas.now_passing_version} | {target_file} |"
            )
    else:
        lines.append("_None._")
    # Issue #466: token budget aborts are a distinct failure category,
    # reported separately from task regressions because they indicate a
    # task-sizing problem, not a harness defect.
    lines += ["", "## Token Budget Aborts", ""]
    lines.append(f"_Token budget aborts: {token_budget_abort_count} session(s)_")

    # Issue #1114: per-dimension regression breakdown.
    if group_by is not None and slice_regressions is not None:
        counts: dict[str, int] = {}
        dim_label = ""
        if group_by == "skill":
            counts = slice_regressions.skill
            dim_label = "Skill"
        elif group_by == "task_family":
            counts = slice_regressions.task_family
            dim_label = "Task Family"
        elif group_by == "difficulty_tier":
            counts = slice_regressions.difficulty_tier
            dim_label = "Difficulty Tier"
        if counts:
            lines += ["", f"## Per-{dim_label} Regressions", ""]
            lines.append(f"| {dim_label} | Regressions |")
            lines.append("| --- | --- |")
            for group_name in sorted(counts.keys()):
                lines.append(f"| {group_name} | {counts[group_name]} |")
        else:
            lines += ["", f"## Per-{dim_label} Regressions", ""]
            lines.append("_None._")

    lines.append("")
    return "\n".join(lines)


def _count_token_budget_aborts(
    logger: TraceLogger,
    since: str | None = None,
) -> int:
    """Count sessions with at least one ``task_aborted(reason="token_budget")`` or ``token_budget_aborted`` event.

    Issue #466: token budget aborts are task-shaped failures (a task exceeded
    the model's context budget), not harness regressions. They are reported
    as a separate signal in the regression report.

    Sessions are counted once regardless of how many times the abort fires
    within them. Uses one :meth:`TraceLogger.query_events` cursor.

    Issue #1270 — the ``since`` filter is pushed down to the store so
    time-bounded queries do not materialize events outside the window.

    Issue #1355: also counts the dedicated ``token_budget_aborted`` event.
    """
    sessions_with_abort: set[str] = set()
    for event in logger.query_events(kind=TASK_ABORTED_KIND, since=since):
        if event.payload.get("reason") == TOKEN_BUDGET_REASON:
            sessions_with_abort.add(event.session_id)
    # Issue #1355: also count the dedicated token_budget_aborted event
    for event in logger.query_events(kind=TOKEN_BUDGET_ABORTED_KIND, since=since):
        sessions_with_abort.add(event.session_id)
    return len(sessions_with_abort)


def _groups_for_task(meta: TaskKpiMetadata, group_by: GroupByDim) -> set[str]:
    """Return the set of group keys a task contributes to for *group_by*.

    Mirrors :func:`foundry_x.observability.kpis._groups_for_task` so the
    regression report can perform task-level groupings without importing the
    KPI module.
    """
    if group_by == "skill":
        return set(meta.skills)
    if group_by == "task_family":
        return set(meta.task_families)
    if group_by == "difficulty_tier":
        return {meta.difficulty_tier} if meta.difficulty_tier is not None else set()
    return set()


class _SliceAcc:
    """Mutable accumulator for one slice key (issue #1114)."""

    __slots__ = ("prior_passed", "regression_count", "sessions")

    def __init__(self) -> None:
        self.regression_count: int = 0
        self.sessions: set[str] = set()
        self.prior_passed: set[str] = set()


def _slice_regression_counts(
    logger: TraceLogger,
    harness_version: str | None,
    group_by: GroupByDim,
    task_metadata: dict[str, TaskKpiMetadata],
) -> dict[str, int]:
    """Count task regressions per group for a task-level dimension (issue #1114).

    Walks the same ``critic_verdict`` cursor as :func:`analyze_regressions`
    but attributes each verdict to the groups its tasks declare for the chosen
    *group_by* dimension.  A task contributes to a group's regression count
    when it appears in ``failed_checks`` after having appeared in
    ``passed_checks`` in an earlier verdict.

    A task can belong to several groups (e.g. several skills), so one
    regression may be counted against several groups.  Groups with no
    regressions are omitted from the returned dict.

    Returns ``{}`` when *group_by* is ``None`` or *task_metadata* is empty.
    """
    acc: dict[str, _SliceAcc] = {}
    for event in logger.query_events(kind=VERDICT_KIND, harness_version=harness_version):
        record = VerdictRecord(**event.payload)
        touched: set[str] = set()
        for task in (*record.passed_checks, *record.failed_checks):
            meta = task_metadata.get(task)
            if meta is not None:
                touched |= _groups_for_task(meta, group_by)
        for group in touched:
            bucket = acc.setdefault(group, _SliceAcc())
            bucket.sessions.add(event.session_id)
            for task in record.failed_checks:
                if task in bucket.prior_passed:
                    bucket.regression_count += 1
            for task in record.passed_checks:
                meta = task_metadata.get(task)
                if meta is not None and group in _groups_for_task(meta, group_by):
                    bucket.prior_passed.add(task)
    return {
        group: bucket.regression_count
        for group, bucket in acc.items()
        if bucket.regression_count > 0
    }


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
        if result.token_efficiency is not None and (
            best_eff is None or result.token_efficiency > best_eff
        ):
            best_eff = result.token_efficiency
        if result.cost_per_task is not None and (
            best_cost is None or result.cost_per_task < best_cost
        ):
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
