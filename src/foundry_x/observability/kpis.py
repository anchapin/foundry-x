"""Compute the three PRD success-metric KPIs from trace data.

The PRD (``docs/PRD.md`` §5) defines:

* **Cycle Time** — time from *Agent Failure* to *Harness Edit Proposal*.
* **Regression Rate** — number of previously-solved tasks that break after
  a harness edit.
* **Improvement Rate** — success rate on a standardized benchmark before
  vs. after harness evolution.

This module derives approximations of those metrics from the events already
recorded by :class:`~foundry_x.trace.logger.TraceLogger`:

* ``cycle_time_seconds`` — the operational proxy: mean wall-clock time from
  the first ``task_received`` event to the first ``critic_verdict`` event
  per session (the closest measurable proxy for the business-level "Agent
  Failure" → "Harness Edit Proposal" definition above).
* ``regression_rate`` — fraction of sessions with a ``critic_verdict`` in which
  a task previously seen in ``passed_checks`` later appears in ``failed_checks``
  (the persisted :class:`~foundry_x.observability.regression_report.VerdictRecord`
  shape).
* ``improvement_rate`` — fraction of ``critic_verdict`` events whose persisted
  payload has ``approved: true``.

When the source events are absent the function degrades gracefully,
returning ``None`` (cycle time) or ``0.0`` so the CLI can print ``N/A``.

Issue #120 adds an auxiliary per-session ``injection_blocked`` count derived
from the firewall events persisted by ``InjectionFirewallHook``. The
counts are surfaced only when at least one session has ≥1 block, so a
clean store does not grow the KPI output.

Issue #82: this module previously opened a raw ``sqlite3`` connection on
``logger.path`` and issued bespoke ``SELECT`` statements — see ADR-0003
("No raw SQL strings in business logic"). The store schema is now reached
exclusively through :class:`TraceLogger`'s ``list_sessions`` and
``iter_events`` methods, which own the row format and yield events one at
a time so a future streaming caller does not have to load everything.

Issue #183: an append-only JSONL history log (``--log-to`` /
``--from-history``) gives the regression signal a temporal axis —
operators can see cycle time drifting across harness edits without
manually diffing four JSON snapshots. The per-session
``injection_blocks`` map is intentionally excluded from history
entries; the trend table is a one-row-per-run summary, not a
per-session inventory.

Issue #898: ``compute_kpis`` accepts a ``group_by`` parameter
(``"skill"`` / ``"task_family"`` / ``"difficulty_tier"``) plus a
``task_metadata`` map and populates a matching ``per_*`` field on
:class:`KpiSummary` with per-group ``improvement_rate`` and
``regression_rate`` slices. The CLI exposes this via ``--group-by``
(and optional ``--task-metadata``); the slices are an on-demand
diagnostic view and are excluded from the history log.

Issue #895: ``cycle_time_seconds`` only counts sessions that produced a
``critic_verdict``; sessions that fail before the Critic runs (model
errors, early wall-clock / event-limit / token-budget aborts) are
silently excluded, creating survivorship bias. :class:`KpiSummary` now
carries ``excluded_from_cycle_time`` — the count of sessions that had a
``task_received`` but did not contribute a positive delta to the mean —
so an operator can tell whether the mean reflects the full population or
a self-selected subpopulation of survivors.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, ValidationError

from foundry_x.evolution.digester import INJECTION_BLOCKED_KIND
from foundry_x.evolution.loop import EVOLVER_DURATION_KIND
from foundry_x.observability.regression_report import VerdictRecord
from foundry_x.trace.logger import TraceEvent, TraceLogger

KPI_HISTORY_SCHEMA_VERSION = 1


def _get_trace_db(args: argparse.Namespace) -> str:
    """Return the trace-db path, emitting a deprecation warning if --db was used."""
    if getattr(args, "db", None) is not None:
        warnings.warn(
            "--db is deprecated; use --trace-db instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return args.db
    return args.trace_db


TASK_ABORTED_KIND = "task_aborted"
TOKEN_BUDGET_REASON = "token_budget"
# Issue #1355: the runner emits ``token_budget_aborted`` as a dedicated
# terminal failure marker when the running token total exceeds the budget.
# This constant centralizes the kind spelling for KPI and regression consumers.
TOKEN_BUDGET_ABORTED_KIND = "token_budget_aborted"
# Issue #869: the runner emits ``task_aborted(reason="event_limit")`` when the
# per-session event cap is exceeded (see ``execution/runner.py:1523``). The
# constant lives next to ``TOKEN_BUDGET_REASON`` so any future reference
# (Digester classification, regression report, etc.) shares the same spelling
# without re-typing the literal.
EVENT_LIMIT_REASON = "event_limit"


CONTEXT_PRUNED_KIND = "context_pruned"
# Issue #871: the runner emits ``model_retry`` whenever a transient model API
# failure is retried (see ``execution/runner.py:1449``). Keep the kind spelling
# centralized so KPI and session-card aggregation cannot drift.
MODEL_RETRY_KIND = "model_retry"
# Issue #872: the runner emits ``tool_argument_parse_error`` when the model
# produces malformed tool-call arguments (see ``execution/runner.py:1684``).
# The constant lives next to the other kind vocabulary strings so any future
# reference (Digester classification, regression report, etc.) shares the
# same spelling without re-typing the literal.
TOOL_ARGUMENT_PARSE_ERROR_KIND = "tool_argument_parse_error"
# Issue #899: the runner emits ``server_unavailable`` when the
# ``FoundryServerManager`` reports ``/health`` returning a non-200 status
# mid-session and triggers the supervisor's restart loop. The
# ``server_restart_count`` KPI is the cumulative number of such events
# across the trace store.
SERVER_UNAVAILABLE_KIND = "server_unavailable"

# Issue #953: the evolver emits ``generation_exhausted`` when all LLM
# generation retries have been exhausted without producing a valid edit.
# The ``evolver_llm_failure_count`` KPI is the total number of such events;
# ``evolver_llm_failure_rate`` is the fraction of sessions with at least one.
GENERATION_EXHAUSTED_KIND = "generation_exhausted"

# Issue #1281: the runner emits ``model_cost`` when the CloudModelAdapter
# receives a cost event from the provider. The ``model_cost_count`` KPI is
# the total number of such events; ``total_model_cost_usd`` is the cumulative
# estimated cost in USD.
MODEL_COST_KIND = "model_cost"
# Issue #1281: the runner emits ``model_rate_limit`` when the CloudModelAdapter
# receives a rate-limit update from the provider. The ``model_rate_limit_count``
# KPI is the total number of such events.
MODEL_RATE_LIMIT_KIND = "model_rate_limit"
# Issue #1281: the runner emits ``fetch_blocked`` when the WebFetchHook
# detects a URL whose host is not in FETCH_ALLOWED_DOMAINS. The
# ``fetch_blocked_count`` KPI is the total number of such events.
FETCH_BLOCKED_KIND = "fetch_blocked"

#: Dimension accepted by :func:`compute_kpis`'s ``group_by`` parameter
#: (issue #898, #1039). Each value selects which field drives the
#: per-slice breakdown of ``improvement_rate`` and ``regression_rate``.
#:
#: *Task-level* dimensions (``skill`` / ``task_family`` /
#: ``difficulty_tier``) slice by :class:`TaskKpiMetadata` attributes.
#:
#: *Session-level* dimensions (``model_id`` / ``quantization`` /
#: ``harness_version``) slice by
#: :class:`~foundry_x.trace.logger.TraceSession` attributes — verdicts
#: are bucketed by the session's model/quantization/harness fields
#: rather than task metadata.
GroupByDim = Literal[
    "skill",
    "task_family",
    "difficulty_tier",
    "model_id",
    "quantization",
    "harness_version",
]


class TaskKpiMetadata(BaseModel):
    """Grouping metadata for one benchmark task (ADR-0006 boundary model).

    Issue #898 — the per-skill / per-task-family / per-difficulty-tier KPI
    slices need a way to attribute each task name that appears in a
    ``critic_verdict``'s ``passed_checks`` / ``failed_checks`` to the
    dimensions declared on its :class:`~benchmarks.models.BenchmarkTask`
    (``requires_skills``, ``tags``, ``difficulty_tier``). The KPI layer
    must not import ``benchmarks`` at module load time — the dependency
    runs the other way (``benchmarks`` depends on ``foundry_x``, as in
    ``src/foundry_x/evolution/critic.py``) — so this model is the boundary
    contract: the CLI builds it via :func:`build_task_metadata` and hands
    it to :func:`compute_kpis`.

    A task with no declared metadata for a dimension simply contributes
    no groups for that dimension — the verdict is then invisible to the
    corresponding slice, which is the desired graceful degradation.
    """

    name: str
    skills: list[str] = Field(default_factory=list)
    task_families: list[str] = Field(default_factory=list)
    difficulty_tier: str | None = None


class SkillKpiSlice(BaseModel):
    """Per-group ``improvement_rate`` / ``regression_rate`` for one slice key (issue #898).

    Despite the ``Skill`` prefix this model is reused for all three
    grouping dimensions (``skill``, ``task_family``, ``difficulty_tier``);
    the name matches the issue's acceptance criterion (#2) which calls out
    the per-skill case explicitly. ADR-0006 places it at the module
    boundary so JSON consumers and the ``foundry-kpis`` CLI share one
    contract.

    The two rate fields follow the same definition as the aggregate KPIs,
    just over the subpopulation of verdicts whose checks touch the group:

    * ``improvement_rate`` — approved verdicts attributed to the group /
      total verdicts attributed to the group.
    * ``regression_rate`` — sessions attributed to the group in which a
      task belonging to the group regressed / sessions attributed to the
      group with a verdict.

    A verdict is *attributed to* a group when any of its checks names a
    task whose metadata lists that group. Because a task may declare
    several skills (or tags), a single verdict can be attributed to
    several groups — the slices are independent views, not a partition,
    so their verdict/session counts need not sum to the aggregate. In
    :attr:`KpiComparison.slice_deltas` the rate fields carry the
    candidate-minus-baseline delta (sign-agnostic, as with the aggregate
    deltas) and the counts carry the candidate side for reference.
    """

    improvement_rate: float = 0.0
    regression_rate: float = 0.0
    verdict_count: int = 0
    session_count: int = 0


class KpiSummary(BaseModel):
    """Structured summary of the three PRD KPIs.

    Issue #1338 adds ``cycle_time_p50_seconds`` and ``cycle_time_p95_seconds``:
    the 50th and 95th percentiles of the per-session ``task_received`` →
    ``critic_verdict`` deltas, complementing the mean in
    ``cycle_time_seconds`` so operators can tell whether a mean shift reflects
    a uniform change across all sessions or a long-tail of outlier sessions.

    Issue #120 adds ``injection_blocks``: a ``session_id -> count`` map
    of ``injection_blocked`` events per session, sourced from the firewall
    hook. Empty by default; populated only when the trace store has at
    least one ``injection_blocked`` event.

    Issue #271 adds ``token_totals``: a ``session_id -> int`` map of the
    cumulative ``total_tokens`` consumed per session, summed from the
    ``usage`` payloads the runner records on each ``model_response`` event
    (issue #191). Empty by default; populated only when at least one
    ``model_response`` event carries a ``usage`` dict, so a trace store
    with no token data (e.g. an endpoint that never reports usage) keeps
    the summary compact. Like ``injection_blocks`` this is an auxiliary
    operator signal, not one of the three PRD success-metric KPIs.

    Issue #604 adds ``evolver_duration_ms``: mean wall-clock milliseconds
    spent inside ``evolver.propose()`` per session, sourced from the
    ``evolver_duration_ms`` field of :class:`~foundry_x.evolution.loop.EvolutionResult`.
    ``None`` when no evolver phase was recorded for any session.

    Issue #585 adds ``hooks_disabled_count`` and ``hooks_disabled_rate``:
    the total count of ``hook_registry_error`` events and the fraction of
    sessions with at least one such event. Emitted when
    ``harness.hooks.get_registry()`` raises, disabling all hooks including
    the security-critical ``InjectionFirewallHook``.

    Issue #466 adds ``token_budget_abort_count``: the number of sessions
    that recorded at least one ``task_aborted(reason="token_budget")``
    event. Surfaced as an auxiliary operator signal alongside
    ``injection_blocks`` and ``token_totals``.

    Issue #551 adds ``token_budget_hit_rate``: the fraction of sessions
    that recorded at least one ``task_aborted(reason="token_budget")``
    event. This is a fourth tracked metric exposed via ``foundry-kpis``
    and the regression report, alongside the three PRD KPIs.

    Issue #580 adds ``streaming_quality``: a ``session_id ->
    StreamingQualityData`` map of per-session streaming quality metrics
    (avg TTFT, chunk count, avg chunk interval) derived from the timing
    fields on each ``model_response`` event. Empty by default; populated
    only when at least one ``model_response`` event carries timing data.

    Issue #951 adds ``context_efficiency``: mean per-session
    ``1 - (sum(dropped) / sum(threshold + dropped))`` across all sessions
    (ADR-0021 §6), sourced from the ``dropped`` and ``threshold`` fields
    of ``context_pruned`` events. Issue #979 fixes a survivorship-bias bug:
    a session that never pruned (zero ``context_pruned`` events) now
    contributes ``1.0`` (perfect efficiency — nothing was dropped) instead
    of being silently excluded. ``None`` only when the trace store has no
    sessions. Near 1.0 means pruning rarely fired; near 0.0 means heavy
    pruning throughout sessions.

    Issue #626 adds ``context_pruned_count``: a ``session_id -> count`` map
    of ``context_pruned`` events per session, sourced from the pruning hook.
    Empty by default; populated only when at least one ``context_pruned``
    event has been recorded. Like ``injection_blocks`` and ``token_totals``
    this is an auxiliary operator signal.

    Issue #800 adds ``hooks_disabled_rate`` and ``wall_clock_abort_count``:
    the fraction of sessions that recorded a ``hook_registry_error`` event
    and the total count of ``task_aborted`` events whose ``reason`` is
    ``"wall_clock"``, respectively.

    Issue #871 adds ``model_retry_count``: the total number of
    ``model_retry`` events emitted when a transient model API failure is
    retried. A rising count signals provider flakiness or API reliability
    degradation, so it is exposed as an auxiliary operator signal.

    Issue #872 adds ``tool_argument_parse_error_count``: the total number
    of ``tool_argument_parse_error`` events emitted by the runner when the
    model produces tool-call arguments that cannot be parsed as JSON. A
    rising rate signals model output quality degradation or a mismatch
    between the tool schema and the model's capabilities, so the counter
    is surfaced alongside ``wall_clock_abort_count`` as an auxiliary
    operator signal.

    Issue #869 adds ``event_limit_abort_count``: the total count of
    ``task_aborted(reason="event_limit")`` events emitted by the runner
    when the per-session event cap (``FOUNDRY_MAX_EVENTS_PER_SESSION``)
    is exceeded. A rising rate signals runaway or looping agent
    behavior — the agent produced more events than the harness expected,
    which usually means the context-pruning or stop-on-error policies
    are not strict enough. Surfaced alongside ``token_budget_abort_count``
    and ``wall_clock_abort_count`` as an auxiliary operator signal.

    Issue #899 adds ``server_restart_count``: the total count of
    ``server_unavailable`` events emitted by the runner when the
    ``FoundryServerManager`` reports a mid-session ``/health`` failure
    and triggers the supervisor's restart loop. A rising rate signals
    that llama-server (or the local model backend) is crashing or
    becoming unreachable mid-benchmark — surfacing it as an auxiliary
    operator signal so operators can correlate cycle-time regressions
    with infrastructure reliability issues without confusing this with
    model-quality signals.

    Issue #895 adds ``excluded_from_cycle_time``: the number of sessions
    that have a ``task_received`` event but did not contribute a positive
    ``task_received`` → ``critic_verdict`` delta to ``cycle_time_seconds``
    — i.e. sessions that failed before the Critic ran (model errors,
    early wall-clock / event-limit / token-budget aborts), plus the rare
    session whose timestamps could not be parsed or whose delta was
    non-positive. Surfaced as an auxiliary operator signal so the
    survivorship bias in ``cycle_time_seconds`` (which reflects only
    successful evolutions) is interpretable: a high exclusion count means
    the mean is computed over a small, self-selected subpopulation.

    Issue #1113 adds ``excluded_wall_clock``, ``excluded_token_budget``,
    ``excluded_event_limit``, and ``excluded_other``: the breakdown of
    ``excluded_from_cycle_time`` by the ``reason`` field of the
    ``task_aborted`` event that caused each session's exclusion.  Sessions
    that were excluded but have no ``task_aborted`` event (e.g. a session
    whose timestamps could not be parsed or whose delta was non-positive)
    contribute to ``excluded_other``.

    Issue #898 adds ``per_skill`` / ``per_task_family`` /
    ``per_difficulty_tier``: ``dict[str, SkillKpiSlice]`` breakdowns of
    ``improvement_rate`` and ``regression_rate``. Only the dimension
    selected via :func:`compute_kpis`'s ``group_by`` parameter is
    populated; the other two stay empty so the JSON snapshot stays
    compact and the selected dimension is unambiguous. See
    :class:`SkillKpiSlice` for the verdict-attribution semantics.

    Issue #953 adds ``evolver_llm_failure_count`` and ``evolver_llm_failure_rate``:
    the total number of ``generation_exhausted`` events emitted by the evolver
    when all LLM generation retries have been exhausted without producing a valid
    edit, and the fraction of sessions with at least one such event. A rising
    rate signals LLM provider flakiness or model degradation that prevents the
    evolver from proposing harness edits — surfaced as an auxiliary operator
    signal alongside ``model_retry_count`` and ``tool_argument_parse_error_count``.

    Issue #1112 adds ``token_budget_overrun_pct``: the mean percentage by which
    sessions that hit ``task_aborted(reason="token_budget")`` exceeded their
    token budget, i.e. ``mean((tokens_used - token_budget) / token_budget)``
    across aborted sessions. Returns ``None`` when no session hit the token
    budget, so operators can distinguish a clean store (None) from one where
    all sessions exceeded their budgets (a real percentage).

    Issue #1281 adds ``model_cost_count`` and ``total_model_cost_usd``: the
    total number of ``model_cost`` events and the cumulative estimated cost in
    USD, sourced from the ``estimated_cost_usd`` field on each event. Also adds
    ``model_rate_limit_count``: the total number of ``model_rate_limit`` events.
    And ``fetch_blocked_count``: the total number of ``fetch_blocked`` events
    emitted when the WebFetchHook blocks a URL not in FETCH_ALLOWED_DOMAINS.
    All three are auxiliary operator signals surfaced alongside
    ``model_retry_count`` and ``server_restart_count``.

    Issue #1271 adds ``streaming_quality_mean_ttft_ms``,
    ``streaming_quality_p50_ttft_ms``, and ``streaming_quality_p95_ttft_ms``:
    aggregate TTFT statistics across all sessions, plus
    ``mean_prompt_tokens_per_step`` and ``mean_completion_tokens_per_step``
    for token efficiency analysis per model response step.
    """

    cycle_time_seconds: float | None = None
    # Issue #1338: p50 and p95 of per-session cycle-time deltas.
    cycle_time_p50_seconds: float | None = None
    cycle_time_p95_seconds: float | None = None
    regression_rate: float = 0.0
    improvement_rate: float = 0.0
    injection_blocks: dict[str, int] = {}
    token_totals: dict[str, int] = {}
    evolver_duration_ms: float | None = None
    hooks_disabled_count: int = 0
    hooks_disabled_rate: float = 0.0
    token_budget_abort_count: int = 0
    token_budget_hit_rate: float = 0.0
    context_efficiency: float | None = None
    streaming_quality: dict[str, StreamingQualityData] = {}
    context_pruned_count: dict[str, int] = {}
    wall_clock_abort_count: int = 0
    failure_class_distribution: dict[str, int] = {}
    model_retry_count: int = 0
    tool_argument_parse_error_count: int = 0
    event_limit_abort_count: int = 0
    server_restart_count: int = 0
    excluded_from_cycle_time: int = 0
    excluded_wall_clock: int = 0
    excluded_token_budget: int = 0
    excluded_event_limit: int = 0
    excluded_other: int = 0
    evolver_llm_failure_count: int = 0
    evolver_llm_failure_rate: float = 0.0
    token_budget_overrun_pct: float | None = None
    model_cost_count: int = 0
    total_model_cost_usd: float = 0.0
    model_rate_limit_count: int = 0
    fetch_blocked_count: int = 0
    # Issue #1271: aggregate streaming quality across all sessions for the
    # given harness version.  ``streaming_quality_mean_ttft_ms`` is the mean
    # of per-session average TTFT values; ``streaming_quality_p50_ttft_ms``
    # and ``streaming_quality_p95_ttft_ms`` are the 50th and 95th percentiles
    # of those per-session averages.  ``mean_prompt_tokens_per_step`` and
    # ``mean_completion_tokens_per_step`` are the mean prompt and completion
    # token counts per ``model_response`` step, aggregated across all sessions.
    streaming_quality_mean_ttft_ms: float | None = None
    streaming_quality_p50_ttft_ms: float | None = None
    streaming_quality_p95_ttft_ms: float | None = None
    mean_prompt_tokens_per_step: float | None = None
    mean_completion_tokens_per_step: float | None = None
    per_skill: dict[str, SkillKpiSlice] = {}
    per_task_family: dict[str, SkillKpiSlice] = {}
    per_difficulty_tier: dict[str, SkillKpiSlice] = {}
    # Issue #1039: session-level KPI slices.  Unlike the task-level slices
    # above (keyed by task metadata), these are keyed by session attributes
    # (model_id, quantization, harness_version) and bucket verdicts by the
    # session that produced them.  Only the dimension selected via
    # ``group_by`` is populated; the others stay empty.
    per_model_id: dict[str, SkillKpiSlice] = {}
    per_quantization: dict[str, SkillKpiSlice] = {}
    per_harness_version: dict[str, SkillKpiSlice] = {}
    # Issue #1269: hook overhead percentiles per tool, computed from
    # ``hook_overhead_ms`` and ``hook_post_overhead_ms`` on tool_call events.
    hook_overhead: dict[str, HookOverheadRow] = {}
    # Issue #1269: aggregate hook overhead percentiles across all tools for
    # ``foundry-kpis --format json`` output (the dict above is the per-tool
    # breakdown; these scalars are the overall aggregates).
    hook_overhead_ms_p50: float = 0.0
    hook_overhead_ms_p95: float = 0.0
    hook_post_overhead_ms_p50: float = 0.0
    hook_post_overhead_ms_p95: float = 0.0


class StreamingQualityData(BaseModel):
    """Streaming quality metrics for one session (issue #580).

    Aggregated from the timing fields on each ``model_response`` event:
    ``time_to_first_token_ms``, ``chunk_count``, and ``total_stream_ms``.
    """

    avg_ttft_ms: float | None = None
    total_chunks: int = 0
    avg_chunk_interval_ms: float | None = None


class HookOverheadRow(BaseModel):
    """Hook overhead percentiles for one tool (issue #1269).

    Aggregated from the ``hook_overhead_ms`` and ``hook_post_overhead_ms``
    fields on every ``tool_call`` event the Runner emits. Computed via one
    ``query_events`` cursor with the kind filter pushed down.
    """

    count: int = 0
    hook_overhead_ms_p50: float = 0.0
    hook_overhead_ms_p95: float = 0.0
    hook_post_overhead_ms_p50: float = 0.0
    hook_post_overhead_ms_p95: float = 0.0


class KpiComparison(BaseModel):
    """Baseline-vs-candidate harness-version comparison (issue #100).

    ``deltas`` holds the raw ``candidate - baseline`` difference for each
    numeric KPI; the rendering layer interprets the sign per the PRD's
    "good direction" — improvement-rate up is good, regression-rate and
    cycle-time down are good. ``injection_blocks`` is intentionally
    excluded from the comparison because it is an auxiliary signal, not
    one of the three PRD success-metric KPIs.

    Issue #736 adds ``baseline_session_count`` and ``candidate_session_count``
    so that callers can distinguish "no change" (deltas near 0.0 with real
    sessions) from "no data" (deltas are 0.0 because one version has zero
    sessions in the trace store).

    Issue #898 adds ``slice_deltas``: per-group candidate-minus-baseline
    deltas keyed by grouping dimension (``"skill"`` / ``"task_family"`` /
    ``"difficulty_tier"``), then by group name. Only populated when
    :func:`compare_kpis` is called with ``group_by`` set; each
    :class:`SkillKpiSlice` carries the delta in its rate fields and the
    candidate's verdict/session counts for reference.
    """

    baseline: KpiSummary
    candidate: KpiSummary
    deltas: dict[str, float | None]
    baseline_session_count: int = 0
    candidate_session_count: int = 0
    slice_deltas: dict[str, dict[str, SkillKpiSlice]] = {}


class KpiHistoryEntry(BaseModel):
    """One line in the append-only KPI history log (issue #183).

    Mirrors the scalar fields that :func:`append_kpi_history` writes to
    disk: the three PRD-KPI fields from :class:`KpiSummary`, a
    ``timestamp`` (ISO-8601, stamped at append time), an optional
    ``harness_version`` (preserved when the operator filtered the run
    with ``--harness-version``), and the scalar auxiliary reliability
    signals.  Per-session maps (``injection_blocks``, ``token_totals``,
    ``streaming_quality``, ``context_pruned_count``) and recomputed
    coverage / slice fields are intentionally absent — the history is a
    one-row-per-run summary, and per-session inventory is the trace
    store's job.

    Issue #585 adds ``hooks_disabled_count`` and ``hooks_disabled_rate``.

    Issue #933 adds the six auxiliary signals that were previously
    written by :func:`append_kpi_history` but silently dropped on
    read-back (pydantic ``extra='ignore'``): ``model_retry_count``,
    ``tool_argument_parse_error_count``, ``event_limit_abort_count``,
    ``server_restart_count``, ``token_budget_abort_count``, and
    ``token_budget_hit_rate``.  Two dead fields (``injection_blocks``
    and ``wall_clock_abort_count``) were removed because they are in
    the exclude set and never appear in the JSONL line; any consumer
    already sees their defaults.

    Issue #953 adds ``evolver_llm_failure_count`` and ``evolver_llm_failure_rate``.

    Issue #1112 adds ``token_budget_overrun_pct``: the mean percentage by which
    sessions that hit ``task_aborted(reason="token_budget")`` exceeded their
    token budget. ``None`` when no session hit the token budget.

    Issue #1281 adds ``model_cost_count``, ``total_model_cost_usd``,
    ``model_rate_limit_count``, and ``fetch_blocked_count``.

    Issue #1271 adds ``streaming_quality_mean_ttft_ms``,
    ``streaming_quality_p50_ttft_ms``, and ``streaming_quality_p95_ttft_ms``,
    plus ``mean_prompt_tokens_per_step`` and ``mean_completion_tokens_per_step``.

    Issue #1269 adds the four aggregate hook overhead percentiles:
    ``hook_overhead_ms_p50``, ``hook_overhead_ms_p95``,
    ``hook_post_overhead_ms_p50``, ``hook_post_overhead_ms_p95``. These are
    the p50/p95 of all tool_call events' ``hook_overhead_ms`` and
    ``hook_post_overhead_ms`` fields, respectively, across all tools in
    the analysis window.

    Issue #1338 adds ``cycle_time_p50_seconds`` and ``cycle_time_p95_seconds``:
    the 50th and 95th percentiles of per-session cycle-time deltas.

    Issue #1334 adds ``schema_version`` so readers built against an older
    schema can detect and warn about unknown fields rather than silently
    dropping them via pydantic's ``extra='ignore'`` default.
    """

    schema_version: int = 1
    timestamp: str
    harness_version: str | None = None
    cycle_time_seconds: float | None = None
    cycle_time_p50_seconds: float | None = None
    cycle_time_p95_seconds: float | None = None
    regression_rate: float = 0.0
    improvement_rate: float = 0.0
    hooks_disabled_count: int = 0
    hooks_disabled_rate: float = 0.0
    token_budget_abort_count: int = 0
    token_budget_hit_rate: float = 0.0
    token_budget_overrun_pct: float | None = None
    model_retry_count: int = 0
    tool_argument_parse_error_count: int = 0
    event_limit_abort_count: int = 0
    server_restart_count: int = 0
    failure_class_distribution: dict[str, int] = {}
    evolver_llm_failure_count: int = 0
    evolver_llm_failure_rate: float = 0.0
    model_cost_count: int = 0
    total_model_cost_usd: float = 0.0
    model_rate_limit_count: int = 0
    fetch_blocked_count: int = 0
    streaming_quality_mean_ttft_ms: float | None = None
    streaming_quality_p50_ttft_ms: float | None = None
    streaming_quality_p95_ttft_ms: float | None = None
    mean_prompt_tokens_per_step: float | None = None
    mean_completion_tokens_per_step: float | None = None
    # Issue #1269: aggregate hook overhead percentiles across all tools.
    hook_overhead_ms_p50: float = 0.0
    hook_overhead_ms_p95: float = 0.0
    hook_post_overhead_ms_p50: float = 0.0
    hook_post_overhead_ms_p95: float = 0.0


class KpiTrends(BaseModel):
    """Computed trend statistics from KPI history entries (issue #1031).

    Derives direction, slope, and percent change for each of the three PRD KPIs
    by fitting a simple linear regression over the history entries.  Entries
    are weighted equally (ordinary least squares); no entry-level weights are
    applied.  ``None`` values in the source data are excluded from the fit.

    The trend direction uses the PRD "good direction" convention:

    * Cycle time: decreasing is *improving* (good), increasing is *worsening*.
    * Regression rate: decreasing is *improving*, increasing is *worsening*.
    * Improvement rate: increasing is *improving*, decreasing is *worsening*.

    ``entry_count`` and ``time_span_seconds`` let operators distinguish a
    meaningful trend (many entries over a long span) from a noisy two-point
    snapshot.
    """

    cycle_time_slope: float | None = None  # seconds per entry
    cycle_time_direction: Literal["improving", "worsening", "stable", "insufficient_data"] = (
        "insufficient_data"
    )
    cycle_time_percent_change: float | None = None  # from first to last valid entry

    regression_rate_slope: float | None = None  # per entry
    regression_rate_direction: Literal["improving", "worsening", "stable", "insufficient_data"] = (
        "insufficient_data"
    )
    regression_rate_percent_change: float | None = None

    improvement_rate_slope: float | None = None  # per entry
    improvement_rate_direction: Literal["improving", "worsening", "stable", "insufficient_data"] = (
        "insufficient_data"
    )
    improvement_rate_percent_change: float | None = None

    entry_count: int = 0
    time_span_seconds: float | None = None  # first to last timestamp


class _TrendField:
    """Helper to compute trend stats for one scalar field."""

    __slots__ = ("first_value", "last_value", "times", "values")

    def __init__(self) -> None:
        self.values: list[float] = []
        self.times: list[float] = []  # seconds from first entry
        self.first_value: float | None = None
        self.last_value: float | None = None

    def add(self, value: float | None, timestamp: str) -> None:
        if value is None:
            return
        self.values.append(value)
        if self.first_value is None:
            self.first_value = value
        self.last_value = value
        # Parse ISO timestamp to seconds from epoch
        dt = datetime.fromisoformat(timestamp)
        self.times.append(dt.timestamp())


def compute_trends(entries: Sequence[KpiHistoryEntry]) -> KpiTrends:
    """Compute trend statistics from KPI history entries (issue #1031).

    Fits a simple linear regression over each of the three PRD KPI fields.
    Returns a :class:`KpiTrends` object with slope, direction, and percent
    change.  ``None`` values are excluded from the fit.

    An *improving* direction means the KPI moved toward its PRD "good"
    direction; a *worsening* direction means it moved away.  ``stable``
    is returned when the absolute slope is below a small epsilon threshold.
    ``insufficient_data`` is returned when fewer than two valid data points
    exist for the field.

    Parameters
    ----------
    entries:
        History entries in chronological order (oldest first).  Typically
        produced by :func:`read_kpi_history`.

    Returns
    -------
    A :class:`KpiTrends` with per-KPI trend fields and aggregate metadata
    (``entry_count``, ``time_span_seconds``).
    """
    if not entries:
        return KpiTrends()

    # Time of first entry (epoch seconds) for computing relative spans.
    first_dt = datetime.fromisoformat(entries[0].timestamp)
    first_ts = first_dt.timestamp()

    # Collect valid data points for each field.
    cycle = _TrendField()
    reg = _TrendField()
    imp = _TrendField()

    for entry in entries:
        cycle.add(entry.cycle_time_seconds, entry.timestamp)
        reg.add(entry.regression_rate, entry.timestamp)
        imp.add(entry.improvement_rate, entry.timestamp)

    # Compute time span
    last_dt = datetime.fromisoformat(entries[-1].timestamp)
    time_span = last_dt.timestamp() - first_ts if len(entries) > 1 else None

    return KpiTrends(
        cycle_time_slope=_slope(cycle.values, cycle.times),
        cycle_time_direction=_trend_direction(
            cycle.values, cycle.first_value, cycle.last_value, higher_is_better=False
        ),
        cycle_time_percent_change=_percent_change(cycle.first_value, cycle.last_value),
        regression_rate_slope=_slope(reg.values, reg.times),
        regression_rate_direction=_trend_direction(
            reg.values, reg.first_value, reg.last_value, higher_is_better=False
        ),
        regression_rate_percent_change=_percent_change(reg.first_value, reg.last_value),
        improvement_rate_slope=_slope(imp.values, imp.times),
        improvement_rate_direction=_trend_direction(
            imp.values, imp.first_value, imp.last_value, higher_is_better=True
        ),
        improvement_rate_percent_change=_percent_change(imp.first_value, imp.last_value),
        entry_count=len(entries),
        time_span_seconds=time_span,
    )


def _slope(values: list[float], times: list[float]) -> float | None:
    """Simple linear regression slope (delta per second).

    Returns ``None`` when fewer than two points are available.
    Uses the standard OLS formula for a line through the origin (times
    are already relative to the first entry).
    """
    if len(values) < 2 or len(times) < 2:
        return None
    n = len(values)
    sum_x = sum(times)
    sum_y = sum(values)
    sum_xy = sum(t * v for t, v in zip(times, values))
    sum_x2 = sum(t * t for t in times)
    denom = n * sum_x2 - sum_x * sum_x
    if abs(denom) < 1e-12:
        return 0.0
    return (n * sum_xy - sum_x * sum_y) / denom


def _trend_direction(
    values: list[float],
    first: float | None,
    last: float | None,
    higher_is_better: bool,
) -> Literal["improving", "worsening", "stable", "insufficient_data"]:
    if len(values) < 2 or first is None or last is None:
        return "insufficient_data"
    # Use the first-to-last delta as the primary trend signal.
    # The percent change is computed separately; this only determines direction.
    delta = last - first
    eps = 1e-9
    if abs(delta) < eps:
        return "stable"
    # For higher_is_better=True: positive delta = improving.
    # For higher_is_better=False: negative delta = improving.
    if (delta > 0) is higher_is_better:
        return "improving"
    return "worsening"


def percentile(sorted_values: list[float], q: float) -> float:
    """Return the *q*-th percentile of *sorted_values* using nearest-rank.

    *sorted_values* must be in ascending order; the function does not
    re-sort. Nearest-rank (a.k.a. ``ceil(q/100 * n)``) is deliberately
    chosen over linear interpolation: it is deterministic, has no
    floating-point interpolation edge cases, and matches the operator
    intuition "p95 means the worst of the top 5%".

    Returns ``0.0`` for an empty input.
    """
    n = len(sorted_values)
    if n == 0:
        return 0.0
    rank = max(1, math.ceil(q / 100.0 * n))
    return float(sorted_values[min(rank, n) - 1])


def _percent_change(first: float | None, last: float | None) -> float | None:
    """Percent change from first to last value, or None if first is zero/None."""
    if first is None or last is None:
        return None
    if abs(first) < 1e-12:
        return None
    return ((last - first) / abs(first)) * 100.0


def render_trends_markdown(trends: KpiTrends) -> str:
    """Render trend statistics as a Markdown summary (issue #1031).

    Renders a compact table with one row per KPI showing the direction,
    slope (per-entry delta), and percent change from first to last entry.
    An ``entry_count`` footer describes how many history entries underpin
    the trend.
    """
    if trends.entry_count == 0:
        return "_No KPI history entries — cannot compute trends._"

    lines: list[str] = [
        "| KPI | Direction | Slope (per entry) | % Change |",
        "| --- | --- | --- | --- |",
        _trend_row(
            "Cycle Time (s)",
            trends.cycle_time_direction,
            trends.cycle_time_slope,
            trends.cycle_time_percent_change,
            higher_is_better=False,
        ),
        _trend_row(
            "Regression Rate",
            trends.regression_rate_direction,
            trends.regression_rate_slope,
            trends.regression_rate_percent_change,
            higher_is_better=False,
        ),
        _trend_row(
            "Improvement Rate",
            trends.improvement_rate_direction,
            trends.improvement_rate_slope,
            trends.improvement_rate_percent_change,
            higher_is_better=True,
        ),
        "",
        f"_Computed from {trends.entry_count} history entry(ies)"
        + (
            f" spanning {trends.time_span_seconds / 3600:.1f} hours._"
            if trends.time_span_seconds and trends.time_span_seconds > 0
            else "._"
        ),
    ]
    return "\n".join(lines)


def _trend_row(
    label: str,
    direction: Literal["improving", "worsening", "stable", "insufficient_data"],
    slope: float | None,
    pct: float | None,
    higher_is_better: bool,
) -> str:
    """Render one KPI trend row."""
    arrow = {
        "improving": "📈 improving" if higher_is_better else "📉 improving",
        "worsening": "📉 worsening" if higher_is_better else "📈 worsening",
        "stable": "➡️ stable",
        "insufficient_data": "⚠️ N/A",
    }.get(direction, "?")

    slope_str = f"{slope:+.4f}" if slope is not None else "N/A"
    pct_str = f"{pct:+.1f}%" if pct is not None else "N/A"

    return f"| {label} | {arrow} | {slope_str} | {pct_str} |"


def _failure_class_distribution(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Aggregate ``failure_class`` counts from persisted Critic verdicts (issue #705).

    Returns a ``failure_class -> count`` map across every ``critic_verdict``
    event matching *harness_version*. Verdicts without a ``failure_class``
    (e.g. from older stores or clean sessions that short-circuit before the
    Critic runs) are ignored so the map only includes sessions that ran the
    full pipeline.
    """
    distribution: dict[str, int] = {}
    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        record = VerdictRecord(**event.payload)
        if record.failure_class is not None:
            distribution[record.failure_class] = distribution.get(record.failure_class, 0) + 1
    return distribution


def _session_slice_or_task_slice(
    logger: TraceLogger,
    *,
    harness_version: str | None,
    group_by: GroupByDim | None,
    task_metadata: dict[str, TaskKpiMetadata] | None,
) -> dict[str, SkillKpiSlice]:
    """Dispatch to session-level or task-level slicing based on *group_by* (issue #1039).

    Session-level dimensions (``model_id``, ``quantization``,
    ``harness_version``) use :func:`_slice_session_verdict_rates` which
    groups verdicts by session attributes.  Task-level dimensions use the
    existing :func:`_slice_verdict_rates` which groups by task metadata.
    """
    if group_by is None:
        return {}
    if group_by in _SESSION_LEVEL_DIMS:
        return _slice_session_verdict_rates(
            logger,
            harness_version=harness_version,
            group_by=group_by,
        )
    return _slice_verdict_rates(
        logger,
        harness_version=harness_version,
        group_by=group_by,
        task_metadata=task_metadata,
    )


def compute_kpis(
    logger: TraceLogger,
    harness_version: str | None = None,
    *,
    group_by: GroupByDim | None = None,
    task_metadata: dict[str, TaskKpiMetadata] | None = None,
) -> KpiSummary:
    """Compute KPIs from the trace store backing *logger*.

    Parameters
    ----------
    logger:
        A :class:`~foundry_x.trace.logger.TraceLogger`.
    harness_version:
        When provided, only sessions created with this harness version are
        considered.
    group_by:
        Issue #898 — when set to ``"skill"``, ``"task_family"``, or
        ``"difficulty_tier"``, additionally breaks ``improvement_rate`` and
        ``regression_rate`` down per group and populates the matching field
        on the returned :class:`KpiSummary` (``per_skill`` /
        ``per_task_family`` / ``per_difficulty_tier``). Requires
        *task_metadata* to attribute verdict checks to groups; when
        *task_metadata* is ``None`` or empty the slice fields stay empty
        (graceful degradation) and the aggregate KPIs are unaffected.
    task_metadata:
        ``task name -> TaskKpiMetadata`` map. Build it with
        :func:`build_task_metadata` (auto-loaded from the benchmark
        registry) or supply your own (e.g. from a JSON file via the
        ``--task-metadata`` CLI flag).

    Issue #273 — the per-session helpers below each call
    :meth:`TraceLogger.query_events` exactly once per event kind. The
    previous shape issued ``list_sessions()`` and then ``iter_events(sid)``
    once per session per kind (S*K connect sites); the new shape is K
    streaming cursors total, with the ``harness_version`` filter pushed
    down to the store so a multi-session fixture does not need to be
    materialized in Python.
    """
    (
        cycle_time,
        cycle_time_p50,
        cycle_time_p95,
        excluded_wall_clock,
        excluded_token_budget,
        excluded_event_limit,
        excluded_other,
    ) = _cycle_time(logger, harness_version=harness_version)
    excluded_from_cycle_time = (
        excluded_wall_clock + excluded_token_budget + excluded_event_limit + excluded_other
    )
    regression_rate, improvement_rate = _verdict_rates(
        logger, harness_version=harness_version, task_metadata=task_metadata
    )
    injection_blocks = _injection_blocks(logger, harness_version=harness_version)
    token_totals = _token_totals(logger, harness_version=harness_version)
    hooks_disabled_count, hooks_disabled_rate = _hook_registry_errors(
        logger, harness_version=harness_version
    )
    token_budget_abort_count = _token_budget_aborts(logger, harness_version=harness_version)
    token_budget_hit_rate = _token_budget_hit_rate(logger, harness_version=harness_version)
    token_budget_overrun_pct = _token_budget_overrun(logger, harness_version=harness_version)
    streaming_quality = _streaming_quality(logger, harness_version=harness_version)
    context_efficiency = _context_efficiency(logger, harness_version=harness_version)
    context_pruned_count = _context_pruned(logger, harness_version=harness_version)
    wall_clock_abort_count = _wall_clock_abort_count(logger, harness_version=harness_version)
    failure_class_distribution = _failure_class_distribution(
        logger, harness_version=harness_version
    )
    model_retry_count = _model_retry_count(logger, harness_version=harness_version)
    tool_argument_parse_error_count = _tool_argument_parse_error_count(
        logger, harness_version=harness_version
    )
    event_limit_abort_count = _event_limit_abort_count(logger, harness_version=harness_version)
    server_restart_count = _server_restart_count(logger, harness_version=harness_version)
    evolver_llm_failure_count, evolver_llm_failure_rate = _evolver_llm_failure(
        logger, harness_version=harness_version
    )
    evolver_duration_ms = _evolver_duration_ms(logger, harness_version=harness_version)
    model_cost_count, total_model_cost_usd = _model_cost_count(
        logger, harness_version=harness_version
    )
    model_rate_limit_count = _model_rate_limit_count(logger, harness_version=harness_version)
    fetch_blocked_count = _fetch_blocked_count(logger, harness_version=harness_version)
    (
        streaming_quality_mean_ttft_ms,
        streaming_quality_p50_ttft_ms,
        streaming_quality_p95_ttft_ms,
    ) = _streaming_quality_aggregate(logger, harness_version=harness_version)
    mean_prompt_tokens_per_step, mean_completion_tokens_per_step = _token_per_step(
        logger, harness_version=harness_version
    )
    (
        hook_overhead,
        hook_overhead_ms_p50,
        hook_overhead_ms_p95,
        hook_post_overhead_ms_p50,
        hook_post_overhead_ms_p95,
    ) = _hook_overhead(logger, harness_version=harness_version)

    return KpiSummary(
        cycle_time_seconds=cycle_time,
        cycle_time_p50_seconds=cycle_time_p50,
        cycle_time_p95_seconds=cycle_time_p95,
        regression_rate=regression_rate,
        improvement_rate=improvement_rate,
        injection_blocks=injection_blocks,
        token_totals=token_totals,
        hooks_disabled_count=hooks_disabled_count,
        hooks_disabled_rate=hooks_disabled_rate,
        token_budget_abort_count=token_budget_abort_count,
        token_budget_hit_rate=token_budget_hit_rate,
        token_budget_overrun_pct=token_budget_overrun_pct,
        context_efficiency=context_efficiency,
        streaming_quality=streaming_quality,
        context_pruned_count=context_pruned_count,
        wall_clock_abort_count=wall_clock_abort_count,
        failure_class_distribution=failure_class_distribution,
        model_retry_count=model_retry_count,
        tool_argument_parse_error_count=tool_argument_parse_error_count,
        event_limit_abort_count=event_limit_abort_count,
        server_restart_count=server_restart_count,
        excluded_from_cycle_time=excluded_from_cycle_time,
        excluded_wall_clock=excluded_wall_clock,
        excluded_token_budget=excluded_token_budget,
        excluded_event_limit=excluded_event_limit,
        excluded_other=excluded_other,
        evolver_llm_failure_count=evolver_llm_failure_count,
        evolver_llm_failure_rate=evolver_llm_failure_rate,
        evolver_duration_ms=evolver_duration_ms,
        model_cost_count=model_cost_count,
        total_model_cost_usd=total_model_cost_usd,
        model_rate_limit_count=model_rate_limit_count,
        fetch_blocked_count=fetch_blocked_count,
        streaming_quality_mean_ttft_ms=streaming_quality_mean_ttft_ms,
        streaming_quality_p50_ttft_ms=streaming_quality_p50_ttft_ms,
        streaming_quality_p95_ttft_ms=streaming_quality_p95_ttft_ms,
        mean_prompt_tokens_per_step=mean_prompt_tokens_per_step,
        mean_completion_tokens_per_step=mean_completion_tokens_per_step,
        hook_overhead=hook_overhead,
        hook_overhead_ms_p50=hook_overhead_ms_p50,
        hook_overhead_ms_p95=hook_overhead_ms_p95,
        hook_post_overhead_ms_p50=hook_post_overhead_ms_p50,
        hook_post_overhead_ms_p95=hook_post_overhead_ms_p95,
        **_slice_field(
            _session_slice_or_task_slice(
                logger,
                harness_version=harness_version,
                group_by=group_by,
                task_metadata=task_metadata,
            ),
            group_by,
        ),
    )


def _slice_field(
    slices: dict[str, SkillKpiSlice],
    group_by: GroupByDim | None,
) -> dict[str, dict[str, SkillKpiSlice]]:
    """Map a computed slice dict onto the matching ``KpiSummary`` field.

    Returns a ``{field_name: slices}`` kwargs dict for the ``**`` spread
    in :func:`compute_kpis`. When *group_by* is ``None`` or the slices are
    empty (no task metadata available), returns ``{}`` so the summary is
    built with the field defaults and stays compact.
    """
    if group_by is None or not slices:
        return {}
    field_name = {
        "skill": "per_skill",
        "task_family": "per_task_family",
        "difficulty_tier": "per_difficulty_tier",
        "model_id": "per_model_id",
        "quantization": "per_quantization",
        "harness_version": "per_harness_version",
    }[group_by]
    return {field_name: slices}


def compare_kpis(
    logger: TraceLogger,
    baseline_version: str,
    candidate_version: str,
    *,
    group_by: GroupByDim | None = None,
    task_metadata: dict[str, TaskKpiMetadata] | None = None,
) -> KpiComparison:
    """Compute a baseline-vs-candidate comparison (issue #100).

    Each version is reduced to its own :class:`KpiSummary` via
    :func:`compute_kpis`, then the candidate-minus-baseline deltas are
    derived for the three PRD KPIs. The sign convention (which direction
    is "good") is applied at render time, not here, so the structured
    ``deltas`` stay sign-agnostic for JSON consumers.

    Issue #736: session counts are included so callers can distinguish
    "no change" (deltas near 0.0 with real sessions) from "no data"
    (deltas are 0.0 because one version has zero sessions).

    Issue #898: when *group_by* is set, both summaries are computed with
    the same ``group_by`` / ``task_metadata`` and per-slice deltas are
    attached as :attr:`KpiComparison.slice_deltas` (one entry per group;
    rate fields are candidate-minus-baseline).
    """
    baseline = compute_kpis(
        logger,
        harness_version=baseline_version,
        group_by=group_by,
        task_metadata=task_metadata,
    )
    candidate = compute_kpis(
        logger,
        harness_version=candidate_version,
        group_by=group_by,
        task_metadata=task_metadata,
    )
    baseline_session_count = len(logger.list_sessions(harness_version=baseline_version))
    candidate_session_count = len(logger.list_sessions(harness_version=candidate_version))
    return KpiComparison(
        baseline=baseline,
        candidate=candidate,
        deltas=_compute_deltas(baseline, candidate),
        baseline_session_count=baseline_session_count,
        candidate_session_count=candidate_session_count,
        slice_deltas=_compute_slice_deltas(baseline, candidate, group_by),
    )


def _slices_for(summary: KpiSummary, group_by: GroupByDim | None) -> dict[str, SkillKpiSlice]:
    """Return the slice dict on *summary* matching *group_by*."""
    if group_by == "skill":
        return summary.per_skill
    if group_by == "task_family":
        return summary.per_task_family
    if group_by == "difficulty_tier":
        return summary.per_difficulty_tier
    if group_by == "model_id":
        return summary.per_model_id
    if group_by == "quantization":
        return summary.per_quantization
    if group_by == "harness_version":
        return summary.per_harness_version
    return {}


def _compute_slice_deltas(
    baseline: KpiSummary,
    candidate: KpiSummary,
    group_by: GroupByDim | None,
) -> dict[str, dict[str, SkillKpiSlice]]:
    """Per-slice candidate-minus-baseline deltas keyed by dimension (issue #898).

    Returns ``{}`` when *group_by* is ``None``. The union of baseline and
    candidate group keys is walked so a group that exists on only one
    side still appears (the missing side contributes a 0.0 rate). Each
    :class:`SkillKpiSlice` carries the delta in its rate fields and the
    *candidate*'s verdict/session counts for reference, matching the
    "candidate is the subject of the delta" convention of
    :func:`_compute_deltas`.
    """
    if group_by is None:
        return {}
    base = _slices_for(baseline, group_by)
    cand = _slices_for(candidate, group_by)
    inner: dict[str, SkillKpiSlice] = {}
    for group in sorted(set(base) | set(cand)):
        b = base.get(group)
        c = cand.get(group)
        b_imp = b.improvement_rate if b else 0.0
        c_imp = c.improvement_rate if c else 0.0
        b_reg = b.regression_rate if b else 0.0
        c_reg = c.regression_rate if c else 0.0
        inner[group] = SkillKpiSlice(
            improvement_rate=c_imp - b_imp,
            regression_rate=c_reg - b_reg,
            verdict_count=c.verdict_count if c else 0,
            session_count=c.session_count if c else 0,
        )
    return {group_by: inner}


def _compute_deltas(
    baseline: KpiSummary,
    candidate: KpiSummary,
) -> dict[str, float | int | None]:
    def _delta(b: float | None, c: float | None) -> float | None:
        if b is None or c is None:
            return None
        return c - b

    return {
        "cycle_time_seconds": _delta(baseline.cycle_time_seconds, candidate.cycle_time_seconds),
        # Issue #1338: cycle-time percentile deltas (lower is better).
        "cycle_time_p50_seconds": _delta(
            baseline.cycle_time_p50_seconds, candidate.cycle_time_p50_seconds
        ),
        "cycle_time_p95_seconds": _delta(
            baseline.cycle_time_p95_seconds, candidate.cycle_time_p95_seconds
        ),
        "regression_rate": _delta(baseline.regression_rate, candidate.regression_rate),
        "improvement_rate": _delta(baseline.improvement_rate, candidate.improvement_rate),
        "token_budget_hit_rate": _delta(
            baseline.token_budget_hit_rate, candidate.token_budget_hit_rate
        ),
        # Issue #1112: token budget overrun percentage delta (lower is better —
        # fewer tokens over budget means more efficient sessions).
        "token_budget_overrun_pct": _delta(
            baseline.token_budget_overrun_pct, candidate.token_budget_overrun_pct
        ),
        "context_efficiency": _delta(baseline.context_efficiency, candidate.context_efficiency),
        "hooks_disabled_rate": _delta(baseline.hooks_disabled_rate, candidate.hooks_disabled_rate),
        "wall_clock_abort_count": candidate.wall_clock_abort_count
        - baseline.wall_clock_abort_count,
        "model_retry_count": candidate.model_retry_count - baseline.model_retry_count,
        "tool_argument_parse_error_count": (
            candidate.tool_argument_parse_error_count - baseline.tool_argument_parse_error_count
        ),
        "event_limit_abort_count": candidate.event_limit_abort_count
        - baseline.event_limit_abort_count,
        "server_restart_count": candidate.server_restart_count - baseline.server_restart_count,
        # Issue #895: exclusion-count delta (candidate - baseline). A rising
        # count means more sessions are failing before the Critic runs, which
        # widens the survivorship-bias blind spot in ``cycle_time_seconds``.
        "excluded_from_cycle_time": candidate.excluded_from_cycle_time
        - baseline.excluded_from_cycle_time,
        # Issue #1113: per-abort-reason exclusion breakdown deltas.
        "excluded_wall_clock": candidate.excluded_wall_clock - baseline.excluded_wall_clock,
        "excluded_token_budget": (candidate.excluded_token_budget - baseline.excluded_token_budget),
        "excluded_event_limit": (candidate.excluded_event_limit - baseline.excluded_event_limit),
        "excluded_other": candidate.excluded_other - baseline.excluded_other,
        # Issue #953: evolver LLM failure count and rate deltas.
        "evolver_llm_failure_count": (
            candidate.evolver_llm_failure_count - baseline.evolver_llm_failure_count
        ),
        "evolver_llm_failure_rate": _delta(
            baseline.evolver_llm_failure_rate, candidate.evolver_llm_failure_rate
        ),
        # Issue #1346: evolver duration delta (lower is better — faster evolver).
        "evolver_duration_ms": _delta(baseline.evolver_duration_ms, candidate.evolver_duration_ms),
        # Issue #1281: model cost, rate limit, and fetch blocked deltas.
        "model_cost_count": candidate.model_cost_count - baseline.model_cost_count,
        "total_model_cost_usd": _delta(
            baseline.total_model_cost_usd, candidate.total_model_cost_usd
        ),
        "model_rate_limit_count": (
            candidate.model_rate_limit_count - baseline.model_rate_limit_count
        ),
        "fetch_blocked_count": candidate.fetch_blocked_count - baseline.fetch_blocked_count,
        # Issue #1271: aggregate streaming quality TTFT deltas (lower is better —
        # faster first token is improvement; slower is regression).
        "streaming_quality_mean_ttft_ms": _delta(
            baseline.streaming_quality_mean_ttft_ms, candidate.streaming_quality_mean_ttft_ms
        ),
        "streaming_quality_p50_ttft_ms": _delta(
            baseline.streaming_quality_p50_ttft_ms, candidate.streaming_quality_p50_ttft_ms
        ),
        "streaming_quality_p95_ttft_ms": _delta(
            baseline.streaming_quality_p95_ttft_ms, candidate.streaming_quality_p95_ttft_ms
        ),
        # Issue #1271: token per-step deltas (higher is better — more tokens
        # per step means more efficient model usage).
        "mean_prompt_tokens_per_step": _delta(
            baseline.mean_prompt_tokens_per_step, candidate.mean_prompt_tokens_per_step
        ),
        "mean_completion_tokens_per_step": _delta(
            baseline.mean_completion_tokens_per_step, candidate.mean_completion_tokens_per_step
        ),
        # Issue #1269: hook overhead deltas (higher hook overhead is worse).
        "hook_overhead_ms_p50": _delta(
            baseline.hook_overhead_ms_p50, candidate.hook_overhead_ms_p50
        ),
        "hook_overhead_ms_p95": _delta(
            baseline.hook_overhead_ms_p95, candidate.hook_overhead_ms_p95
        ),
        "hook_post_overhead_ms_p50": _delta(
            baseline.hook_post_overhead_ms_p50, candidate.hook_post_overhead_ms_p50
        ),
        "hook_post_overhead_ms_p95": _delta(
            baseline.hook_post_overhead_ms_p95, candidate.hook_post_overhead_ms_p95
        ),
    }


def _cycle_time(
    logger: TraceLogger,
    harness_version: str | None = None,
    since: str | None = None,
) -> tuple[float | None, float | None, float | None, int, int, int, int]:
    """Mean, p50, and p95 wall-clock time from ``task_received`` to ``critic_verdict`` plus exclusion breakdown.

    Returns
    -------
    ``(mean_seconds, p50_seconds, p95_seconds, excluded_wall_clock, excluded_token_budget, excluded_event_limit, excluded_other)``.

    The mean is over sessions that have both a ``task_received`` and a
    ``critic_verdict`` event with a strictly positive delta; it is
    ``None`` when no session qualified. ``p50_seconds`` and ``p95_seconds``
    are the 50th and 95th percentiles of the same deltas (issue #1338).

    Issue #273 — previously looped every session id and called
    ``iter_events`` twice per session to find the first event of each
    kind. Now two :meth:`TraceLogger.query_events` cursors stream every
    qualifying event in timestamp order; ``setdefault`` keeps the first
    (earliest) event per session, which is exactly the prior
    first-event-of-kind semantics.

    Issue #895 — ``excluded_count`` is the number of sessions that have a
    ``task_received`` event but did **not** contribute a positive delta to
    the mean: sessions without a ``critic_verdict`` (the survivorship-bias
    case called out in the issue — model errors, early wall-clock /
    event-limit / token-budget aborts), plus the rare session whose
    timestamps could not be parsed or whose delta was non-positive.
    Surfacing the count alongside the mean lets an operator tell a mean
    computed over every session from one computed over a small, self-
    selected subpopulation of survivors.

    Issue #1113 — the exclusion count is broken down by the ``reason``
    field of the ``task_aborted`` event for each excluded session:
    ``wall_clock``, ``token_budget``, ``event_limit``, or ``other``
    (e.g. no abort event or a reason not in the three tracked categories).

    Issue #1270 — the ``since`` filter is pushed down to the store so
    time-bounded queries do not materialize events outside the window.

    Issue #1338 — per-session deltas are collected into a list and sorted
    to derive p50 and p95 via the ``percentile()`` function.
    """
    start_events: dict[str, TraceEvent] = {}
    for event in logger.query_events(
        kind="task_received", harness_version=harness_version, since=since
    ):
        start_events.setdefault(event.session_id, event)
    end_events: dict[str, TraceEvent] = {}
    for event in logger.query_events(
        kind="critic_verdict", harness_version=harness_version, since=since
    ):
        end_events.setdefault(event.session_id, event)

    # Issue #1113: build a session_id -> reason map from task_aborted events
    # so we can attribute excluded sessions to their abort reason.
    abort_reasons: dict[str, str] = {}
    for event in logger.query_events(
        kind="task_aborted", harness_version=harness_version, since=since
    ):
        if event.session_id not in abort_reasons:
            abort_reasons[event.session_id] = event.payload.get("reason", "other")

    deltas: list[float] = []
    excluded_wall_clock = 0
    excluded_token_budget = 0
    excluded_event_limit = 0
    excluded_other = 0

    for sid, start_event in start_events.items():
        end_event = end_events.get(sid)
        if end_event is None:
            # Issue #895: a session with ``task_received`` but no
            # ``critic_verdict`` failed before the Critic ran and is
            # excluded from the mean — count it so the survivorship bias
            # is visible rather than silent.
            # Issue #1113: break down by abort reason.
            reason = abort_reasons.get(sid, "other")
            if reason == "wall_clock":
                excluded_wall_clock += 1
            elif reason == "token_budget":
                excluded_token_budget += 1
            elif reason == "event_limit":
                excluded_event_limit += 1
            else:
                excluded_other += 1
            continue
        try:
            t0 = datetime.fromisoformat(start_event.timestamp)
            t1 = datetime.fromisoformat(end_event.timestamp)
        except ValueError:
            # Issue #1113: timestamp parse failure — categorize as "other".
            excluded_other += 1
            continue
        delta = (t1 - t0).total_seconds()
        if delta > 0:
            deltas.append(delta)
        else:
            excluded_other += 1

    if not deltas:
        return (
            None,
            None,
            None,
            excluded_wall_clock,
            excluded_token_budget,
            excluded_event_limit,
            excluded_other,
        )
    sorted_deltas = sorted(deltas)
    return (
        sum(deltas) / len(deltas),
        percentile(sorted_deltas, 50.0),
        percentile(sorted_deltas, 95.0),
        excluded_wall_clock,
        excluded_token_budget,
        excluded_event_limit,
        excluded_other,
    )


def _verdict_rates(
    logger: TraceLogger,
    harness_version: str | None = None,
    task_metadata: dict[str, TaskKpiMetadata] | None = None,
) -> tuple[float, float]:
    """Derive regression and improvement rates from persisted Critic verdicts.

    Verdicts are persisted as the :class:`VerdictRecord` shape
    (``approved`` / ``passed_checks`` / ``failed_checks`` / ``notes``), not the
    synthetic ``{"verdict", "regression"}`` payload the earlier implementation
    assumed (issue #98).

    Issue #273 — a single :meth:`TraceLogger.query_events` cursor walks
    every ``critic_verdict`` row across all matching sessions in
    timestamp order, so the ``prior_passed`` tracker sees verdicts in
    the same order the previous per-session nested loop produced.

    * *improvement_rate* = approved non-smoke verdicts / total non-smoke verdicts.
    * *regression_rate* = sessions with >=1 regressed task / sessions with a
      verdict, where a task regresses when it appears in ``failed_checks`` after
      having appeared in ``passed_checks`` in an earlier verdict.

    ADR-0034 §2 — smoke-tier exclusion
    ----------------------------------
    Smoke-tier tasks do not exercise agent capability, so a harness that
    passes all smoke tasks but fails all easy/medium/hard tasks has NOT
    improved. When *task_metadata* is supplied, verdicts whose every task
    belongs to the ``smoke`` difficulty tier are EXCLUDED from the
    improvement-rate denominator (but still counted in regression-rate, since
    a smoke-task failure indicates a broken pipeline, not an agent regression).
    Verdicts that mix smoke and non-smoke tasks are attributed to the
    non-smoke portion and count normally.
    """

    total_verdicts = 0
    approved = 0
    prior_passed: dict[str, str] = {}
    sessions_with_verdicts: set[str] = set()
    regression_sessions: set[str] = set()

    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        sessions_with_verdicts.add(event.session_id)
        record = VerdictRecord(**event.payload)

        has_non_smoke = False
        has_smoke_failed = False
        if task_metadata is not None:
            all_tasks = set(record.passed_checks) | set(record.failed_checks)
            has_non_smoke = any(
                task_metadata.get(t) is not None and task_metadata[t].difficulty_tier != "smoke"
                for t in all_tasks
            )
            has_smoke_failed = any(
                task_metadata.get(t) is not None
                and task_metadata[t].difficulty_tier == "smoke"
                and t in set(record.failed_checks)
                for t in all_tasks
            )

        if task_metadata is None or (has_non_smoke and not has_smoke_failed):
            total_verdicts += 1
            if record.verdict:
                approved += 1

        for task in record.failed_checks:
            if task in prior_passed:
                regression_sessions.add(event.session_id)
        for task in record.passed_checks:
            prior_passed[task] = event.session_id

    improvement_rate = approved / total_verdicts if total_verdicts else 0.0
    regression_rate = (
        len(regression_sessions) / len(sessions_with_verdicts) if sessions_with_verdicts else 0.0
    )
    return regression_rate, improvement_rate


def _groups_for_task(meta: TaskKpiMetadata, group_by: GroupByDim) -> set[str]:
    """Return the set of group keys a task contributes to for *group_by*.

    * ``skill``        → ``meta.skills`` (a task may require several skills).
    * ``task_family``  → ``meta.task_families`` (a task may carry several
      ``BenchmarkTask.tags``).
    * ``difficulty_tier`` → ``{meta.difficulty_tier}`` (exactly one tier, or
      empty when the task declares none).
    """
    if group_by == "skill":
        return set(meta.skills)
    if group_by == "task_family":
        return set(meta.task_families)
    if group_by == "difficulty_tier":
        return {meta.difficulty_tier} if meta.difficulty_tier is not None else set()
    return set()


class _SliceAcc:
    """Mutable accumulator for one slice key (issue #898).

    Mirrors the local variables the aggregate :func:`_verdict_rates`
    keeps: a verdict count, an approved count, the set of sessions with a
    verdict, the set of sessions with a regression, and the set of tasks
    that previously passed (scoped to this group so a task is only a
    regression for the groups it actually belongs to).
    """

    __slots__ = ("approved", "prior_passed", "regression_sessions", "sessions", "total")

    def __init__(self) -> None:
        self.total = 0
        self.approved = 0
        self.sessions: set[str] = set()
        self.regression_sessions: set[str] = set()
        self.prior_passed: set[str] = set()


def _slice_verdict_rates(
    logger: TraceLogger,
    harness_version: str | None,
    group_by: GroupByDim | None,
    task_metadata: dict[str, TaskKpiMetadata] | None,
) -> dict[str, SkillKpiSlice]:
    """Per-group ``improvement_rate`` / ``regression_rate`` (issue #898).

    Walks the same single ``critic_verdict`` cursor as the aggregate
    :func:`_verdict_rates` (one streaming scan, ``harness_version`` pushed
    down — issue #273) but buckets each verdict into every group its
    checks touch. Returns an empty dict when *group_by* is ``None`` or
    *task_metadata* is empty, so the caller's slice fields stay at their
    defaults and the aggregate KPIs are unaffected.

    A verdict is attributed to a group when any of its checks names a
    task whose :class:`TaskKpiMetadata` lists that group. Because a task
    may declare several skills (or tags), a single verdict can land in
    several groups' buckets — the slices are independent views, not a
    partition. ``prior_passed`` is scoped per group so a regressing task
    only counts against the groups it belongs to.
    """
    if group_by is None or not task_metadata:
        return {}

    acc: dict[str, _SliceAcc] = {}
    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        record = VerdictRecord(**event.payload)

        all_tasks = set(record.passed_checks) | set(record.failed_checks)
        has_non_smoke = any(
            task_metadata.get(t) is not None and task_metadata[t].difficulty_tier != "smoke"
            for t in all_tasks
        )

        # Resolve every group this verdict touches up front so the per-
        # group loop below does not re-walk the metadata per check.
        touched: set[str] = set()
        for task in (*record.passed_checks, *record.failed_checks):
            meta = task_metadata.get(task)
            if meta is not None:
                touched |= _groups_for_task(meta, group_by)
        # No explicit ``if not touched: continue`` here: an empty ``touched``
        # (a verdict naming only unknown tasks, or tasks with no values for the
        # chosen dimension) is a natural no-op for the ``for group in touched``
        # loop below, so the guard was dead code with no behavioural effect.
        for group in touched:
            bucket = acc.setdefault(group, _SliceAcc())
            bucket.sessions.add(event.session_id)
            if has_non_smoke:
                bucket.total += 1
                if group == "smoke":
                    if record.verdict:
                        bucket.approved += 1
                else:
                    tier_passed = any(
                        task_metadata.get(t) is not None
                        and group in _groups_for_task(task_metadata[t], group_by)
                        and t in set(record.passed_checks)
                        for t in all_tasks
                    )
                    if tier_passed:
                        bucket.approved += 1
            elif group == "smoke":
                bucket.total += 1
                tier_passed = any(
                    task_metadata.get(t) is not None
                    and task_metadata[t].difficulty_tier == "smoke"
                    and t in set(record.passed_checks)
                    for t in all_tasks
                )
                if tier_passed:
                    bucket.approved += 1
            for task in record.failed_checks:
                if task in bucket.prior_passed:
                    bucket.regression_sessions.add(event.session_id)
            for task in record.passed_checks:
                meta = task_metadata.get(task)
                # Only seed prior_passed for the groups this task belongs
                # to, so a multi-skill task is not a false regression for
                # an unrelated skill that happened to share the verdict.
                if meta is not None and group in _groups_for_task(meta, group_by):
                    bucket.prior_passed.add(task)

    slices: dict[str, SkillKpiSlice] = {}
    for group, bucket in acc.items():
        slices[group] = SkillKpiSlice(
            improvement_rate=bucket.approved / bucket.total if bucket.total else 0.0,
            regression_rate=(
                len(bucket.regression_sessions) / len(bucket.sessions) if bucket.sessions else 0.0
            ),
            verdict_count=bucket.total,
            session_count=len(bucket.sessions),
        )
    return slices


# Issue #1039 — session-level dimensions --------------------------------

#: Mapping from session-level :class:`GroupByDim` values to the
#: :class:`~foundry_x.trace.logger.TraceSession` attribute name.
_SESSION_DIM_ATTRS: dict[str, str] = {
    "model_id": "model_id",
    "quantization": "quantization",
    "harness_version": "harness_version",
}


def _build_session_group_map(
    logger: TraceLogger,
    *,
    group_by: GroupByDim,
    harness_version: str | None,
) -> dict[str, str]:
    """Map ``session_id -> group key`` for a session-level dimension.

    When the session attribute is ``None`` the session is omitted from the
    map so it does not appear in the slice breakdown (graceful degradation
    per the issue's acceptance criteria).  ``harness_version`` is always
    applied to ``list_sessions`` so sessions outside the filter window
    are excluded.
    """
    attr = _SESSION_DIM_ATTRS[group_by]
    mapping: dict[str, str] = {}
    for session in logger.list_sessions(harness_version=harness_version):
        key = getattr(session, attr, None)
        if key is not None:
            mapping[session.session_id] = str(key)
    return mapping


def _slice_session_verdict_rates(
    logger: TraceLogger,
    *,
    harness_version: str | None,
    group_by: GroupByDim,
) -> dict[str, SkillKpiSlice]:
    """Session-level per-group ``improvement_rate`` / ``regression_rate`` (issue #1039).

    Works like :func:`_slice_verdict_rates` but buckets verdicts by a
    session attribute (``model_id``, ``quantization``, or
    ``harness_version``) instead of task metadata.  A session whose
    attribute is ``None`` is excluded — the slice shows only sessions
    that declared a value for the chosen dimension.

    The ``prior_passed`` set is shared across groups (not scoped per
    group) because a regression in session-level slicing means "the same
    task regressed in the same session group" — the session group
    determines the bucket, not the task's metadata.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) and
    one :meth:`TraceLogger.list_sessions` call.
    """
    if group_by not in _SESSION_DIM_ATTRS:
        return {}

    session_groups = _build_session_group_map(
        logger, group_by=group_by, harness_version=harness_version
    )
    if not session_groups:
        return {}

    acc: dict[str, _SliceAcc] = {}
    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        group = session_groups.get(event.session_id)
        if group is None:
            continue
        record = VerdictRecord(**event.payload)
        bucket = acc.setdefault(group, _SliceAcc())
        bucket.total += 1
        bucket.sessions.add(event.session_id)
        if record.verdict:
            bucket.approved += 1
        for task in record.failed_checks:
            if task in bucket.prior_passed:
                bucket.regression_sessions.add(event.session_id)
        for task in record.passed_checks:
            bucket.prior_passed.add(task)

    slices: dict[str, SkillKpiSlice] = {}
    for group, bucket in acc.items():
        slices[group] = SkillKpiSlice(
            improvement_rate=bucket.approved / bucket.total if bucket.total else 0.0,
            regression_rate=(
                len(bucket.regression_sessions) / len(bucket.sessions) if bucket.sessions else 0.0
            ),
            verdict_count=bucket.total,
            session_count=len(bucket.sessions),
        )
    return slices


# Issue #1039 — helpers for the session-level slice field wiring -------

# Task-level dimensions (sliced by TaskKpiMetadata).
_TASK_LEVEL_DIMS: frozenset[str] = frozenset({"skill", "task_family", "difficulty_tier"})
# Session-level dimensions (sliced by TraceSession attributes).
_SESSION_LEVEL_DIMS: frozenset[str] = frozenset({"model_id", "quantization", "harness_version"})


def build_task_metadata() -> dict[str, TaskKpiMetadata]:
    """Build the ``task name -> TaskKpiMetadata`` map (issue #898).

    Lazy-imports :func:`benchmarks.registry.load_all_tasks` so this KPI
    module never imports ``benchmarks`` at load time — the dependency
    direction stays ``benchmarks`` → ``foundry_x`` (mirroring
    ``src/foundry_x/evolution/critic.py``'s lazy registry wiring). Maps
    each ``BenchmarkTask``'s ``requires_skills`` → ``skills``,
    ``tags`` → ``task_families``, and ``difficulty_tier`` through.

    Returns an empty map when the registry cannot be imported (e.g.
    running the CLI outside the repo) so callers degrade gracefully:
    :func:`compute_kpis` with no metadata leaves the slice fields empty.
    """
    try:
        from benchmarks.registry import load_all_tasks
    except ImportError:
        return {}
    metadata: dict[str, TaskKpiMetadata] = {}
    for task in load_all_tasks():
        metadata[task.name] = TaskKpiMetadata(
            name=task.name,
            skills=list(task.requires_skills),
            task_families=list(task.tags),
            difficulty_tier=task.difficulty_tier,
        )
    return metadata


def _injection_blocks(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Per-session count of ``injection_blocked`` events (issue #120).

    Returns a ``session_id -> count`` map including only sessions with at
    least one block. Sessions without blocks are omitted so the rendering
    path can decide whether to add an extra section based on the map being
    non-empty (per the issue's "show … when at least one is present").

    Issue #273 — one :meth:`TraceLogger.query_events` cursor replaces
    the previous per-session ``iter_events`` loop; the kind filter is
    pushed down so only ``injection_blocked`` rows cross the boundary.
    """
    blocks: dict[str, int] = {}
    for event in logger.query_events(
        kind=INJECTION_BLOCKED_KIND,
        harness_version=harness_version,
    ):
        blocks[event.session_id] = blocks.get(event.session_id, 0) + 1
    return blocks


def _token_totals(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Per-session cumulative token totals (issue #271).

    Sums ``usage.total_tokens`` across every ``model_response`` event the
    runner records (issue #191). The runner itself keeps a running
    ``tokens_used`` counter (issue #197); summing the per-response
    ``total_tokens`` reproduces that cumulative figure without depending on
    the ``tokens_used`` key being present, so events written before that
    field landed still contribute.

    A ``model_response`` whose ``usage`` is missing or ``None`` (an
    OpenAI-compatible endpoint that omits accounting) contributes zero and
    does **not** seed the session into the map — only sessions with at
    least one event carrying a ``usage`` dict appear, mirroring the
    ``_injection_blocks`` "show only when present" contract.

    Like the other per-session helpers this uses one
    :meth:`TraceLogger.query_events` cursor (issue #273) with the kind and
    ``harness_version`` filters pushed down, so a multi-session store is a
    single ordered scan rather than S round-trips.
    """
    totals: dict[str, int] = {}
    for event in logger.query_events(
        kind="model_response",
        harness_version=harness_version,
    ):
        usage = event.payload.get("token_usage")
        if not isinstance(usage, dict):
            continue
        step_total = usage.get("total_tokens", 0)
        # ``bool`` is a subclass of ``int``; guard against truthy flags.
        if isinstance(step_total, bool) or not isinstance(step_total, int):
            continue
        totals[event.session_id] = totals.get(event.session_id, 0) + step_total
    return totals


def _hook_registry_errors(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[int, float]:
    """Total count and session-fraction of ``hook_registry_error`` events (issue #585).

    Returns ``(total_count, disabled_rate)`` where ``disabled_rate`` is the
    fraction of sessions with a ``task_received`` event that also had at
    least one ``hook_registry_error``. A registry error means every hook —
    including the security-critical ``InjectionFirewallHook`` — is silently
    disabled for the entire session, so any presence is noteworthy.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    sessions_with_errors: set[str] = set()
    total_count = 0
    for event in logger.query_events(
        kind="hook_registry_error",
        harness_version=harness_version,
    ):
        total_count += 1
        sessions_with_errors.add(event.session_id)

    if not sessions_with_errors:
        return 0, 0.0

    sessions_with_task: set[str] = set()
    for event in logger.query_events(kind="task_received", harness_version=harness_version):
        sessions_with_task.add(event.session_id)

    rate = len(sessions_with_errors) / len(sessions_with_task) if sessions_with_task else 0.0
    return total_count, rate


def _token_budget_aborts(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count sessions that hit ``task_aborted(reason="token_budget")`` (issue #466).

    Unlike ``_injection_blocks`` which returns a per-session map, this
    function returns a single integer: the number of sessions that
    recorded at least one ``task_aborted`` event with
    ``reason="token_budget"``. Sessions are counted once regardless of
    how many times the abort fires within them.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with
    the kind and ``harness_version`` filters pushed down.
    """
    sessions_with_abort: set[str] = set()
    for event in logger.query_events(
        kind=TASK_ABORTED_KIND,
        harness_version=harness_version,
    ):
        if event.payload.get("reason") == TOKEN_BUDGET_REASON:
            sessions_with_abort.add(event.session_id)
    # Issue #1355: also count the dedicated token_budget_aborted event
    for event in logger.query_events(
        kind=TOKEN_BUDGET_ABORTED_KIND,
        harness_version=harness_version,
    ):
        sessions_with_abort.add(event.session_id)
    return len(sessions_with_abort)


def _token_budget_hit_rate(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> float:
    """Fraction of sessions with at least one ``task_aborted(reason="token_budget")`` or ``token_budget_aborted`` event.

    Issue #551 — the token budget hit rate is a fourth tracked metric
    exposed via ``foundry-kpis`` alongside the three PRD KPIs. It signals
    whether the harness is driving tasks that repeatedly hit the token
    budget, which would indicate the context-pruning hook is not aggressive
    enough, or that the model-context window is being misspent.

    A session contributes to the numerator if it has at least one
    ``task_aborted`` event whose ``payload["reason"] == "token_budget"``,
    or at least one ``token_budget_aborted`` event (issue #1355).
    The denominator is the total number of sessions that have a
    ``task_received`` event (matching the harness version filter), which
    is the natural population boundary for the KPI.
    """
    sessions_with_abort: set[str] = set()
    all_sessions: set[str] = set()

    for event in logger.query_events(kind="task_received", harness_version=harness_version):
        all_sessions.add(event.session_id)

    for event in logger.query_events(kind="task_aborted", harness_version=harness_version):
        if event.payload.get("reason") == "token_budget":
            sessions_with_abort.add(event.session_id)

    # Issue #1355: also count the dedicated token_budget_aborted event
    for event in logger.query_events(
        kind=TOKEN_BUDGET_ABORTED_KIND, harness_version=harness_version
    ):
        sessions_with_abort.add(event.session_id)

    if not all_sessions:
        return 0.0
    return len(sessions_with_abort) / len(all_sessions)


def _token_budget_overrun(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> float | None:
    """Mean token budget overrun percentage across sessions that hit ``task_aborted(reason="token_budget")`` or ``token_budget_aborted`` (issues #1112, #1355).

    For each session that recorded at least one ``task_aborted`` event with
    ``reason="token_budget"``, or at least one ``token_budget_aborted`` event,
    extracts ``tokens_used`` and ``token_budget`` from the payload and computes
    the percentage overrun: ``(tokens_used - token_budget) / token_budget * 100``.

    Sessions are first-attempt-only (only the first abort event per session is
    considered) to avoid skewing the mean with repeated aborts in the same
    session. Returns ``None`` when no session hit the token budget, so operators
    can distinguish a clean store (None) from one where all sessions exceeded
    their budgets (a real percentage).

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    session_overruns: dict[str, float] = {}
    for event in logger.query_events(
        kind=TASK_ABORTED_KIND,
        harness_version=harness_version,
    ):
        if event.payload.get("reason") == TOKEN_BUDGET_REASON:
            sid = event.session_id
            if sid in session_overruns:
                continue
            tokens_used = event.payload.get("tokens_used")
            token_budget = event.payload.get("token_budget")
            if isinstance(tokens_used, int) and isinstance(token_budget, int) and token_budget > 0:
                overrun_pct = (tokens_used - token_budget) / token_budget * 100.0
                session_overruns[sid] = overrun_pct

    # Issue #1355: also process the dedicated token_budget_aborted event
    for event in logger.query_events(
        kind=TOKEN_BUDGET_ABORTED_KIND,
        harness_version=harness_version,
    ):
        sid = event.session_id
        if sid in session_overruns:
            continue
        tokens_used = event.payload.get("tokens_used")
        token_budget = event.payload.get("token_budget")
        if isinstance(tokens_used, int) and isinstance(token_budget, int) and token_budget > 0:
            overrun_pct = (tokens_used - token_budget) / token_budget * 100.0
            session_overruns[sid] = overrun_pct

    if not session_overruns:
        return None
    return sum(session_overruns.values()) / len(session_overruns)


def _streaming_quality(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, StreamingQualityData]:
    """Per-session streaming quality metrics (issue #580).

    Aggregates ``time_to_first_token_ms``, ``chunk_count``, and ``total_stream_ms``
    from every ``model_response`` event the runner records. Computes per-session
    average TTFT and average chunk interval (total_stream_ms / chunk_count).

    Sessions with no ``model_response`` events, or whose events lack timing
    data, are omitted from the returned map (mirrors the "show only when present"
    contract of ``_injection_blocks`` and ``_token_totals``).
    """
    session_ttfts: dict[str, list[int]] = {}
    session_chunks: dict[str, list[int]] = {}
    session_stream_ms: dict[str, list[int]] = {}

    for event in logger.query_events(
        kind="model_response",
        harness_version=harness_version,
    ):
        sid = event.session_id
        ttft = event.payload.get("time_to_first_token_ms")
        chunk_count = event.payload.get("chunk_count")
        total_stream_ms = event.payload.get("total_stream_ms")

        if isinstance(ttft, int):
            session_ttfts.setdefault(sid, []).append(ttft)
        if isinstance(chunk_count, int):
            session_chunks.setdefault(sid, []).append(chunk_count)
        if isinstance(total_stream_ms, int):
            session_stream_ms.setdefault(sid, []).append(total_stream_ms)

    result: dict[str, StreamingQualityData] = {}
    for sid in session_ttfts:
        ttfts = session_ttfts.get(sid, [])
        chunks = session_chunks.get(sid, [])
        stream_ms = session_stream_ms.get(sid, [])

        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else None
        total_chunks = sum(chunks) if chunks else 0
        total_ms = sum(stream_ms) if stream_ms else 0
        avg_interval = total_ms / total_chunks if total_chunks > 0 else None

        result[sid] = StreamingQualityData(
            avg_ttft_ms=avg_ttft,
            total_chunks=total_chunks,
            avg_chunk_interval_ms=avg_interval,
        )
    return result


def _streaming_quality_aggregate(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[float | None, float | None, float | None]:
    """Aggregate TTFT statistics across all sessions (issue #1271).

    Returns ``(mean_ttft_ms, p50_ttft_ms, p95_ttft_ms)`` where each value is
    computed over the per-session average TTFT values.  Sessions with no
    ``model_response`` events carrying ``time_to_first_token_ms`` are omitted.
    Returns ``(None, None, None)`` when no session has TTFT data.

    Uses the ``percentile`` function from :mod:`foundry_x.observability`
    (nearest-rank method, deterministic, matches operator intuition).
    """
    all_ttfts: list[int] = []
    for event in logger.query_events(
        kind="model_response",
        harness_version=harness_version,
    ):
        ttft = event.payload.get("time_to_first_token_ms")
        if isinstance(ttft, int):
            all_ttfts.append(ttft)

    if not all_ttfts:
        return None, None, None

    mean_ttft = sum(all_ttfts) / len(all_ttfts)
    sorted_ttfts = sorted(all_ttfts)
    p50_ttft = percentile(sorted_ttfts, 50.0)
    p95_ttft = percentile(sorted_ttfts, 95.0)
    return mean_ttft, p50_ttft, p95_ttft


def _token_per_step(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[float | None, float | None]:
    """Mean prompt and completion tokens per ``model_response`` step (issue #1271).

    Walks every ``model_response`` event and extracts ``token_usage.prompt_tokens``
    and ``token_usage.completion_tokens``.  Returns the mean across all steps.
    Returns ``(None, None)`` when no event carries token usage data.
    """
    prompt_tokens: list[int] = []
    completion_tokens: list[int] = []
    for event in logger.query_events(
        kind="model_response",
        harness_version=harness_version,
    ):
        usage = event.payload.get("token_usage")
        if not isinstance(usage, dict):
            continue
        pt = usage.get("prompt_tokens")
        ct = usage.get("completion_tokens")
        if isinstance(pt, int):
            prompt_tokens.append(pt)
        if isinstance(ct, int):
            completion_tokens.append(ct)

    mean_prompt = sum(prompt_tokens) / len(prompt_tokens) if prompt_tokens else None
    mean_completion = sum(completion_tokens) / len(completion_tokens) if completion_tokens else None
    return mean_prompt, mean_completion


def _context_pruned(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Per-session count of ``context_pruned`` events (issue #626).

    Returns a ``session_id -> count`` map including only sessions with at
    least one prune. Sessions without pruning are omitted so the rendering
    path can decide whether to add an extra section based on the map being
    non-empty (mirroring the ``_injection_blocks`` "show only when present"
    contract).

    Uses one :meth:`TraceLogger.query_events` cursor with the kind and
    ``harness_version`` filters pushed down.
    """
    counts: dict[str, int] = {}
    for event in logger.query_events(
        kind=CONTEXT_PRUNED_KIND,
        harness_version=harness_version,
    ):
        counts[event.session_id] = counts.get(event.session_id, 0) + 1
    return counts


def _context_efficiency(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> float | None:
    """Mean per-session context efficiency (issue #951, issue #979).

    Per-session efficiency = 1 - (sum(dropped) / sum(threshold + dropped))
    per the ADR-0021 §6 formula. A session that never pruned has
    ``dropped=0`` and therefore contributes ``1.0`` (perfect efficiency —
    nothing was dropped). Returns ``None`` only when the trace store has
    no sessions for the (optional) ``harness_version``.

    Issue #979 — previously sessions with zero ``context_pruned`` events
    were excluded from the mean, creating survivorship bias (only sessions
    that actually pruned were counted). They are now included as ``1.0``
    so the mean reflects the full session population.

    Uses one :meth:`TraceLogger.list_sessions` call and one
    :meth:`TraceLogger.query_events` cursor, both with the
    ``harness_version`` filter pushed down.
    """
    all_sessions = {s.session_id for s in logger.list_sessions(harness_version=harness_version)}

    session_dropped: dict[str, int] = {}
    session_threshold: dict[str, int] = {}
    for event in logger.query_events(
        kind=CONTEXT_PRUNED_KIND,
        harness_version=harness_version,
    ):
        sid = event.session_id
        dropped = event.payload.get("dropped", 0) if event.payload else 0
        threshold = event.payload.get("threshold", 0) if event.payload else 0
        if threshold == 0 and event.payload:
            threshold = event.payload.get("threshold_tokens", 0)
        session_dropped[sid] = session_dropped.get(sid, 0) + dropped
        session_threshold[sid] = session_threshold.get(sid, 0) + threshold

    if not all_sessions:
        return None

    efficiencies: list[float] = []
    for sid in all_sessions:
        dropped = session_dropped.get(sid, 0)
        threshold = session_threshold.get(sid, 0)
        denominator = threshold + dropped
        if denominator > 0:
            efficiency = 1.0 - (dropped / denominator)
        else:
            # Zero pruning (dropped=0, threshold=0) → perfect efficiency.
            efficiency = 1.0
        efficiencies.append(efficiency)

    return sum(efficiencies) / len(efficiencies)


def _wall_clock_abort_count(
    logger: TraceLogger,
    harness_version: str | None = None,
    since: str | None = None,
) -> int:
    """Count sessions aborted by the FOUNDRY_TASK_TIMEOUT wall-clock cap (issue #711, #1005).

    Counts sessions that recorded at least one ``task_aborted`` event with
    ``reason="wall_clock"``, fired by :func:`foundry_x.execution.runner.run_with_limits`
    when ``asyncio.wait_for`` raises :class:`asyncio.TimeoutError`. Sessions are
    counted once regardless of how many times the abort fires within them.

    Issue #1270 — the ``since`` filter is pushed down to the store so
    time-bounded queries do not materialize events outside the window.
    """
    sessions_with_abort: set[str] = set()
    for event in logger.query_events(
        kind="task_aborted",
        harness_version=harness_version,
        since=since,
    ):
        if event.payload.get("reason") == "wall_clock":
            sessions_with_abort.add(event.session_id)
    return len(sessions_with_abort)


def _event_limit_abort_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count sessions aborted by the FOUNDRY_MAX_EVENTS_PER_SESSION event cap (issue #869, #1005).

    Counts sessions that recorded at least one ``task_aborted`` event with
    ``reason="event_limit"``, fired by :func:`foundry_x.execution.runner.run_task`
    when the accumulated event count reaches the per-session cap (see
    ``foundry_x.execution.runner._check_event_limit``). Sessions are counted
    once regardless of how many times the abort fires within them, matching
    the pattern used by :func:`_token_budget_aborts`.

    A non-zero count signals a session that exceeded the configured event
    budget — typically a runaway loop where the agent keeps producing
    events without making progress. The counter is surfaced alongside
    :func:`_token_budget_abort_count` and :func:`_wall_clock_abort_count`
    as an auxiliary operator signal so the operator can distinguish
    event-limit-driven failures from other abort reasons.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    sessions_with_abort: set[str] = set()
    for event in logger.query_events(
        kind=TASK_ABORTED_KIND,
        harness_version=harness_version,
    ):
        if event.payload.get("reason") == EVENT_LIMIT_REASON:
            sessions_with_abort.add(event.session_id)
    return len(sessions_with_abort)


def _model_retry_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count transient model API retry events emitted by the runner (issue #871).

    The runner records one ``model_retry`` event for every failed model API
    attempt that the adapter retries. The count is aggregated across matching
    sessions so operators can spot provider instability and API reliability
    degradation. The kind and ``harness_version`` filters are pushed down to
    one :meth:`TraceLogger.query_events` cursor.
    """
    count = 0
    for event in logger.query_events(
        kind=MODEL_RETRY_KIND,
        harness_version=harness_version,
    ):
        count += 1
    return count


def _tool_argument_parse_error_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count ``tool_argument_parse_error`` events emitted by the runner (issue #872).

    The runner records one such event each time the model emits a tool-call
    whose ``arguments`` JSON cannot be parsed (or is not a JSON object) — see
    ``foundry_x.execution.runner._parse_tool_arguments``. The runner still
    proceeds with an empty ``arguments`` dict so the loop survives, so a
    rising count is the only signal that the model is producing malformed
    tool calls (schema mismatch, instruction drift, etc.).

    A rising rate correlates with model output quality degradation, so this
    counter is surfaced alongside :func:`_wall_clock_abort_count` as an
    auxiliary operator signal — scalar, session-aggregated, fit for the
    ``foundry-kpis`` markdown table and the baseline/candidate delta column.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    count = 0
    for event in logger.query_events(
        kind=TOOL_ARGUMENT_PARSE_ERROR_KIND,
        harness_version=harness_version,
    ):
        count += 1
    return count


def _server_restart_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count ``server_unavailable`` events emitted by the runner (issue #899).

    The runner records one such event each time the
    :class:`~foundry_x.infra.server_manager.FoundryServerManager` reports
    that ``GET /health`` returned a non-200 status and triggers the
    supervisor's bounded exponential-backoff restart loop (see
    ``foundry_x.execution.runner._handle_server_unavailable``). A rising
    count correlates with infrastructure reliability regressions
    (``llama-server`` crashes, GPU OOM, host reboots) rather than
    model-quality signals, so it is surfaced alongside
    :func:`_wall_clock_abort_count` as an auxiliary operator signal.

    The counter is scalar and session-aggregated, fit for the
    ``foundry-kpis`` markdown table and the baseline/candidate delta
    column. Uses one :meth:`TraceLogger.query_events` cursor with the
    kind and ``harness_version`` filters pushed down so a multi-session
    fixture does not need to be materialized in Python (matches the
    per-KPI helper convention introduced in issue #273).
    """
    count = 0
    for event in logger.query_events(
        kind=SERVER_UNAVAILABLE_KIND,
        harness_version=harness_version,
    ):
        count += 1
    return count


def _evolver_llm_failure(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[int, float]:
    """Count ``generation_exhausted`` events and compute session-fraction failure rate (issue #953).

    Returns ``(failure_count, failure_rate)`` where ``failure_count`` is the
    total number of ``generation_exhausted`` events emitted when the evolver's
    LLM generation retries were all exhausted without producing a valid edit,
    and ``failure_rate`` is the fraction of sessions with a ``task_received``
    event that also had at least one such event.

    A non-zero count signals that the LLM is failing to produce parseable,
    valid ProposedEdit objects after retries — a rising rate correlates with
    LLM provider flakiness or model-output degradation. Surfaced as an
    auxiliary operator signal alongside :func:`_model_retry_count` and
    :func:`_tool_argument_parse_error_count`.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    sessions_with_exhausted: set[str] = set()
    total_count = 0
    for event in logger.query_events(
        kind=GENERATION_EXHAUSTED_KIND,
        harness_version=harness_version,
    ):
        total_count += 1
        sessions_with_exhausted.add(event.session_id)

    if not sessions_with_exhausted:
        return 0, 0.0

    sessions_with_task: set[str] = set()
    for event in logger.query_events(kind="task_received", harness_version=harness_version):
        sessions_with_task.add(event.session_id)

    rate = len(sessions_with_exhausted) / len(sessions_with_task) if sessions_with_task else 0.0
    return total_count, rate


def _evolver_duration_ms(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> float | None:
    """Mean ``evolver_duration_ms`` from ``evolver_duration`` events (issue #1346).

    Queries every ``evolver_duration`` trace event emitted by
    :func:`~foundry_x.evolution.loop._emit_evolver_duration` and returns
    the mean of their ``evolver_duration_ms`` payload field.

    Returns ``None`` when no evolver phase was recorded for any session, so
    the field stays compact in the JSON output and operators can distinguish
    "no evolver ran" (None) from "evolver ran with 0 ms duration" (0.0).

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    durations: list[float] = []
    for event in logger.query_events(
        kind=EVOLVER_DURATION_KIND,
        harness_version=harness_version,
    ):
        ms = event.payload.get("evolver_duration_ms")
        if ms is not None:
            durations.append(ms)
    if not durations:
        return None
    return sum(durations) / len(durations)


def _model_cost_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[int, float]:
    """Count ``model_cost`` events and sum estimated_cost_usd (issue #1281).

    Returns ``(cost_count, total_cost_usd)`` where ``cost_count`` is the
    total number of ``model_cost`` events and ``total_cost_usd`` is the
    cumulative estimated cost in USD, summed from the ``estimated_cost_usd``
    field on each event payload.

    Surfaced as an auxiliary operator signal alongside
    :func:`_model_retry_count` and :func:`_server_restart_count`.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    total_cost = 0.0
    count = 0
    for event in logger.query_events(
        kind=MODEL_COST_KIND,
        harness_version=harness_version,
    ):
        count += 1
        cost = event.payload.get("estimated_cost_usd")
        if cost is not None:
            total_cost += cost
    return count, total_cost


def _model_rate_limit_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count ``model_rate_limit`` events emitted by the runner (issue #1281).

    The runner records one ``model_rate_limit`` event each time the
    CloudModelAdapter receives a rate-limit update from the provider. The
    count is aggregated across matching sessions so operators can monitor
    provider headroom.

    Surfaced as an auxiliary operator signal alongside
    :func:`_model_cost_count` and :func:`_fetch_blocked_count`.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    count = 0
    for event in logger.query_events(
        kind=MODEL_RATE_LIMIT_KIND,
        harness_version=harness_version,
    ):
        count += 1
    return count


def _fetch_blocked_count(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> int:
    """Count ``fetch_blocked`` events emitted by the WebFetchHook (issue #1281).

    The hook records one ``fetch_blocked`` event each time a ``web_fetch``
    tool call is blocked because the URL's host is not in FETCH_ALLOWED_DOMAINS.
    The count is aggregated across matching sessions so operators can audit
    which URLs are being blocked.

    Surfaced as an auxiliary operator signal alongside
    :func:`_model_rate_limit_count` and :func:`_model_cost_count`.

    Uses one :meth:`TraceLogger.query_events` cursor (issue #273) with the
    kind and ``harness_version`` filters pushed down.
    """
    count = 0
    for event in logger.query_events(
        kind=FETCH_BLOCKED_KIND,
        harness_version=harness_version,
    ):
        count += 1
    return count


def _percentile(sorted_values: list[float], q: float) -> float:
    """Return the *q*-th percentile of *sorted_values* using nearest-rank.

    Mirrors the formula used in ``tool_latency.aggregate_tool_latency`` so
    the hook overhead percentiles use the same deterministic method as tool
    execution latency percentiles.
    """
    import math

    n = len(sorted_values)
    if n == 0:
        return 0.0
    rank = max(1, math.ceil(q / 100.0 * n))
    return float(sorted_values[min(rank, n) - 1])


def _hook_overhead(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> tuple[dict[str, HookOverheadRow], float, float, float, float]:
    """Aggregate hook overhead percentiles per tool and overall (issue #1269).

    Streams ``tool_call`` events via :meth:`TraceLogger.query_events` and
    buckets ``hook_overhead_ms`` and ``hook_post_overhead_ms`` by tool name.
    Computes p50/p95 per tool and the overall aggregate across all tools.

    Returns
    -------
    ``(per_tool, overall_p50, overall_p95, post_p50, post_p95)`` where the
    scalar values are the overall percentiles across all tools (for
    :class:`KpiHistoryEntry`) and ``per_tool`` is the per-tool breakdown
    for :attr:`KpiSummary.hook_overhead`.
    """
    per_tool_hook: dict[str, list[float]] = {}
    per_tool_post: dict[str, list[float]] = {}
    all_hook: list[float] = []
    all_post: list[float] = []

    for event in logger.query_events(kind="tool_call", harness_version=harness_version):
        payload = event.payload or {}
        hook_ms = payload.get("hook_overhead_ms")
        post_ms = payload.get("hook_post_overhead_ms")
        name = payload.get("name")
        if not isinstance(name, str) or not name:
            continue
        if isinstance(hook_ms, (int, float)) and not isinstance(hook_ms, bool) and hook_ms >= 0:
            per_tool_hook.setdefault(name, []).append(float(hook_ms))
            all_hook.append(float(hook_ms))
        if isinstance(post_ms, (int, float)) and not isinstance(post_ms, bool) and post_ms >= 0:
            per_tool_post.setdefault(name, []).append(float(post_ms))
            all_post.append(float(post_ms))

    all_hook.sort()
    all_post.sort()
    overall_hook_p50 = _percentile(all_hook, 50.0)
    overall_hook_p95 = _percentile(all_hook, 95.0)
    overall_post_p50 = _percentile(all_post, 50.0)
    overall_post_p95 = _percentile(all_post, 95.0)

    rows: dict[str, HookOverheadRow] = {}
    all_tools = set(per_tool_hook.keys()) | set(per_tool_post.keys())
    for tool in all_tools:
        hook_vals = sorted(per_tool_hook.get(tool, []))
        post_vals = sorted(per_tool_post.get(tool, []))
        rows[tool] = HookOverheadRow(
            count=len(hook_vals) + len(post_vals),
            hook_overhead_ms_p50=_percentile(hook_vals, 50.0),
            hook_overhead_ms_p95=_percentile(hook_vals, 95.0),
            hook_post_overhead_ms_p50=_percentile(post_vals, 50.0),
            hook_post_overhead_ms_p95=_percentile(post_vals, 95.0),
        )

    return rows, overall_hook_p50, overall_hook_p95, overall_post_p50, overall_post_p95


def _format_value(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.2f}"


def _format_delta(
    baseline: float | None,
    candidate: float | None,
    higher_is_better: bool,
) -> str:
    """Render a candidate-minus-baseline delta with a PRD sign convention.

    Per issue #100 an *improvement-rate* increase is marked ``positive``
    (good) while a *regression-rate* or *cycle-time* increase is marked
    ``negative`` (bad). ``higher_is_better`` selects which polarity the
    PRD treats as favorable for the given KPI. A near-zero change is
    ``neutral``; an unmeasurable side (``None``) yields ``N/A``.
    """
    if baseline is None or candidate is None:
        return "N/A"
    delta = candidate - baseline
    if abs(delta) < 1e-9:
        mark = "neutral"
    elif (delta > 0) is higher_is_better:
        mark = "positive"
    else:
        mark = "negative"
    return f"{delta:+.2f} ({mark})"


def _render_markdown(summary: KpiSummary) -> str:
    lines = [
        "| KPI | Value |",
        "| --- | --- |",
        f"| Cycle Time (seconds) | {_format_value(summary.cycle_time_seconds)} |",
        # Issue #1338: cycle-time percentiles.
        f"| Cycle Time p50 (seconds) | {_format_value(summary.cycle_time_p50_seconds)} |",
        f"| Cycle Time p95 (seconds) | {_format_value(summary.cycle_time_p95_seconds)} |",
        f"| Regression Rate | {_format_value(summary.regression_rate)} |",
        f"| Improvement Rate | {_format_value(summary.improvement_rate)} |",
        f"| Hooks Disabled Count | {summary.hooks_disabled_count} |",
        f"| Hooks Disabled Rate | {_format_value(summary.hooks_disabled_rate)} |",
        f"| Token Budget Hit Rate | {_format_value(summary.token_budget_hit_rate)} |",
        f"| Token Budget Overrun % | {_format_value(summary.token_budget_overrun_pct)} |",
        f"| Context Efficiency | {_format_value(summary.context_efficiency)} |",
    ]
    # Issue #120: surface per-session ``injection_blocked`` counts only when
    # at least one session has ≥1 block; a clean trace store stays compact.
    if summary.injection_blocks:
        total = sum(summary.injection_blocks.values())
        lines.append("")
        lines.append(
            f"Injection Blocked: {total} block(s) across "
            f"{len(summary.injection_blocks)} session(s)."
        )
        lines.append("")
        lines.append("| Session | injection_blocked |")
        lines.append("| --- | --- |")
        for sid, count in sorted(summary.injection_blocks.items()):
            lines.append(f"| {sid} | {count} |")
    # Issue #271: surface per-session token consumption only when at least
    # one ``model_response`` carried a ``usage`` payload; a trace store with
    # no token accounting (budget never plumbed, or an endpoint that omits
    # usage) keeps the summary compact.
    if summary.token_totals:
        grand_total = sum(summary.token_totals.values())
        lines.append("")
        lines.append(
            f"Token Usage: {grand_total} token(s) across {len(summary.token_totals)} session(s)."
        )
        lines.append("")
        lines.append("| Session | Tokens |")
        lines.append("| --- | --- |")
        for sid, count in sorted(summary.token_totals.items()):
            lines.append(f"| {sid} | {count} |")
    if summary.token_budget_abort_count > 0:
        lines.append("")
        lines.append(
            f"Token Budget Aborts: {summary.token_budget_abort_count} session(s) "
            "hit the token budget limit."
        )
    # Issue #580: surface per-session streaming quality (avg TTFT) only when
    # at least one ``model_response`` carried timing data.
    if summary.streaming_quality:
        lines.append("")
        lines.append("Streaming Quality (avg TTFT):")
        lines.append("")
        lines.append("| Session | avg TTFT (ms) | total chunks | avg chunk interval (ms) |")
        lines.append("| --- | --- | --- | --- |")
        for sid, sq in sorted(summary.streaming_quality.items()):
            avg_ttft = _format_value(sq.avg_ttft_ms)
            avg_interval = _format_value(sq.avg_chunk_interval_ms)
            lines.append(f"| {sid} | {avg_ttft} | {sq.total_chunks} | {avg_interval} |")
    # Issue #626: surface per-session ``context_pruned`` counts only when
    # at least one session has ≥1 prune; a clean trace store stays compact.
    if summary.context_pruned_count:
        total = sum(summary.context_pruned_count.values())
        lines.append("")
        lines.append(
            f"Context Pruned: {total} prune(s) across "
            f"{len(summary.context_pruned_count)} session(s)."
        )
        lines.append("")
        lines.append("| Session | context_pruned |")
        lines.append("| --- | --- |")
        for sid, count in sorted(summary.context_pruned_count.items()):
            lines.append(f"| {sid} | {count} |")
    # Issue #711: surface wall-clock abort count as an auxiliary operator
    # signal. Zero means the timeout cap is not firing (expected for healthy
    # runs); non-zero means a session was aborted by FOUNDRY_TASK_TIMEOUT.
    if summary.wall_clock_abort_count > 0:
        lines.append("")
        lines.append(
            f"Wall-Clock Aborts: {summary.wall_clock_abort_count} session(s) "
            "were aborted by FOUNDRY_TASK_TIMEOUT."
        )
    # Issue #871: surface model API retries when at least one retry event was
    # recorded. A clean trace store stays compact, while any non-zero value is
    # immediately visible as a provider/API reliability signal.
    if summary.model_retry_count > 0:
        lines.append("")
        lines.append(
            f"Model Retries: {summary.model_retry_count} "
            "model API retry event(s) recorded by the runner."
        )
    # Issue #872: surface tool-call argument parse-error count when at least
    # one event was recorded. A rising rate signals model output quality
    # degradation or a schema mismatch, so this is operator-visible signal
    # only (clean store stays compact, mirroring wall-clock aborts).
    if summary.tool_argument_parse_error_count > 0:
        lines.append("")
        lines.append(
            f"Tool Argument Parse Errors: {summary.tool_argument_parse_error_count} "
            "malformed tool-call argument(s) emitted by the runner."
        )
    # Issue #869: surface event-limit abort count as an auxiliary operator
    # signal. Zero means the per-session event cap is not firing (expected
    # for healthy runs); non-zero means a session was aborted by
    # FOUNDRY_MAX_EVENTS_PER_SESSION, which usually indicates a runaway loop.
    if summary.event_limit_abort_count > 0:
        lines.append("")
        lines.append(
            f"Event Limit Aborts: {summary.event_limit_abort_count} session(s) "
            "hit the per-session event cap."
        )
    # Issue #899: surface server-restart count as an auxiliary operator
    # signal. Zero means the supervisor never saw an unhealthy /health
    # response (expected for healthy runs); non-zero means the
    # ``FoundryServerManager`` triggered its bounded restart loop at
    # least once during the harness-version window — typically a sign
    # of infrastructure flakiness, not a model-quality regression.
    if summary.server_restart_count > 0:
        lines.append("")
        lines.append(
            f"Server Restarts: {summary.server_restart_count} server_unavailable "
            "event(s) recorded by the runner's mid-session health-check."
        )
    # Issue #953: surface evolver LLM failure count when > 0. A non-zero count
    # signals that the evolver's LLM generation retries were exhausted without
    # producing a valid ProposedEdit — a rising rate correlates with provider
    # flakiness or model-output degradation.
    if summary.evolver_llm_failure_count > 0:
        lines.append("")
        lines.append(
            f"Evolver LLM Failures: {summary.evolver_llm_failure_count} "
            f"generation_exhausted event(s) "
            f"(rate: {_format_value(summary.evolver_llm_failure_rate)})."
        )
    # Issue #1281: surface model cost events when at least one was recorded.
    if summary.model_cost_count > 0:
        lines.append("")
        lines.append(
            f"Model Cost: {summary.model_cost_count} cost event(s) recorded "
            f"(total: ${summary.total_model_cost_usd:.4f} USD)."
        )
    # Issue #1281: surface model rate-limit events when at least one was recorded.
    if summary.model_rate_limit_count > 0:
        lines.append("")
        lines.append(
            f"Model Rate Limits: {summary.model_rate_limit_count} "
            "rate-limit event(s) recorded by the runner."
        )
    # Issue #1281: surface fetch_blocked events when at least one was recorded.
    if summary.fetch_blocked_count > 0:
        lines.append("")
        lines.append(
            f"Fetch Blocked: {summary.fetch_blocked_count} "
            "fetch_blocked event(s) — URL(s) blocked by WebFetchHook."
        )
    # Issue #895, #1113: surface the cycle-time exclusion count and its
    # per-abort-reason breakdown when > 0 so the survivorship bias in
    # ``cycle_time_seconds`` is visible — a high count means the mean is
    # computed over a small subpopulation of sessions that survived to a
    # ``critic_verdict``. Zero (the clean-store case) keeps the summary compact.
    if summary.excluded_from_cycle_time > 0:
        lines.append("")
        lines.append(
            f"Excluded From Cycle Time: {summary.excluded_from_cycle_time} "
            "session(s) had a task_received but no usable critic_verdict "
            "(failed before the Critic ran)."
        )
        # Issue #1113: show the breakdown by abort reason.
        total_breakdown = (
            summary.excluded_wall_clock
            + summary.excluded_token_budget
            + summary.excluded_event_limit
            + summary.excluded_other
        )
        if total_breakdown > 0:
            lines.append("")
            lines.append("| Abort Reason | Excluded Sessions |")
            lines.append("| --- | --- |")
            lines.append(f"| wall_clock | {summary.excluded_wall_clock} |")
            lines.append(f"| token_budget | {summary.excluded_token_budget} |")
            lines.append(f"| event_limit | {summary.excluded_event_limit} |")
            lines.append(f"| other | {summary.excluded_other} |")
    if summary.failure_class_distribution:
        total = sum(summary.failure_class_distribution.values())
        lines.append("")
        lines.append(
            f"Failure Class Distribution: {total} verdict(s) across "
            f"{len(summary.failure_class_distribution)} class(es)."
        )
        lines.append("")
        lines.append("| Failure Class | Count |")
        lines.append("| --- | --- |")
        for cls, count in sorted(summary.failure_class_distribution.items()):
            lines.append(f"| {cls} | {count} |")
    # Issue #898, #1039: render the populated per-slice breakdown (only the
    # dimension selected via ``--group-by`` is non-empty, so at most one
    # of these sections appears). Compact and omitted entirely when no
    # task/session metadata was supplied.
    for label, slices in (
        ("Skill", summary.per_skill),
        ("Task Family", summary.per_task_family),
        ("Difficulty Tier", summary.per_difficulty_tier),
        ("Model ID", summary.per_model_id),
        ("Quantization", summary.per_quantization),
        ("Harness Version", summary.per_harness_version),
    ):
        if slices:
            lines.extend(_render_slice_section(label, slices))
    return "\n".join(lines)


def _render_slice_section(label: str, slices: dict[str, SkillKpiSlice]) -> list[str]:
    """Render one per-slice breakdown table (issue #898).

    *label* is the human-facing dimension name (``"Skill"`` /
    ``"Task Family"`` / ``"Difficulty Tier"``); *slices* is the
    ``group -> SkillKpiSlice`` map. Groups are sorted for deterministic
    output. The verdict/session counts are included so the operator can
    tell a noisy 1-verdict slice from a stable 50-verdict one.
    """
    lines = [
        "",
        f"### Per-{label} Slices (issue #898)",
        "",
        f"| {label} | Improvement Rate | Regression Rate | Verdicts | Sessions |",
        "| --- | --- | --- | --- | --- |",
    ]
    for key in sorted(slices):
        s = slices[key]
        lines.append(
            f"| {key} | {_format_value(s.improvement_rate)} | "
            f"{_format_value(s.regression_rate)} | {s.verdict_count} | {s.session_count} |"
        )
    return lines


def _resolve_format(args_format: str | None, out: str | None) -> str:
    """Return ``"markdown"`` or ``"json"``.

    The explicit ``--format`` flag always wins. When unset, the format is
    inferred from the ``--out`` file extension (``.json`` → JSON);
    otherwise Markdown is returned. Issue #101 keeps the decision local to
    the CLI layer so the pydantic model remains the single source of truth.
    """
    if args_format is not None:
        return args_format
    if out is not None and Path(out).suffix.lower() == ".json":
        return "json"
    return "markdown"


def _load_task_metadata(path: Path) -> dict[str, TaskKpiMetadata]:
    """Load a ``--task-metadata`` JSON file into a ``TaskKpiMetadata`` map (issue #898).

    The JSON shape is ``{task_name: {skills, task_families, difficulty_tier}}``
    (all fields optional; the key supplies ``name``). Non-dict values are
    skipped so a partially-malformed file does not abort the whole run;
    a non-object top-level document raises :class:`ValueError` because no
    metadata can be recovered from it.
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError(f"--task-metadata JSON must be an object, got {type(data).__name__}")
    metadata: dict[str, TaskKpiMetadata] = {}
    for name, fields in data.items():
        if not isinstance(fields, dict):
            continue
        payload = {**fields, "name": name}
        metadata[name] = TaskKpiMetadata.model_validate(payload)
    return metadata


def _render_json(summary: KpiSummary) -> str:
    """Serialize a KPI summary as a stable JSON snapshot (issue #101)."""
    return summary.model_dump_json(indent=2)


def _render_comparison_markdown(baseline: KpiSummary, candidate: KpiSummary) -> str:
    """Render baseline / candidate / delta columns for the three PRD KPIs plus issue #800 additions.

    Issue #100 requires the comparison to surface a delta column whose
    sign convention follows the PRD: improvement-rate up is good,
    regression-rate and cycle-time up are bad. Issue #800 adds
    ``hooks_disabled_rate`` (lower is worse) and ``wall_clock_abort_count``
    (lower is worse) to the comparison table.
    """
    lines = [
        "| KPI | Baseline | Candidate | Delta |",
        "| --- | --- | --- | --- |",
        (
            "| Cycle Time (seconds) | "
            f"{_format_value(baseline.cycle_time_seconds)} | "
            f"{_format_value(candidate.cycle_time_seconds)} | "
            f"{_format_delta(baseline.cycle_time_seconds, candidate.cycle_time_seconds, higher_is_better=False)} |"
        ),
        # Issue #1338: cycle-time percentile rows (lower is better).
        (
            "| Cycle Time p50 (seconds) | "
            f"{_format_value(baseline.cycle_time_p50_seconds)} | "
            f"{_format_value(candidate.cycle_time_p50_seconds)} | "
            f"{_format_delta(baseline.cycle_time_p50_seconds, candidate.cycle_time_p50_seconds, higher_is_better=False)} |"
        ),
        (
            "| Cycle Time p95 (seconds) | "
            f"{_format_value(baseline.cycle_time_p95_seconds)} | "
            f"{_format_value(candidate.cycle_time_p95_seconds)} | "
            f"{_format_delta(baseline.cycle_time_p95_seconds, candidate.cycle_time_p95_seconds, higher_is_better=False)} |"
        ),
        (
            "| Regression Rate | "
            f"{_format_value(baseline.regression_rate)} | "
            f"{_format_value(candidate.regression_rate)} | "
            f"{_format_delta(baseline.regression_rate, candidate.regression_rate, higher_is_better=False)} |"
        ),
        (
            "| Improvement Rate | "
            f"{_format_value(baseline.improvement_rate)} | "
            f"{_format_value(candidate.improvement_rate)} | "
            f"{_format_delta(baseline.improvement_rate, candidate.improvement_rate, higher_is_better=True)} |"
        ),
        (
            "| Token Budget Hit Rate | "
            f"{_format_value(baseline.token_budget_hit_rate)} | "
            f"{_format_value(candidate.token_budget_hit_rate)} | "
            f"{_format_delta(baseline.token_budget_hit_rate, candidate.token_budget_hit_rate, higher_is_better=False)} |"
        ),
        (
            "| Token Budget Overrun % | "
            f"{_format_value(baseline.token_budget_overrun_pct)} | "
            f"{_format_value(candidate.token_budget_overrun_pct)} | "
            f"{_format_delta(baseline.token_budget_overrun_pct, candidate.token_budget_overrun_pct, higher_is_better=False)} |"
        ),
        (
            "| Context Efficiency | "
            f"{_format_value(baseline.context_efficiency)} | "
            f"{_format_value(candidate.context_efficiency)} | "
            f"{_format_delta(baseline.context_efficiency, candidate.context_efficiency, higher_is_better=True)} |"
        ),
        (
            "| Hooks Disabled Rate | "
            f"{_format_value(baseline.hooks_disabled_rate)} | "
            f"{_format_value(candidate.hooks_disabled_rate)} | "
            f"{_format_delta(baseline.hooks_disabled_rate, candidate.hooks_disabled_rate, higher_is_better=False)} |"
        ),
        (
            "| Wall Clock Abort Count | "
            f"{baseline.wall_clock_abort_count} | "
            f"{candidate.wall_clock_abort_count} | "
            f"{_format_delta(float(baseline.wall_clock_abort_count), float(candidate.wall_clock_abort_count), higher_is_better=False)} |"
        ),
        (
            "| Model Retry Count | "
            f"{baseline.model_retry_count} | "
            f"{candidate.model_retry_count} | "
            f"{_format_delta(float(baseline.model_retry_count), float(candidate.model_retry_count), higher_is_better=False)} |"
        ),
        (
            "| Tool Argument Parse Error Count | "
            f"{baseline.tool_argument_parse_error_count} | "
            f"{candidate.tool_argument_parse_error_count} | "
            f"{_format_delta(float(baseline.tool_argument_parse_error_count), float(candidate.tool_argument_parse_error_count), higher_is_better=False)} |"
        ),
        (
            "| Event Limit Abort Count | "
            f"{baseline.event_limit_abort_count} | "
            f"{candidate.event_limit_abort_count} | "
            f"{_format_delta(float(baseline.event_limit_abort_count), float(candidate.event_limit_abort_count), higher_is_better=False)} |"
        ),
        (
            "| Server Restart Count | "
            f"{baseline.server_restart_count} | "
            f"{candidate.server_restart_count} | "
            f"{_format_delta(float(baseline.server_restart_count), float(candidate.server_restart_count), higher_is_better=False)} |"
        ),
        # Issue #895: exclusion count is an auxiliary signal (lower is
        # better — fewer sessions lost to pre-Critic failures means less
        # survivorship bias in ``cycle_time_seconds``).
        (
            "| Excluded From Cycle Time | "
            f"{baseline.excluded_from_cycle_time} | "
            f"{candidate.excluded_from_cycle_time} | "
            f"{_format_delta(float(baseline.excluded_from_cycle_time), float(candidate.excluded_from_cycle_time), higher_is_better=False)} |"
        ),
        # Issue #1113: per-abort-reason exclusion breakdown.
        (
            "| Excl. wall_clock | "
            f"{baseline.excluded_wall_clock} | "
            f"{candidate.excluded_wall_clock} | "
            f"{_format_delta(float(baseline.excluded_wall_clock), float(candidate.excluded_wall_clock), higher_is_better=False)} |"
        ),
        (
            "| Excl. token_budget | "
            f"{baseline.excluded_token_budget} | "
            f"{candidate.excluded_token_budget} | "
            f"{_format_delta(float(baseline.excluded_token_budget), float(candidate.excluded_token_budget), higher_is_better=False)} |"
        ),
        (
            "| Excl. event_limit | "
            f"{baseline.excluded_event_limit} | "
            f"{candidate.excluded_event_limit} | "
            f"{_format_delta(float(baseline.excluded_event_limit), float(candidate.excluded_event_limit), higher_is_better=False)} |"
        ),
        (
            "| Excl. other | "
            f"{baseline.excluded_other} | "
            f"{candidate.excluded_other} | "
            f"{_format_delta(float(baseline.excluded_other), float(candidate.excluded_other), higher_is_better=False)} |"
        ),
        # Issue #953: evolver LLM failure count and rate.
        (
            "| Evol LLM Failure Count | "
            f"{baseline.evolver_llm_failure_count} | "
            f"{candidate.evolver_llm_failure_count} | "
            f"{_format_delta(float(baseline.evolver_llm_failure_count), float(candidate.evolver_llm_failure_count), higher_is_better=False)} |"
        ),
        (
            "| Evol LLM Failure Rate | "
            f"{_format_value(baseline.evolver_llm_failure_rate)} | "
            f"{_format_value(candidate.evolver_llm_failure_rate)} | "
            f"{_format_delta(baseline.evolver_llm_failure_rate, candidate.evolver_llm_failure_rate, higher_is_better=False)} |"
        ),
        # Issue #1281: model cost, rate limit, and fetch blocked counts.
        (
            "| Model Cost Count | "
            f"{baseline.model_cost_count} | "
            f"{candidate.model_cost_count} | "
            f"{_format_delta(float(baseline.model_cost_count), float(candidate.model_cost_count), higher_is_better=False)} |"
        ),
        (
            "| Total Model Cost (USD) | "
            f"{baseline.total_model_cost_usd:.4f} | "
            f"{candidate.total_model_cost_usd:.4f} | "
            f"{_format_delta(baseline.total_model_cost_usd, candidate.total_model_cost_usd, higher_is_better=False)} |"
        ),
        (
            "| Model Rate Limit Count | "
            f"{baseline.model_rate_limit_count} | "
            f"{candidate.model_rate_limit_count} | "
            f"{_format_delta(float(baseline.model_rate_limit_count), float(candidate.model_rate_limit_count), higher_is_better=False)} |"
        ),
        (
            "| Fetch Blocked Count | "
            f"{baseline.fetch_blocked_count} | "
            f"{candidate.fetch_blocked_count} | "
            f"{_format_delta(float(baseline.fetch_blocked_count), float(candidate.fetch_blocked_count), higher_is_better=False)} |"
        ),
        # Issue #1271: aggregate streaming quality TTFT rows (lower is better —
        # faster first token is improvement; slower is regression).
        (
            "| Mean TTFT (ms) | "
            f"{_format_value(baseline.streaming_quality_mean_ttft_ms)} | "
            f"{_format_value(candidate.streaming_quality_mean_ttft_ms)} | "
            f"{_format_delta(baseline.streaming_quality_mean_ttft_ms, candidate.streaming_quality_mean_ttft_ms, higher_is_better=False)} |"
        ),
        (
            "| p50 TTFT (ms) | "
            f"{_format_value(baseline.streaming_quality_p50_ttft_ms)} | "
            f"{_format_value(candidate.streaming_quality_p50_ttft_ms)} | "
            f"{_format_delta(baseline.streaming_quality_p50_ttft_ms, candidate.streaming_quality_p50_ttft_ms, higher_is_better=False)} |"
        ),
        (
            "| p95 TTFT (ms) | "
            f"{_format_value(baseline.streaming_quality_p95_ttft_ms)} | "
            f"{_format_value(candidate.streaming_quality_p95_ttft_ms)} | "
            f"{_format_delta(baseline.streaming_quality_p95_ttft_ms, candidate.streaming_quality_p95_ttft_ms, higher_is_better=False)} |"
        ),
        # Issue #1271: token per-step rows (higher is better — more tokens
        # per step means more efficient model usage).
        (
            "| Mean Prompt Tokens/Step | "
            f"{_format_value(baseline.mean_prompt_tokens_per_step)} | "
            f"{_format_value(candidate.mean_prompt_tokens_per_step)} | "
            f"{_format_delta(baseline.mean_prompt_tokens_per_step, candidate.mean_prompt_tokens_per_step, higher_is_better=True)} |"
        ),
        (
            "| Mean Completion Tokens/Step | "
            f"{_format_value(baseline.mean_completion_tokens_per_step)} | "
            f"{_format_value(candidate.mean_completion_tokens_per_step)} | "
            f"{_format_delta(baseline.mean_completion_tokens_per_step, candidate.mean_completion_tokens_per_step, higher_is_better=True)} |"
        ),
        # Issue #1269: hook overhead deltas (higher is worse — adds latency to every tool call).
        (
            "| Hook Overhead ms p50 | "
            f"{_format_value(baseline.hook_overhead_ms_p50)} | "
            f"{_format_value(candidate.hook_overhead_ms_p50)} | "
            f"{_format_delta(baseline.hook_overhead_ms_p50, candidate.hook_overhead_ms_p50, higher_is_better=False)} |"
        ),
        (
            "| Hook Overhead ms p95 | "
            f"{_format_value(baseline.hook_overhead_ms_p95)} | "
            f"{_format_value(candidate.hook_overhead_ms_p95)} | "
            f"{_format_delta(baseline.hook_overhead_ms_p95, candidate.hook_overhead_ms_p95, higher_is_better=False)} |"
        ),
        (
            "| Hook Post Overhead ms p50 | "
            f"{_format_value(baseline.hook_post_overhead_ms_p50)} | "
            f"{_format_value(candidate.hook_post_overhead_ms_p50)} | "
            f"{_format_delta(baseline.hook_post_overhead_ms_p50, candidate.hook_post_overhead_ms_p50, higher_is_better=False)} |"
        ),
        (
            "| Hook Post Overhead ms p95 | "
            f"{_format_value(baseline.hook_post_overhead_ms_p95)} | "
            f"{_format_value(candidate.hook_post_overhead_ms_p95)} | "
            f"{_format_delta(baseline.hook_post_overhead_ms_p95, candidate.hook_post_overhead_ms_p95, higher_is_better=False)} |"
        ),
    ]
    return "\n".join(lines)


def _render_comparison_json(comparison: KpiComparison) -> str:
    """Serialize a baseline-vs-candidate comparison as JSON (issue #100)."""
    return comparison.model_dump_json(indent=2)


def _render_comparison_slice_deltas(comparison: KpiComparison) -> str:
    """Render per-slice candidate-minus-baseline delta tables (issue #898).

    Appended under the main comparison table when
    :attr:`KpiComparison.slice_deltas` is non-empty. The rate columns are
    raw candidate-minus-baseline deltas (sign-agnostic — the rendering
    layer for the aggregate comparison applies the PRD "good direction"
    mark, but per-slice deltas are kept neutral so an operator can scan
    all groups at once). Candidate verdict counts anchor the delta so a
    reviewer can spot slices built from one or two verdicts.
    """
    labels = {
        "skill": "Skill",
        "task_family": "Task Family",
        "difficulty_tier": "Difficulty Tier",
        "model_id": "Model ID",
        "quantization": "Quantization",
        "harness_version": "Harness Version",
    }
    blocks: list[str] = []
    for dim, slices in comparison.slice_deltas.items():
        label = labels.get(dim, dim)
        lines = [
            f"### Per-{label} Slice Deltas (issue #898)",
            "",
            f"| {label} | Improvement Δ | Regression Δ | Candidate Verdicts |",
            "| --- | --- | --- | --- |",
        ]
        for key in sorted(slices):
            s = slices[key]
            lines.append(
                f"| {key} | {s.improvement_rate:+.4f} | "
                f"{s.regression_rate:+.4f} | {s.verdict_count} |"
            )
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks)


def _now_iso() -> str:
    """Return a UTC ISO-8601 timestamp with offset suffix.

    Issue #183 uses this to stamp each appended history row. The
    timezone-aware form keeps the line unambiguous when CI runs
    across multiple regions; ``datetime.fromisoformat`` (Python 3.11+)
    accepts the ``+00:00`` suffix without modification.
    """
    return datetime.now(UTC).isoformat()


def append_kpi_history(
    path: Path,
    summary: KpiSummary,
    harness_version: str | None = None,
) -> None:
    """Append one KPI snapshot to the append-only JSONL history log (issue #183).

    Each run produces exactly one line. The three PRD-KPI fields are
    emitted via :meth:`KpiSummary.model_dump` with ``injection_blocks``,
    ``token_totals``, ``streaming_quality``, and ``wall_clock_abort_count``
    excluded (the "minus per-session maps" half of the round-trip contract).
    ``hooks_disabled_count``, ``hooks_disabled_rate``,
    ``token_budget_abort_count``, ``token_budget_hit_rate``,
    ``token_budget_overrun_pct``, ``model_retry_count``,
    ``tool_argument_parse_error_count``,
    ``event_limit_abort_count``, ``server_restart_count``,
    ``evolver_llm_failure_count``, ``evolver_llm_failure_rate``,
    ``model_cost_count``, ``total_model_cost_usd``,
    ``model_rate_limit_count``, and ``fetch_blocked_count`` are scalar
    fields and are included so the trend table can show their drift
    across harness edits. Then ``timestamp`` and the optional
    ``harness_version`` are added. ``schema_version`` is written so
    readers can detect when they are running against an older schema
    (issue #1334). Parent directories are created on demand so the
    operator does not have to ``mkdir`` before the first run.
    ``failure_class_distribution`` is included so the trend table
    can show per-class deltas (issue #705).

    The file is opened in append mode and a single ``\\n``-terminated
    line is written per call, so concurrent appends from independent
    ``foundry-kpis`` invocations interleave cleanly at line
    boundaries rather than corrupting the JSON payload of the
    previous line.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = summary.model_dump(
        mode="json",
        exclude={
            "injection_blocks",
            "token_totals",
            "streaming_quality",
            "wall_clock_abort_count",
            # Issue #933: ``context_pruned_count`` is a per-session dict,
            # not a scalar trend metric — exclude it so the JSONL history
            # line stays compact and its key set stable.
            "context_pruned_count",
            # Issue #895: ``excluded_from_cycle_time`` is an auxiliary
            # coverage signal recomputed from the trace store on demand
            # (like the per-slice fields below), not a trend metric — keep
            # the JSONL history line compact and its key set stable.
            "excluded_from_cycle_time",
            # Issue #1113: the per-abort-reason breakdown is also an auxiliary
            # coverage signal recomputed from the trace store on demand.
            "excluded_wall_clock",
            "excluded_token_budget",
            "excluded_event_limit",
            "excluded_other",
            # Issue #898, #1039: per-slice breakdowns are an on-demand
            # diagnostic view (populated only with --group-by), not a
            # trend metric — exclude them so the JSONL history line stays
            # compact. They are recomputed from the trace store on demand.
            "per_skill",
            "per_task_family",
            "per_difficulty_tier",
            "per_model_id",
            "per_quantization",
            "per_harness_version",
            # Issue #1269: ``hook_overhead`` is a per-tool dict, not a scalar
            # trend metric — exclude it so the JSONL history line stays compact.
            # The four scalar hook overhead fields are included directly.
            "hook_overhead",
        },
    )
    payload["timestamp"] = _now_iso()
    if harness_version is not None:
        payload["harness_version"] = harness_version
    payload["schema_version"] = KPI_HISTORY_SCHEMA_VERSION
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload) + "\n")


def read_kpi_history(path: Path) -> list[KpiHistoryEntry]:
    """Read every line of the JSONL history log (issue #183).

    Returns entries in file order — which, for an append-only log,
    is chronological order. Blank lines are tolerated; lines that
    fail pydantic validation are skipped so a single malformed entry
    (e.g. written by a future schema-bumped version of the CLI)
    does not blank the trend table. A missing file yields an empty
    list so the caller can render the placeholder table without a
    precondition check.

    Issue #1334: when an entry's ``schema_version`` is lower than the
    module's ``KPI_HISTORY_SCHEMA_VERSION``, a warning is emitted so
    operators can detect when their history reader is older than the
    writer and may be silently dropping newly added fields.
    """
    if not path.exists():
        return []
    entries: list[KpiHistoryEntry] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                entry = KpiHistoryEntry.model_validate_json(stripped)
            except ValidationError:
                continue
            if entry.schema_version < KPI_HISTORY_SCHEMA_VERSION:
                warnings.warn(
                    f"KPI history entry has schema_version={entry.schema_version} "
                    f"but the module expects {KPI_HISTORY_SCHEMA_VERSION}. "
                    "Some fields may be missing. Consider upgrading your tools.",
                    UserWarning,
                    stacklevel=2,
                )
            entries.append(entry)
    return entries


def _sparkline(values: list[float | None]) -> str:
    """Render a minimal ASCII sparkline for a sequence of values.

    Uses Unicode block characters to approximate a bar-chart feel:
    ``▁▂▃▄▅▆▇█`` (U+2581–U+2588), where each character represents one
    data point scaled linearly across the min/max range.
    ``None`` values render as ``·`` and are excluded from the scale.
    """
    valid = [v for v in values if v is not None]
    if not valid:
        return "N/A"
    lo, hi = min(valid), max(valid)
    span = hi - lo
    blocks = "▁▂▃▄▅▆▇█"

    def _char(v: float | None) -> str:
        if v is None:
            return "·"
        if span == 0:
            idx = len(blocks) - 1
        else:
            idx = min(int((v - lo) / span * (len(blocks) - 1)), len(blocks) - 1)
        return blocks[idx]

    return "".join(_char(v) for v in values)


# Auxiliary reliability signals surfaced in the history trend table
# (issue #933).  Each tuple is (field_name, display_label).  The section
# appears only when at least one entry has a non-zero value, keeping the
# output compact when the harness is clean.
_RELIABILITY_SIGNALS: list[tuple[str, str]] = [
    ("model_retry_count", "Model Retries"),
    ("tool_argument_parse_error_count", "Tool Arg Parse Errors"),
    ("event_limit_abort_count", "Event-Limit Aborts"),
    ("server_restart_count", "Server Restarts"),
    ("token_budget_abort_count", "Token Budget Aborts"),
    ("token_budget_hit_rate", "Token Budget Hit Rate"),
    ("hooks_disabled_count", "Hooks Disabled"),
    ("hooks_disabled_rate", "Hooks Disabled Rate"),
    ("evolver_llm_failure_count", "Evol LLM Failures"),
    ("evolver_llm_failure_rate", "Evol LLM Failure Rate"),
    ("model_cost_count", "Model Costs"),
    ("total_model_cost_usd", "Total Model Cost (USD)"),
    ("model_rate_limit_count", "Model Rate Limits"),
    ("fetch_blocked_count", "Fetch Blocked"),
]


def render_history_markdown(
    entries: Sequence[KpiHistoryEntry],
    *,
    trend: bool = False,
) -> str:
    """Render a Markdown trend table from KPI history entries (issue #183).

    The table preserves file order, which is the same as append order
    for a JSONL log. Each row carries the timestamp plus the three
    PRD KPIs formatted with two decimals; ``None`` cycle times render
    as ``N/A`` (same convention as :func:`_render_markdown`).

    When *trend* is ``True`` (issue #622), three ASCII sparkline
    columns are appended to the table — one per numeric KPI — giving
    operators a quick visual read of direction without opening a chart.

    An empty history renders a single placeholder line so CI summary
    cells that template-embed the table are never completely blank.

    Plotting (matplotlib, ASCII sparklines) is explicitly out of
    scope per the issue; a pure table is the contract.

    Issue #705: a Failure Class Distribution section is appended when
    at least one entry carries a non-empty ``failure_class_distribution``.

    Issue #1334: a Schema Info section is appended showing the current
    ``KPI_HISTORY_SCHEMA_VERSION`` for debugging clarity.
    """
    if not entries:
        return "_No KPI history entries yet._"

    cycle_times = [e.cycle_time_seconds for e in entries]
    regression_rates = [e.regression_rate for e in entries]
    improvement_rates = [e.improvement_rate for e in entries]

    header = "| Timestamp | Cycle Time (s) | Regression Rate | Improvement Rate |"
    if trend:
        header += " Cycle Time | Reg. Rate | Impr. Rate |"
    lines = [header, "| --- | --- | --- | --- |" + (" --- | --- | --- |" if trend else "")]

    sparkline_cycle = _sparkline(cycle_times) if trend else None
    sparkline_reg = _sparkline(regression_rates) if trend else None
    sparkline_imp = _sparkline(improvement_rates) if trend else None

    for idx, entry in enumerate(entries):
        row = (
            f"| {entry.timestamp} | "
            f"{_format_value(entry.cycle_time_seconds)} | "
            f"{_format_value(entry.regression_rate)} | "
            f"{_format_value(entry.improvement_rate)} |"
        )
        if trend:
            sc = sparkline_cycle[idx] if sparkline_cycle else " "
            sr = sparkline_reg[idx] if sparkline_reg else " "
            si = sparkline_imp[idx] if sparkline_imp else " "
            row += f" {sc} | {sr} | {si} |"
        lines.append(row)
    # Reliability Signals section (issue #933): surface the auxiliary
    # signals that are persisted but were previously invisible in the
    # trend table.  The section appears only when at least one entry
    # carries a non-zero value so the default output stays byte-identical
    # to the pre-fix table when the harness is clean.
    if any(getattr(e, field, 0) for e in entries for field, _ in _RELIABILITY_SIGNALS):
        lines.append("")
        lines.append("### Reliability Signals")
        lines.append("")
        lines.append("| Signal | " + " | ".join(e.timestamp[:10] for e in entries) + " |")
        lines.append("| --- | " + " | ".join("---" for _ in entries) + " |")
        for field, label in _RELIABILITY_SIGNALS:
            row_parts = [f"| {label} |"]
            for entry in entries:
                value = getattr(entry, field, 0)
                if isinstance(value, float):
                    row_parts.append(f" {value:.2f} |")
                else:
                    row_parts.append(f" {value} |")
            lines.append("".join(row_parts))
    if any(entry.failure_class_distribution for entry in entries):
        lines.append("")
        lines.append("### Failure Class Distribution")
        lines.append("")
        lines.append(
            "| Failure Class | " + " | ".join(f"{e.timestamp[:10]}" for e in entries) + " |"
        )
        lines.append("| --- | " + " | ".join("---" for _ in entries) + " |")
        all_classes = sorted({cls for entry in entries for cls in entry.failure_class_distribution})
        for cls in all_classes:
            row = [f"| {cls} |"]
            for entry in entries:
                count = entry.failure_class_distribution.get(cls, 0)
                row.append(f" {count} |")
            lines.append("".join(row))
    lines.append("")
    lines.append("### Schema Info")
    lines.append("")
    lines.append(f"Schema version: {KPI_HISTORY_SCHEMA_VERSION}")
    return "\n".join(lines)


def export_prometheus(
    entries: Sequence[KpiHistoryEntry],
    *,
    metric_name_prefix: str = "foundryx",
) -> str:
    """Render Prometheus-format metrics from KPI history entries (issue #565).

    Emits one ``foundryx_kpi_entry`` sample per history entry per KPI
    with a ``kpi`` label identifying the metric. The ``harness_version``
    label is set to the entry's value or ``"unknown"`` if absent.
    This makes it straightforward to scrape and ingest into Grafana
    without a custom exporter.

    The metric is gauge-typed so the most recent value is always the
    current KPI state; the scrape timestamp becomes the ``timestamp``
    field in Prometheus (seconds since epoch).
    """
    if not entries:
        return f"# No KPI history entries — {metric_name_prefix}_kpi_entry is empty.\n"

    lines: list[str] = [
        f"# HELP {metric_name_prefix}_kpi_entry FoundryX KPI from history (issue #565)",
        f"# TYPE {metric_name_prefix}_kpi_entry gauge",
    ]
    for entry in entries:
        ts = entry.timestamp
        harness = entry.harness_version or "unknown"
        labels = f'harness_version="{harness}",kpi="cycle_time_seconds"'
        value = f"{entry.cycle_time_seconds:.6f}" if entry.cycle_time_seconds is not None else "NaN"
        lines.append(f"{metric_name_prefix}_kpi_entry{{{labels}}} {value} {ts}")

        labels = f'harness_version="{harness}",kpi="regression_rate"'
        lines.append(f"{metric_name_prefix}_kpi_entry{{{labels}}} {entry.regression_rate:.6f} {ts}")

        labels = f'harness_version="{harness}",kpi="improvement_rate"'
        lines.append(
            f"{metric_name_prefix}_kpi_entry{{{labels}}} {entry.improvement_rate:.6f} {ts}"
        )

    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class TaskValidationResult:
    """Result of metadata validation for one task (issue #958)."""

    task_name: str
    missing_skills: bool
    missing_tags: bool
    missing_difficulty_tier: bool
    passed_checks: int

    @property
    def is_missing_any_metadata(self) -> bool:
        """Return True if any metadata field is missing/empty."""
        return self.missing_skills or self.missing_tags or self.missing_difficulty_tier


def _get_passed_checks_per_task(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> dict[str, int]:
    """Count total passed_checks per task from the trace store (issue #958).

    Walks every ``critic_verdict`` event and sums ``passed_checks`` per task
    name. Tasks with no verdicts have a count of 0.
    """
    counts: dict[str, int] = {}
    for event in logger.query_events(kind="critic_verdict", harness_version=harness_version):
        record = VerdictRecord(**event.payload)
        for task in record.passed_checks:
            counts[task] = counts.get(task, 0) + 1
    return counts


def validate_task_metadata(
    logger: TraceLogger,
    harness_version: str | None = None,
) -> list[TaskValidationResult]:
    """Validate benchmark task metadata completeness (issue #958).

    Returns tasks that have incomplete metadata (empty requires_skills, empty
    tags, or difficulty_tier equal to the default "easy") along with their
    total passed_checks count from the trace store.

    The returned list is sorted by task name. Tasks with complete metadata
    are omitted from the list.

    Parameters
    ----------
    logger:
        A :class:`~foundry_x.trace.logger.TraceLogger`.
    harness_version:
        When provided, only sessions with this harness version are considered
        for the passed_checks count.

    Returns:
        A list of :class:`TaskValidationResult` objects, one per task with
        incomplete metadata. Empty if all tasks have complete metadata.
    """
    try:
        from benchmarks.registry import TASKS_DIR
    except ImportError:
        return []

    passed_checks = _get_passed_checks_per_task(logger, harness_version=harness_version)

    import importlib

    results: list[TaskValidationResult] = []
    for task_file in sorted(TASKS_DIR.glob("test_*.py")):
        module_name = f"benchmarks.tasks.{task_file.stem}"
        module = importlib.import_module(module_name)
        task = getattr(module, "TASK", None)
        if task is None:
            continue
        missing_skills = len(task.requires_skills) == 0
        missing_tags = len(task.tags) == 0
        missing_difficulty_tier = task.difficulty_tier == "easy"
        if missing_skills or missing_tags or missing_difficulty_tier:
            results.append(
                TaskValidationResult(
                    task_name=task.name,
                    missing_skills=missing_skills,
                    missing_tags=missing_tags,
                    missing_difficulty_tier=missing_difficulty_tier,
                    passed_checks=passed_checks.get(task.name, 0),
                )
            )
    return sorted(results, key=lambda r: r.task_name)


def _render_validation_markdown(results: list[TaskValidationResult]) -> str:
    """Render task metadata validation results as a Markdown table (issue #958).

    Parameters
    ----------
    results:
        List of :class:`TaskValidationResult` objects to render.

    Returns
    -------
    A Markdown-formatted string. Empty string if *results* is empty.
    """
    if not results:
        return ""

    lines: list[str] = [
        "### Task Metadata Validation (issue #958)",
        "",
        "| Task | Missing Skills | Missing Tags | Missing Difficulty | Passed Checks |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in results:
        lines.append(
            f"| {r.task_name} | "
            f"{'⚠️' if r.missing_skills else ' '} | "
            f"{'⚠️' if r.missing_tags else ' '} | "
            f"{'⚠️' if r.missing_difficulty_tier else ' '} | "
            f"{r.passed_checks} |"
        )
    lines.append("")
    return "\n".join(lines)


_TOKEN_BUDGET_OVERRUN_EPILOG = """
Environment variables for alert thresholds:
  FOUNDRY_TOKEN_BUDGET_OVERRUN_MAX
                        Exit 3 when token_budget_overrun_pct exceeds this value.
                        Example: FOUNDRY_TOKEN_BUDGET_OVERRUN_MAX=100.0
                        (issue #1354).
  FOUNDRY_CONTEXT_EFFICIENCY_MIN
                        Exit 2 when context_efficiency falls below this value.
                        (issue #1286).
"""


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="foundry-kpis",
        description="Compute and display the three PRD success-metric KPIs.",
        epilog=_TOKEN_BUDGET_OVERRUN_EPILOG,
    )
    parser.add_argument(
        "--trace-db",
        default="./logs/traces.db",
        help="Path to the trace SQLite database (default: ./logs/traces.db).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="Deprecated: use --trace-db instead.",
    )
    parser.add_argument(
        "--harness-version",
        default=None,
        help="Only consider sessions with this harness version.",
    )
    parser.add_argument(
        "--baseline-harness-version",
        default=None,
        help=(
            "Baseline harness version for a baseline-vs-candidate comparison"
            " (issue #100). Must be paired with --candidate-harness-version."
        ),
    )
    parser.add_argument(
        "--candidate-harness-version",
        default=None,
        help=(
            "Candidate harness version for a baseline-vs-candidate comparison"
            " (issue #100). Must be paired with --baseline-harness-version."
        ),
    )
    parser.add_argument(
        "--group-by",
        dest="group_by",
        choices=(
            "skill",
            "task_family",
            "difficulty_tier",
            "model_id",
            "quantization",
            "harness_version",
        ),
        default=None,
        help=(
            "Break improvement_rate and regression_rate down by this dimension"
            " (issues #898, #1039). Task-level: 'skill' (per harness skill,"
            " from BenchmarkTask.requires_skills), 'task_family' (per"
            " BenchmarkTask tag), or 'difficulty_tier'"
            " (smoke/easy/medium/hard). Session-level: 'model_id',"
            " 'quantization', or 'harness_version' (per-session"
            " TraceSession attributes). Task-level dimensions require"
            " task metadata (--task-metadata); session-level dimensions"
            " use session attributes directly."
            " Works in both single-summary and baseline-vs-candidate"
            " comparison modes."
            " NOTE: Tasks with missing metadata (empty requires_skills, empty"
            " tags, or difficulty_tier='easy' by default) are silently"
            " excluded from task-level slice views. Sessions with None"
            " for the chosen session attribute are excluded from"
            " session-level slice views."
        ),
    )
    parser.add_argument(
        "--task-metadata",
        dest="task_metadata",
        default=None,
        help=(
            "JSON file mapping task names to {skills, task_families,"
            " difficulty_tier}. When --group-by is set and this is omitted,"
            " metadata is auto-built from benchmarks.registry.load_all_tasks()"
            " (issue #898). Use this flag to supply metadata outside the"
            " repo or to override the registry."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default=None,
        help=(
            "Output format. Default: 'markdown'. When --out ends in '.json',"
            " 'json' is selected automatically."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write output to this path instead of stdout.",
    )
    parser.add_argument(
        "--log-to",
        default=None,
        help=(
            "Append the single-summary KPI snapshot to this JSONL history"
            " log (issue #183). The per-session injection_blocks map is"
            " excluded; only the three PRD KPIs plus a timestamp and"
            " optional harness version are persisted. Comparison runs"
            " do not log — the history is per single-summary run."
        ),
    )
    parser.add_argument(
        "--from-history",
        default=None,
        help=(
            "Render a Markdown trend table from this JSONL history log"
            " (issue #183) and exit without reading the trace store."
            " The file is read in append order; missing or empty files"
            " render a placeholder table."
        ),
    )
    parser.add_argument(
        "--alert-threshold",
        type=float,
        default=None,
        help=(
            "Exit non-zero when the computed regression_rate exceeds this"
            " threshold (issue #565). Applies only to live KPI computation"
            " (not --from-history). Example: --alert-threshold 0.1"
            " causes a non-zero exit when regression_rate > 0.1."
        ),
    )
    parser.add_argument(
        "--cycle-time-alert-threshold",
        type=float,
        default=None,
        dest="cycle_time_alert_threshold",
        help=(
            "Exit non-zero when cycle_time_seconds exceeds this value (issue #621)."
            " The exit message names the triggering KPI and value."
        ),
    )
    parser.add_argument(
        "--export-prometheus",
        action="store_true",
        default=False,
        help=(
            "Emit Prometheus-format metrics to stdout (issue #565)."
            " When used with --from-history, reads the JSONL history log"
            " and exports one sample per entry per KPI. The output is"
            " compatible with Prometheus scraping and Grafana ingestion."
        ),
    )
    parser.add_argument(
        "--trend",
        action="store_true",
        default=False,
        help=(
            "Append ASCII sparkline columns to the Markdown trend table"
            " (issue #565). Requires --from-history. Each KPI gets a"
            " Unicode-block sparkline showing the full history at a glance."
        ),
    )
    parser.add_argument(
        "--validate-metadata",
        action="store_true",
        default=False,
        dest="validate_metadata",
        help=(
            "Validate benchmark task metadata completeness and emit a table of"
            " tasks with missing metadata (issue #958). Exits 0 and prints"
            " an empty table if all tasks are fully annotated. Tasks with"
            " missing skills, tags, or difficulty_tier (using the default"
            " 'easy') are excluded from --group-by slice views."
        ),
    )
    args = parser.parse_args(argv)

    baseline_version = args.baseline_harness_version
    candidate_version = args.candidate_harness_version
    if (baseline_version is None) != (candidate_version is None):
        parser.error(
            "--baseline-harness-version and --candidate-harness-version must be supplied together"
        )

    if args.trend and args.from_history is None:
        parser.error(
            "--trend requires --from-history: sparklines are rendered from the KPI history log"
        )

    if args.group_by is not None and args.from_history is not None:
        parser.error(
            "--group-by cannot be combined with --from-history: per-slice"
            " breakdowns are computed live from the trace store, not the"
            " history log"
        )

    if args.from_history is not None:
        # Issue #183: trend rendering is a pure read of the JSONL log;
        # it does not require a trace store, so we short-circuit before
        # opening the SQLite database. ``--out`` still works as a sink.
        entries = read_kpi_history(Path(args.from_history))
        if args.export_prometheus:
            output = export_prometheus(entries)
        else:
            output = render_history_markdown(entries, trend=args.trend)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0

    if args.validate_metadata:
        logger = TraceLogger(_get_trace_db(args))
        results = validate_task_metadata(logger, harness_version=args.harness_version)
        output = _render_validation_markdown(results)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0

    fmt = _resolve_format(args.format, args.out)
    logger = TraceLogger(_get_trace_db(args))

    # Issue #898, #1039: build the task-name -> metadata map only for
    # task-level dimensions; session-level dimensions (model_id,
    # quantization, harness_version) do not need task metadata.
    task_metadata = None
    if args.group_by is not None and args.group_by in _TASK_LEVEL_DIMS:
        task_metadata = (
            _load_task_metadata(Path(args.task_metadata))
            if args.task_metadata is not None
            else build_task_metadata()
        )

    if baseline_version is not None and candidate_version is not None:
        comparison = compare_kpis(
            logger,
            baseline_version,
            candidate_version,
            group_by=args.group_by,
            task_metadata=task_metadata,
        )
        if fmt == "json":
            output = _render_comparison_json(comparison)
        else:
            output = _render_comparison_markdown(comparison.baseline, comparison.candidate)
            if comparison.slice_deltas:
                output += "\n\n" + _render_comparison_slice_deltas(comparison)
        if args.out:
            Path(args.out).write_text(output, encoding="utf-8")
        else:
            print(output)
        return 0
    else:
        summary = compute_kpis(
            logger,
            harness_version=args.harness_version,
            group_by=args.group_by,
            task_metadata=task_metadata,
        )
        if args.log_to is not None:
            append_kpi_history(
                Path(args.log_to),
                summary,
                harness_version=args.harness_version,
            )
        if (
            args.cycle_time_alert_threshold is not None
            and summary.cycle_time_seconds is not None
            and summary.cycle_time_seconds > args.cycle_time_alert_threshold
        ):
            print(
                f"ALERT: cycle_time_seconds ({summary.cycle_time_seconds:.2f}) exceeds "
                f"threshold ({args.cycle_time_alert_threshold:.2f})",
                file=sys.stderr,
            )
            return 1
        output = _render_json(summary) if fmt == "json" else _render_markdown(summary)

    if args.out:
        Path(args.out).write_text(output, encoding="utf-8")
    else:
        print(output)

    # Issue #565: alert threshold — regression rate above threshold triggers CI gate.
    if args.alert_threshold is not None and summary.regression_rate > args.alert_threshold:
        sys.stderr.write(
            f"[ALERT] regression_rate {summary.regression_rate:.4f}"
            f" exceeds threshold {args.alert_threshold:.4f}\n"
        )
        return 1

    # Issue #1286: FOUNDRY_CONTEXT_EFFICIENCY_MIN triggers exit 2 when efficiency
    # falls below the configured floor. Backward-compatible: absent env var is ignored.
    min_efficiency = os.environ.get("FOUNDRY_CONTEXT_EFFICIENCY_MIN")
    if (
        min_efficiency is not None
        and summary.context_efficiency is not None
        and summary.context_efficiency < float(min_efficiency)
    ):
        sys.stderr.write(
            f"[ALERT] context_efficiency {summary.context_efficiency:.4f}"
            f" is below FOUNDRY_CONTEXT_EFFICIENCY_MIN={min_efficiency}\n"
        )
        return 2

    # Issue #1354: FOUNDRY_TOKEN_BUDGET_OVERRUN_MAX triggers exit 3 when overrun
    # exceeds the configured ceiling. Backward-compatible: absent env var is ignored.
    max_overrun = os.environ.get("FOUNDRY_TOKEN_BUDGET_OVERRUN_MAX")
    if (
        max_overrun is not None
        and summary.token_budget_overrun_pct is not None
        and summary.token_budget_overrun_pct > float(max_overrun)
    ):
        sys.stderr.write(
            f"[ALERT] token_budget_overrun_pct {summary.token_budget_overrun_pct:.4f}"
            f" exceeds FOUNDRY_TOKEN_BUDGET_OVERRUN_MAX={max_overrun}\n"
        )
        return 3

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
