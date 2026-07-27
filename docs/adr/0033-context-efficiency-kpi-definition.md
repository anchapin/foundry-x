# ADR-0033: `context_efficiency` KPI definition

## Status

Accepted. 2026-07-27.

## Context

ADR-0021 §6 recorded the `context_pruned` event payload contracts and noted:

> The `foundry-kpis` tool does not yet compute a `context_efficiency` KPI.
> When issue #553 acceptance is confirmed, a future ADR should define:
>
> ```
> context_efficiency = 1 - (dropped_events / total_events_in_session)
> ```

Issues #951 and #979 implemented the KPI. This ADR formalizes the definition,
resolving the ambiguity in the ADR-0021 placeholder formula (which did not
distinguish between event-count pruning and token-aware pruning) and
documenting the survivorship-bias correction from issue #979.

## Decision

### 1. What it measures

`context_efficiency` measures how much of the configured pruning budget
(`FOUNDRY_CONTEXT_TOKENS` / `threshold`) was consumed before the hook
dropped events. A value near **1.0** means the session ran to completion
or to the abort threshold without triggering pruning — the context window
was sufficient for the task. A value near **0.0** means the hook dropped
events throughout the session, indicating either an overly tight budget or a
task whose context footprint exceeds the configured threshold.

It is an **auxiliary operator signal**, not one of the three PRD success-metric
KPIs. It is exposed via `foundry-kpis` and in the baseline/candidate comparison
delta column alongside `token_budget_hit_rate` and `hooks_disabled_rate`.

### 2. How it is computed

The implementation in `src/foundry_x/observability/kpis.py:_context_efficiency`
(issues #951, #979) computes per-session efficiency as:

```
per_session_efficiency = 1 - (sum(dropped) / sum(threshold + dropped))
```

where `dropped` and `threshold` are summed across every `context_pruned`
event in the session:

| Pruner | `dropped` source | `threshold` source |
|---|---|---|
| Event-count (`FOUNDRY_CONTEXT_TOKENS` absent) | `payload.dropped` | `payload.threshold` |
| Token-aware (`FOUNDRY_CONTEXT_TOKENS` set) | `payload.dropped` | `payload.threshold_tokens` |

The denominator (`threshold + dropped`) is the effective budget: the
configured ceiling plus the events that were added beyond it. A session
that never fires the pruning hook contributes `0 / 0` for that session,
which is defined as **1.0** (perfect efficiency — nothing was dropped).

The aggregate KPI is the **mean** of per-session efficiencies across all
sessions matching the harness version filter.

**Survivorship-bias correction (issue #979):** Earlier versions of the
implementation silently excluded sessions with zero `context_pruned` events
from the mean. This created upward bias because only sessions that
actually pruned were counted. Sessions that never prune now contribute
`1.0`, so the mean reflects the full population.

### 3. Interaction with `token_budget_hit_rate`

`context_efficiency` and `token_budget_hit_rate` address different failure
modes:

| KPI | Trigger | Indicates |
|---|---|---|
| `token_budget_hit_rate` | `task_aborted(reason="token_budget")` | Hard abort — loop terminated |
| `context_efficiency` | `context_pruned` events | Proactive pruning — loop continued |

A session that frequently prunes but never hits the token budget is
*efficient but possibly under-configured*: the task may be solvable in
less context. A session that hits the token budget has exhausted both
pruning and the context window — the task may require a larger budget or
a more aggressive pruning strategy.

Both KPIs together give operators a two-axis signal:

```
high efficiency + low abort rate  → healthy
high efficiency + high abort rate   → budget too tight for this task
low efficiency + low abort rate     → pruning is aggressive but working
low efficiency + high abort rate    → prune-and-still-overshoot (rare)
```

### 4. Good and bad value ranges

| Range | Interpretation |
|---|---|
| 0.95 – 1.00 | Excellent. The configured budget was sufficient; pruning rarely fired. |
| 0.80 – 0.95 | Healthy. Normal pruning activity; the budget is appropriately sized. |
| 0.50 – 0.80 | Marginal. Either the task has high context footprint or the threshold is set aggressively. Investigate `context_pruned_count` per session and whether `FOUNDRY_CONTEXT_TOKENS` should be raised. |
| 0.00 – 0.50 | Problematic. Severe pruning throughout sessions. Likely causes coherence degradation (the model is operating on a heavily truncated history). Raise `FOUNDRY_CONTEXT_TOKENS` or reduce task complexity. |

Operators should treat the boundary at **0.80** as a signal threshold:
a regression from 0.90 to 0.60 across harness versions warrants investigation
even if `token_budget_hit_rate` is unchanged.

### 5. How it feeds into `kpi-improvement-rate`

`context_efficiency` is an **auxiliary reliability signal**, not an
ingredient of `kpi-improvement-rate`. It is surfaced in two ways:

1. **Trend table** (`foundry-kpis --trend`): the sparkline and delta
   columns show direction across harness versions so operators can spot
   regressions in context utilization alongside the three PRD KPIs.

2. **Baseline/candidate comparison** (`foundry-kpis --baseline <v1> --candidate <v2>`):
   the delta column renders `context_efficiency` with `higher_is_better=True`,
   matching the sign convention for `improvement_rate`. A negative delta
   (lower efficiency in the candidate) flags a potential context-utilization
   regression even when the candidate's `improvement_rate` is unchanged.

The PRD's `improvement_rate` measures whether harness edits increase the
fraction of approved proposals. `context_efficiency` measures whether those
edits degrade the context budget — a harness change that increases
`improvement_rate` but drops `context_efficiency` by 0.30 pp has improved
task success at the cost of context utilization, and the operator should
evaluate whether that tradeoff is acceptable.

## Consequences

- `context_efficiency` is now formally defined and documented.
- The ADR-0021 §6 placeholder formula is superseded by the
  `1 - (sum(dropped) / sum(threshold + dropped))` per-session definition.
- The survivorship-bias correction (sessions with zero pruning contribute
  `1.0`) is part of the official definition.
- No code changes are required; this ADR is a documentation artifact
  codifying the implementation from issues #951 and #979.

## Cross-References

- [ADR-0021 §6](./0021-context-pruning-at-scale.md#6-kpi-layer-integration):
  original `context_pruned` payload contracts and the placeholder formula.
- [`src/foundry_x/observability/kpis.py`](../../src/foundry_x/observability/kpis.py):
  `_context_efficiency` implementation (lines 1348–1400).
- [`src/foundry_x/observability/kpis.py`](../../src/foundry_x/observability/kpis.py):
  `KpiSummary.context_efficiency` field docstring (lines 244–252).
- [`src/foundry_x/observability/kpis.py`](../../src/foundry_x/observability/kpis.py):
  `KpiComparison.deltas` delta rendering with `higher_is_better=True`
  (`_render_comparison_markdown`, lines 1889–1894).
- [`harness/hooks/context_pruning.py`](../../harness/hooks/context_pruning.py):
  `TokenAwarePruningHook` and `ContextPruningHook` implementation.
- [`docs/CONTEXT.md`](../../docs/CONTEXT.md): `context_pruned` event
  payload contract documentation.
- Issue #951: `context_efficiency` KPI addition.
- Issue #979: survivorship-bias correction for sessions with zero pruning.
