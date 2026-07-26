# FoundryX GitHub Issue Discovery — Final Portfolio

**Generated:** 2026-07-18
**Repository:** anchapin/foundry-x (branch: develop, HEAD: 99d833bb)
**Workflow:** ISSUE_GENERATION_PROMPT.md — Stages 0–5 complete

---

## Execution Summary

| Stage | Status | Details |
|-------|--------|---------|
| Stage 0: Repo rules | ✅ | Read all 12 required files + 22 ADRs |
| Stage 1: Evidence pack | ✅ | 409 issues, 413 PRs, 200 labels, trace evidence |
| Stage 2: Scout sub-agents | ✅ | 9 scouts launched, 8 completed + 1 relaunched (13 proposals) |
| Stage 3: Validation | ✅ | 4 parallel validators (evidence, duplicate, architecture, KPI/QA) |
| Stage 4: Scoring & gates | ✅ | Hard gates applied, conflicts resolved |
| Stage 5: Live recheck & synthesis | ✅ | All surviving proposals verified against live codebase |

**Scout coverage:** trace, execution, evolution, observability, benchmarks, harness, infra, docs, security

---

## Eliminated Proposals (8)

| ID | Reason | Evidence |
|----|--------|----------|
| trace-01 | **SUPERSEDED** | Issue #740 closed, PR #772 merged 2026-07-17 — already fixed |
| execution-01 | **SUPERSEDED** | Issue #739 closed, PR #774 merged 2026-07-17 — already fixed |
| evolution-01 | **SUPERSEDED** | Issue #188 closed, PR #240 merged 2026-07-11 — Critic timeout already implemented |
| evolution-02 | **PHANTOM** | `_run_loop_async` does not exist in codebase; `cli.py` has `async def n()` — proposal references stale/renamed code |
| evolution-03 | **PHANTOM** | No trace evidence of async Critic verdict loss; PR #42 already covers verdict persistence |
| OBS-007 | **SUPERSEDED** | PR #571 added threshold alerting for regression_rate; PR #679 extended to cycle_time — alerting framework already exists |
| harness-01 | **BLOCKED (ADR-0004)** | Adding `_phase = 4` to `harness/hooks/token_aware_pruning.py` is a harness hand-edit — must go through Evolver→Critic loop, not an issue |
| security-01 | **PHANTOM** | `resolve_bash_path` does not exist anywhere in `src/` or `harness/` — function name is wrong or code was already removed |

---

## Final Ranked Portfolio (12 proposals)

### Tier 1: High Confidence, High Impact (5 proposals)

#### 1. OBS-001 — Add `model_error` to Digester `FAILURE_KINDS`
- **Area:** observability
- **Effort:** size-s
- **Confidence:** HIGH
- **Score:** 92/100

**Problem:** The runner emits `model_error` (runner.py:1592) when `adapter.complete` raises, and CONTEXT.md documents it as a failure signal. But the Digester's `FAILURE_KINDS` (digester.py:61-69) does not include `model_error`. The Digester's failure-walk (line 211) ignores `model_error` as a first-failure marker, misclassifying root causes.

**Evidence:**
- `src/foundry_x/execution/runner.py:1592` — `kind="model_error"`
- `src/foundry_x/evolution/digester.py:61-69` — `FAILURE_KINDS = {"tool_error", "task_failed", "run_failed", "agent_error", "error"}` — no `model_error`
- `src/foundry_x/evolution/digester.py:211` — `if event.kind in FAILURE_KINDS:` — silently skips `model_error`

**Impact:** When `model_error` precedes other failures in a session, the Digester reports the later failure as root cause instead of the actual model error.

**KPI:** Advances `kpi-regression-rate` (correct root-cause classification prevents misattributed regressions)

**Proposed fix:** Add `"model_error"` and `"hook_registry_error"` to `FAILURE_KINDS` frozenset in digester.py:61-69. Also extend `FAILURE_PAYLOAD_KEYS` to include `"error_type"` and `"message"` (the payload keys emitted by `model_error`).

**Acceptance criteria:**
- A session with `model_error` at step 3 followed by `tool_error` at step 5 classifies `model_error` as the first failure
- `uv run pytest tests/evolution/` passes

---

#### 2. OBS-002 — Add `hook_registry_error` to Digester `FAILURE_KINDS`
- **Area:** observability
- **Effort:** size-s
- **Confidence:** HIGH (was medium, upgraded after evidence verification)
- **Score:** 88/100

**Problem:** The runner emits `hook_registry_error` (runner.py:675) when `harness.hooks.get_registry()` raises, silently disabling ALL hooks including the security-critical `InjectionFirewallHook`. KPIs track the count (#585), but the Digester's `FAILURE_KINDS` does not include it. Sessions with disabled security hooks pass Digester without security classification.

**Evidence:**
- `src/foundry_x/execution/runner.py:675` — `kind="hook_registry_error"`
- `src/foundry_x/evolution/digester.py:61-69` — absent from `FAILURE_KINDS`
- `src/foundry_x/observability/kpis.py:506` — `_hook_registry_errors()` tracks count but Digester ignores it

**Impact:** Sessions with disabled injection firewall pass evolution loop without security flagging.

**KPI:** Advances `kpi-regression-rate` (security-relevant failures must be classified)

**Recommendation:** Merge with OBS-001 into a single "Extend Digester FAILURE_KINDS" proposal — both add failure kinds to the same frozenset in the same file.

---

#### 3. infra-01 — Extend uv-pin test allowlist to all workflows
- **Area:** infra
- **Effort:** size-s
- **Confidence:** HIGH
- **Score:** 87/100

**Problem:** `tests/infra/test_workflow_uv_pin.py:32` hard-codes `EXPECTED_WORKFLOWS = ["ci.yml", "audit.yml", "critic.yml", "docker.yml"]`, but 13 workflow file references use the hash-verified `install-uv` composite action. The parametrized tests cannot detect a regression in lint.yml, pre-commit.yml, test.yml, quantization-sweep.yml, or real-llm.yml — any of those could be edited to call `pip install --upgrade uv` and the supply-chain guardrail (issue #208, ADR-0002, SECURITY.md threat #3) would silently re-open.

**Evidence:**
- `tests/infra/test_workflow_uv_pin.py:32` — `EXPECTED_WORKFLOWS = ["ci.yml", "audit.yml", "critic.yml", "docker.yml"]` (4 workflows)
- `.github/workflows/*.yml` — 13 `install-uv` references across 9 distinct workflow files
- `docs/SECURITY.md:23` — threat #3 requires supply-chain pinning
- `docs/adr/0002-uv-for-dependency-management.md:36-37` — ADR-0002 mandates `uv pip audit` in CI

**Impact:** 5 of 9 workflows using the composite action are unprotected by the regression test.

**KPI:** Advances `kpi-regression-rate` (supply-chain guardrail regression detection)

**Proposed fix:** Replace hard-coded list with glob-driven discovery of `.github/workflows/*.yml`, filter non-workflow files, deduplicate, sort for stable ordering.

---

#### 4. OBS-005 — Track event_limit aborts as a separate KPI
- **Area:** observability
- **Effort:** size-s
- **Confidence:** HIGH
- **Score:** 85/100

**Problem:** The runner emits `task_aborted(reason="event_limit")` when the per-session event cap is exceeded. KPIs track `token_budget_abort_count` (#466) and `wall_clock_abort_count` (#626), but event_limit aborts are invisible. SECURITY.md promises `event_limit_hit` as a tracked metric but kpis.py does not implement it.

**Evidence:**
- `src/foundry_x/execution/runner.py:103,419,420,438,1523` — event_limit abort mechanism exists
- `src/foundry_x/observability/kpis.py` — zero hits for `event_limit`
- `docs/SECURITY.md` — documents `event_limit_hit` as a tracked metric

**Impact:** Operators cannot size the event cap correctly; SECURITY.md promises a metric that doesn't exist.

**KPI:** Advances `kpi-regression-rate` (closing a documented metrics gap)

---

#### 5. docs-01 — Align FOUNDRY_TASK_TIMEOUT default in SECURITY.md
- **Area:** docs
- **Effort:** size-s
- **Confidence:** HIGH
- **Score:** 82/100

**Problem:** SECURITY.md:62 says `FOUNDRY_TASK_TIMEOUT` has "default 300" but `.env.example:11` says `FOUNDRY_TASK_TIMEOUT=600` and the code uses `float(os.environ.get("FOUNDRY_TASK_TIMEOUT", "600"))`. The documented default is wrong.

**Evidence:**
- `docs/SECURITY.md:62` — "default 300"
- `.env.example:11` — `FOUNDRY_TASK_TIMEOUT=600`
- `src/foundry_x/execution/runner.py:456` — `source.get("FOUNDRY_TASK_TIMEOUT", "")` with 600 fallback

**Impact:** Operators following SECURITY.md documentation get a different timeout than the actual default.

**KPI:** Advances `kpi-cycle-time` (correct documentation reduces operator confusion)

---

### Tier 2: High Confidence, Medium Impact (4 proposals)

#### 6. OBS-003 — Track `model_retry` events in KPIs and session card
- **Area:** observability
- **Effort:** size-m
- **Confidence:** HIGH
- **Score:** 78/100

**Problem:** Runner emits `model_retry` (runner.py:1449) on API retry, but no observability surface counts these events. A rising retry rate signals API flakiness or rate limiting.

**Evidence:**
- `src/foundry_x/execution/runner.py:1449` — `kind="model_retry"`
- `src/foundry_x/observability/kpis.py` — zero hits for `model_retry`
- `src/foundry_x/observability/session_card.py` — no retry count

**Impact:** Operators cannot detect API reliability degradation from KPIs.

**Recommendation:** Consider merging with OBS-004 into a single "Add missing event-type KPIs" proposal.

---

#### 7. OBS-004 — Track `tool_argument_parse_error` events in KPIs and session card
- **Area:** observability
- **Effort:** size-m
- **Confidence:** HIGH
- **Score:** 76/100

**Problem:** Runner emits `tool_argument_parse_error` (runner.py:1684) when the model produces malformed tool call arguments. No observability surface counts these events.

**Evidence:**
- `src/foundry_x/execution/runner.py:1684` — `kind="tool_argument_parse_error"`
- `src/foundry_x/observability/kpis.py` — zero hits

**Impact:** Operators cannot detect model output quality trends.

---

#### 8. bench-A-01 — read_multiple_files skill benchmark
- **Area:** benchmarks
- **Effort:** size-m
- **Confidence:** HIGH
- **Score:** 74/100

**Problem:** No benchmark exists for the `read_multiple_files` skill. Issue #617 explicitly scoped out this benchmark when adding `read_file` (singular) benchmark via PR #688.

**Evidence:**
- `benchmarks/tasks/` — no file contains `read_multiple_files`
- Issue #617 body states "Out of Scope: read_multiple_files benchmark"

**Impact:** The multi-file read skill surface is untested in the benchmark suite.

---

#### 9. bench-R-01 — rate_limit hook benchmark
- **Area:** benchmarks
- **Effort:** size-m
- **Confidence:** HIGH
- **Score:** 72/100

**Problem:** No benchmark exists for the `rate_limit` hook. The hook was seeded via PR #216, and `ContextPruningHook` has benchmarks (#782), but `rate_limit` does not.

**Evidence:**
- `benchmarks/tasks/` — no file contains `rate_limit` as a hook benchmark (only `EvolverGuardError` references)
- `harness/hooks/rate_limit.py` — exists, no corresponding benchmark

**Impact:** Rate limiting behavior is not regression-tested in the benchmark suite.

---

### Tier 3: Medium Confidence, Medium Impact (3 proposals)

#### 10. docs-02 — Document hook registry degradation mode
- **Area:** docs
- **Effort:** size-s
- **Confidence:** MEDIUM
- **Score:** 68/100

**Problem:** When `harness.hooks.get_registry()` fails, all hooks are silently disabled including the injection firewall. KPIs track this (#585), the Digester should classify it (OBS-002), but no documentation explains the degradation mode to operators.

**Evidence:**
- `src/foundry_x/execution/runner.py:661-675` — hook registry error emission
- No `docs/` file documents what happens when hooks fail to load

**Impact:** Operators don't know that hook failures disable security controls.

---

#### 11. docs-03 — Clarify evolution CLI --async vs --no-verify flags
- **Area:** docs
- **Effort:** size-s
- **Confidence:** MEDIUM
- **Score:** 65/100

**Problem:** The evolution CLI has `--async` and `--no-verify` flags with overlapping/confusing semantics. No documentation clarifies when to use each.

**Evidence:**
- `src/foundry_x/evolution/cli.py` — CLI definitions exist
- No docs explain the difference

**Impact:** Operators may use the wrong flag, bypassing the Critic gate unintentionally.

**Note:** Validator flagged that the actual flag names need verification before documentation.

---

#### 12. OBS-006 — Add windowed trend analysis to tool latency report
- **Area:** observability
- **Effort:** size-l
- **Confidence:** MEDIUM
- **Score:** 62/100

**Problem:** `tool_latency.py` computes aggregate p50/p95/p99 percentiles but does not support per-window trend analysis. Operators cannot tell if latency is improving or degrading over time.

**Evidence:**
- `src/foundry_x/observability/tool_latency.py` — `aggregate_tool_latency()` does single aggregate
- No trend view exists

**Impact:** Latency regressions are invisible until they become severe.

**Note:** Flagged as potentially premature — needs evidence that flat averages are insufficient. Related to issue #181.

---

## Blocked on Process (1 proposal)

| ID | Title | Reason |
|----|-------|--------|
| harness-01 | Declare `TokenAwarePruningHook._phase = 4` | **ADR-0004:** Harness edits must go through Evolver→Critic loop. File a `ProposedEdit` instead of an issue. Two human approvals required. |

---

## Merge Recommendations

| Merge Group | Proposals | Rationale |
|-------------|-----------|-----------|
| Digester FAILURE_KINDS | OBS-001 + OBS-002 | Both add failure kinds to the same frozenset in digester.py:61-69 |
| Missing event-type KPIs | OBS-003 + OBS-004 | Both add `query_events(kind=...)` calls to kpis.py and session_card.py |

If merged, the portfolio reduces from 12 to **10 unique issues**.

---

## KPI Alignment Summary

| KPI | Proposals advancing it |
|-----|----------------------|
| `kpi-regression-rate` | OBS-001, OBS-002, OBS-005, infra-01 |
| `kpi-cycle-time` | docs-01, docs-02, docs-03 |
| `kpi-improvement-rate` | OBS-003, OBS-004, OBS-006, bench-A-01, bench-R-01 |

---

## Copy-Paste-Ready Issue Templates

### Issue 1: Extend Digester FAILURE_KINDS (OBS-001 + OBS-002)

```
Title: trace(digester): add model_error and hook_registry_error to FAILURE_KINDS

## Problem
The Digester's `FAILURE_KINDS` frozenset (digester.py:61-69) does not include
`model_error` or `hook_registry_error`. This means:
- Sessions where `model_error` precedes other failures are misclassified
- Sessions with disabled security hooks (hook_registry_error) pass without
  security flagging

## Evidence
- runner.py:1592 emits `kind="model_error"` — not in FAILURE_KINDS
- runner.py:675 emits `kind="hook_registry_error"` — not in FAILURE_KINDS
- digester.py:211 checks `if event.kind in FAILURE_KINDS:` — silently skips both
- CONTEXT.md documents `model_error` as a failure signal
- SECURITY.md requires hook_registry_error to be observable

## Fix
1. Add `"model_error"` and `"hook_registry_error"` to `FAILURE_KINDS` in digester.py:61-69
2. Extend `FAILURE_PAYLOAD_KEYS` to include `"error_type"` and `"message"`
   (the payload keys emitted by both event types)
3. Add tests verifying both kinds are classified as first-failure markers

## Acceptance
- A session with `model_error` at step N classifies it as first failure
- A session with `hook_registry_error` flags security degradation
- `uv run pytest tests/evolution/` passes

## Labels: area-observability, size-s, phase-3
```

### Issue 2: Extend uv-pin test allowlist (infra-01)

```
Title: fix(infra): extend uv-pin test allowlist to all workflows

## Problem
tests/infra/test_workflow_uv_pin.py:32 hard-codes only 4 workflows in
EXPECTED_WORKFLOWS, but 9 distinct workflow files use the hash-verified
install-uv composite action. 5 workflows are unprotected by the regression
test — any could silently revert to `pip install --upgrade uv`.

## Evidence
- tests/infra/test_workflow_uv_pin.py:32 — 4 workflows in allowlist
- .github/workflows/*.yml — 13 install-uv references across 9 files
- docs/SECURITY.md threat #3 — supply-chain pinning required
- docs/adr/0002 — ADR mandates uv pin in CI

## Fix
1. Replace hard-coded EXPECTED_WORKFLOWS with glob-driven discovery
2. Add superset assertion so accidental removal triggers loud failure
3. Update module docstring

## Acceptance
- `uv run pytest tests/infra/test_workflow_uv_pin.py -v` covers all 9+ workflows
- Adding `pip install uv` to any workflow fails the test

## Labels: area-infra, size-s, phase-3
```

### Issue 3: Track event_limit aborts (OBS-005)

```
Title: feat(kpis): track event_limit aborts as separate KPI

## Problem
SECURITY.md documents `event_limit_hit` as a tracked metric, but kpis.py
does not implement it. Event-limit aborts are indistinguishable from other
failure modes in KPI output.

## Evidence
- runner.py:103,419,420 — task_aborted(reason="event_limit") emitted
- SECURITY.md — documents event_limit_hit as tracked metric
- kpis.py — zero hits for event_limit

## Fix
1. Add `_event_limit_abort_count()` query to kpis.py
2. Add `event_limit_abort_count` field to KpiSummary
3. Surface in session card and CLI output

## Labels: area-observability, size-s, phase-3
```

### Issue 4: Align FOUNDRY_TASK_TIMEOUT docs (docs-01)

```
Title: docs(security): align FOUNDRY_TASK_TIMEOUT default with code

## Problem
SECURITY.md:62 says default is 300, but .env.example:11 and runner.py
both use 600. The documented default is wrong.

## Evidence
- docs/SECURITY.md:62 — "default 300"
- .env.example:11 — FOUNDRY_TASK_TIMEOUT=600
- runner.py:456 — fallback is 600

## Fix
Update SECURITY.md:62 to say "default 600".

## Labels: area-docs, size-xs, phase-3
```

### Issue 5: Track model_retry in KPIs (OBS-003)

```
Title: feat(kpis): track model_retry events for API reliability visibility

## Problem
Runner emits model_retry (runner.py:1449) on API retry, but no
observability surface counts these events. Rising retry rates signal
API flakiness.

## Evidence
- runner.py:1449 — kind="model_retry"
- kpis.py — zero hits for model_retry
- session_card.py — no retry count

## Fix
1. Add _model_retry_count() query to kpis.py
2. Add model_retry_count to KpiSummary
3. Surface in session card

## Labels: area-observability, size-m, phase-3
```

### Issue 6: Track tool_argument_parse_error in KPIs (OBS-004)

```
Title: feat(kpis): track tool_argument_parse_error for model quality visibility

## Problem
Runner emits tool_argument_parse_error (runner.py:1684) but no
observability surface counts these events. Rising rates signal model
output quality degradation.

## Evidence
- runner.py:1684 — kind="tool_argument_parse_error"
- kpis.py — zero hits

## Fix
1. Add _tool_argument_parse_error_count() query to kpis.py
2. Add count to KpiSummary and session card

## Labels: area-observability, size-m, phase-3
```

### Issue 7: read_multiple_files benchmark (bench-A-01)

```
Title: bench: add read_multiple_files skill benchmark

## Problem
No benchmark exists for the read_multiple_files skill. Issue #617
explicitly scoped out this benchmark when adding read_file (singular).

## Evidence
- benchmarks/tasks/ — no file contains read_multiple_files
- Issue #617 body — "Out of Scope: read_multiple_files benchmark"

## Fix
Create benchmarks/tasks/test_read_multiple_files_skill.py following
the pattern established by test_read_file_skill.py.

## Labels: area-benchmarks, size-m, phase-3
```

### Issue 8: rate_limit hook benchmark (bench-R-01)

```
Title: bench: add rate_limit hook benchmark

## Problem
No benchmark exists for the rate_limit hook. The hook was seeded via
PR #216 but has no regression test in the benchmark suite.

## Evidence
- benchmarks/tasks/ — no rate_limit hook benchmark
- harness/hooks/rate_limit.py — exists, untested in benchmarks

## Fix
Create benchmarks/tasks/test_rate_limit_hook_benchmark.py following
the pattern established by test_context_pruning_benchmark.py.

## Labels: area-benchmarks, size-m, phase-3
```

### Issue 9: Document hook registry degradation (docs-02)

```
Title: docs: document hook registry failure degradation mode

## Problem
When harness.hooks.get_registry() fails, all hooks are silently
disabled including the injection firewall. No documentation explains
this degradation mode to operators.

## Evidence
- runner.py:661-675 — hook registry error emission
- No docs/ file explains the degradation behavior

## Fix
Add a section to docs/OPERATOR.md or docs/ARCHITECTURE.md explaining:
- What happens when hooks fail to load
- Which security controls are affected
- How to detect and recover

## Labels: area-docs, size-s, phase-3
```

### Issue 10: Clarify evolution CLI flags (docs-03)

```
Title: docs: clarify evolution CLI --async vs --no-verify semantics

## Problem
The evolution CLI has --async and --no-verify flags with confusing
overlap. No documentation clarifies when to use each.

## Fix
Document the exact behavior of each flag, when to use each, and
the security implications of --no-verify (bypasses Critic gate).

## Labels: area-docs, size-s, phase-3
```

### Issue 11: Windowed tool latency trends (OBS-006)

```
Title: feat(tool-latency): add windowed trend analysis

## Problem
tool_latency.py computes aggregate percentiles but does not support
per-window trend analysis. Operators cannot detect latency regressions
over time.

## Evidence
- observability/tool_latency.py — aggregate_tool_latency() only
- Related to issue #181

## Fix
Add windowed aggregation (e.g., last 10 sessions, last 24h) with
trend indicators (improving/stable/degrading).

## Labels: area-observability, size-l, phase-3
```

---

## Process Note

The following proposal was identified by the harness scout but cannot be filed as an issue:

**harness-01: Declare `TokenAwarePruningHook._phase = 4`**

Per ADR-0004 and AGENTS.md §2, harness/ files cannot be hand-edited. This must be routed as a `ProposedEdit` through the Evolver→Critic pipeline. The `_phase` attribute controls execution ordering within the hook registry — its absence means the hook's execution order is undefined.

---

## Adjacent Findings (not proposed as issues)

| Finding | Owner | Notes |
|---------|-------|-------|
| No `.github/workflows/*.yml` declares a `permissions:` block | security | Hardening proposal — needs measured failure before filing |
| `infra/llama-cpp/README.md` ROCm section could cross-reference ADR-0021 | docs | Nice-to-have, not evidence-backed gap |
| Shellcheck not in pre-commit hooks (issue #350, #438 exist) | infra | Separate concern, tracked in existing issues |
