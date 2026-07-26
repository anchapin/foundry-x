# Parallel Issue Generation Results

**Session:** `parallel-issue-generation`
**Generated:** 2026-07-18
**Sub-agents:** 6 (Evolution Engine, Trace Store, Harness Layer, Runner/Model Adapter, Phase 4 Discovery, Developer Experience)

---

## Merged Issue List (sorted: P0 → P1 → kpi-improvement-rate weight → alphabetical)

### P0 (blocks core loop)

| # | Title | Source | KPI Tags | Confidence |
|---|-------|--------|----------|------------|
| 1 | `fix(loop): pass failure_class to critic.evaluate in run_evolution_step_async` | Evolution Engine | `kpi-improvement-rate`, `kpi-cycle-time` | 0.95 |
| 2 | `fix(evolver): context-overflow template produces invalid manifest JSON` | Harness Layer | `kpi-token-budget`, `kpi-improvement-rate` | 0.90 |
| 3 | `fix(critic): implement gate_timeout_s as documented in evaluate() docstring` | Evolution Engine | `kpi-cycle-time` | 0.95 |

---

### P1 (improves KPI measurably)

| # | Title | Source | KPI Tags | Confidence |
|---|-------|--------|----------|------------|
| 4 | `fix(evolver): context-overflow template produces valid manifest.json JSON edits` | Evolution Engine | `kpi-improvement-rate` | 0.90 |
| 5 | `fix(trace): event_limit early-return bypasses outcome event emission` | Trace Store | `kpi-cycle-time`, `kpi-improvement-rate` | 0.92 |
| 6 | `fix(harness): remove stale example_skill from manifest skill_inventory` | Harness Layer | `kpi-improvement-rate`, `kpi-regression-rate` | 0.90 |
| 7 | `perf(trace): SQLite prune doesn't reclaim WAL space causing unbounded growth` | Trace Store | — | 0.88 |
| 8 | `docs(evolver): add context-overflow to evolver_meta_prompt.txt failure taxonomy` | Evolution Engine | `kpi-improvement-rate` | 0.90 |
| 9 | `fix(runner): emit one tool_call event per skill execution` | Runner/Adapter | `kpi-improvement-rate` | 0.95 |
| 10 | `feat(kpis): add per-skill and per-task-family improvement_rate and regression_rate slices` | Phase 4 | `kpi-regression-rate`, `kpi-cycle-time`, `kpi-improvement-rate` | 0.85 |
| 11 | `feat(harness): add context-pruning and token-budget guidance to system_prompt` | Harness Layer | `kpi-token-budget`, `kpi-improvement-rate` | 0.85 |
| 12 | `fix(observability): cycle_time KPI excludes sessions that fail before Critic runs` | Trace Store | `kpi-cycle-time` | 0.78 |
| 13 | `feat(benchmark): validate internal suite against external coding evals (HumanEval+/SWE-bench)` | Phase 4 | `kpi-improvement-rate`, `kpi-regression-rate` | 0.75 |
| 14 | `feat(infra): add automated llama-server health-check and restart to foundry-runner` | Phase 4 | `kpi-cycle-time` | 0.80 |
| 15 | `feat(harness): add pre-tool argument validation hook slot` | Harness Layer | `kpi-improvement-rate`, `kpi-token-budget` | 0.75 |
| 16 | `docs: Add a getting-started tutorial for new contributors` | DX | `kpi-cycle-time` | 0.90 |

---

### P2 (nice-to-have)

| # | Title | Source | KPI Tags | Confidence |
|---|-------|--------|----------|------------|
| 17 | `feat(observability): session-summary and session-card lack --latest flag` | Trace Store | `kpi-cycle-time` | 0.85 |
| 18 | `perf(runner): record post-tool hook latency in tool_call event` | Runner/Adapter | `kpi-improvement-rate` | 0.90 |
| 19 | `feat(evolver): add cross-task-family pattern transfer to the evolver meta-prompt` | Phase 4 | `kpi-improvement-rate`, `kpi-cycle-time` | 0.65 |
| 20 | `fix(runner): ttft_ms=None is ambiguous for tool-call-only streaming responses` | Runner/Adapter | `kpi-improvement-rate` | 0.75 |
| 21 | `docs: Add ADR for workspace-root and agent filesystem boundary concept` | DX | `kpi-cycle-time` | 0.75 |
| 22 | `cli: Improve error messages for common failure paths to guide recovery` | DX | `kpi-cycle-time` | 0.70 |
| 23 | `feat(runner): add session checkpoint and resumption via trace replay` | Runner/Adapter | `kpi-cycle-time`, `kpi-improvement-rate` | 0.60 |

---

### P3 (exploratory)

| # | Title | Source | KPI Tags | Confidence |
|---|-------|--------|----------|------------|
| 24 | `chore: Add spell-check and markdownlint to pre-commit hooks` | DX | `kpi-cycle-time` | 0.85 |

---

## Full Issue Details

### P0 Issues

---

#### 1. fix(loop): pass failure_class to critic.evaluate in run_evolution_step_async

**Priority:** P0
**KPI Tags:** `kpi-improvement-rate`, `kpi-cycle-time`
**ADR Tags:** `ADR-0006`
**Confidence:** 0.95
**Source:** Sub-agent 1 — Evolution Engine

**Body:**
```
## Motivation
run_evolution_step_async (line 247) calls critic.evaluate(edit.unified_diff, edit_index=idx) without the failure_class keyword argument, while the sync variant run_evolution_step (line 166) correctly passes failure_class=failure_report.proposed_class. The failure_class field is critical for failure_class_distribution KPI computation in kpis.py:_failure_class_distribution(); without it the per-class counts in KpiSummary are silently wrong for every async evolution run.

## Evidence
src/foundry_x/evolution/loop.py:247 — verdict = critic.evaluate(edit.unified_diff, edit_index=idx) (missing failure_class); loop.py:166 — verdict = critic.evaluate(edit.unified_diff, edit_index=idx, failure_class=failure_report.proposed_class) (correct).

## Risk
Low. This is a one-line fix adding the missing keyword argument. No behaviour changes except the correct attribution of failure classes to KPI records.

## Acceptance Criteria
1. run_evolution_step_async passes failure_class=failure_report.proposed_class to critic.evaluate().
2. A test (or existing test) verifies that async evolution results have verdict.failure_class set correctly.

## ADR(s)
ADR-0006 (pydantic at module boundary)
```

---

#### 2. fix(evolver): context-overflow template produces invalid manifest JSON

**Priority:** P0
**KPI Tags:** `kpi-token-budget`, `kpi-improvement-rate`
**ADR Tags:** `ADR-0004`, `ADR-0012`
**Confidence:** 0.90
**Source:** Sub-agent 3 — Harness Layer

**Body:**
```
## Motivation
The _PROPOSED_CLASS_EDIT_TEMPLATES['context-overflow'] (evolver.py:230-236) appends literal text lines '"context_pruning": {"token_threshold": 6144, "event_threshold": 200}' to manifest.json. Since manifest.json is a JSON file ending with }, appending these lines produces syntactically invalid JSON. The Critic's load_check gate (evaluate() step 5) runs harness/scripts/load_check.py which will fail on malformed JSON, causing the context-overflow template edit to always be rejected — a false-negative at the Critic gate.

## Evidence
src/foundry_x/evolution/evolver.py:230-236 — template appends raw string lines to manifest.json; src/foundry_x/evolution/critic.py:665-686 — load_check gate runs after git apply and rejects malformed harness.

## Risk
Low. The fix is to use a JSON-aware patching approach: read the manifest as JSON, patch the context_pruning key, serialise back, and produce a minimal unified diff.

## Acceptance Criteria
1. context-overflow template edits to manifest.json produce syntactically valid JSON that passes json.loads.
2. The load_check gate in Critic.evaluate() passes for a context-overflow template edit.
3. test_evolver_guard.py (or a new test) validates the context-overflow template produces a valid manifest diff.

## ADR(s)
ADR-0012 (manifest.json as evolver target), ADR-0004 (Critic gate)
```

---

#### 3. fix(critic): implement gate_timeout_s as documented in evaluate() docstring

**Priority:** P0
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** `ADR-0004`, `ADR-0010`
**Confidence:** 0.95
**Source:** Sub-agent 1 — Evolution Engine

**Body:**
```
## Motivation
The Critic.evaluate() docstring (issue #188) states that every subprocess call is bounded by self.gate_timeout_s. In practice the __init__ never accepts this parameter and subprocess.run/subprocess.Popen calls pass no timeout argument. A hanging pytest or load_check subprocess will block run_evolution_step indefinitely, inflating kpi-cycle-time to infinity.

## Evidence
src/foundry_x/evolution/critic.py:568, 601 — docstring references self.gate_timeout_s; src/foundry_x/evolution/critic.py:384-387 — subprocess.run with no timeout= argument; src/foundry_x/evolution/critic.py:207-235 — __init__ signature does not accept gate_timeout_s.

## Risk
Low. Adding a timeout to an existing call that has none is a straightforward safety net.

## Acceptance Criteria
1. Critic.__init__ accepts gate_timeout_s: float | None = None.
2. All subprocess.run calls inside evaluate() pass timeout=gate_timeout_s.
3. On TimeoutExpired the verdict is verdict=False with failed_checks=['pytest:timeout'] or ['load_check:timeout'].
4. test_critic.py has a test exercising the timeout path.

## ADR(s)
ADR-0004 (Critic gate), ADR-0010 (Runner agent loop termination)
```

---

### P1 Issues

---

#### 4. fix(evolver): context-overflow template produces valid manifest.json JSON edits

**Priority:** P1
**KPI Tags:** `kpi-improvement-rate`
**ADR Tags:** `ADR-0012`
**Confidence:** 0.90
**Source:** Sub-agent 1 — Evolution Engine

**Body:**
```
## Motivation
_PROPOSED_CLASS_EDIT_TEMPLATES['context-overflow'] (evolver.py:230-236) appends literal text lines to manifest.json. Since manifest.json is a JSON file ending with }, appending these lines produces syntactically invalid JSON. The Critic's load_check gate fails on malformed JSON, causing the context-overflow template edit to always be rejected.

## Evidence
src/foundry_x/evolution/evolver.py:230-236; src/foundry_x/evolution/critic.py:665-686

## Risk
Low.

## Acceptance Criteria
1. context-overflow template edits produce syntactically valid JSON that passes json.loads.
2. The load_check gate passes for a context-overflow template edit.
3. A test validates the template produces a valid manifest diff.

## ADR(s)
ADR-0012
```

---

#### 5. fix(trace): event_limit early-return bypasses outcome event emission

**Priority:** P1
**KPI Tags:** `kpi-cycle-time`, `kpi-improvement-rate`
**ADR Tags:** `ADR-0010`, `ADR-0007`
**Confidence:** 0.92
**Source:** Sub-agent 2 — Trace Store

**Body:**
```
## Motivation
When the per-session event cap (FOUNDRY_MAX_EVENTS_PER_SESSION) is exceeded at runner.py:1548 — before the agent loop starts — the function returns immediately after recording task_aborted(reason="event_limit"). The finally block that emits outcome is never reached. By contrast, when the cap fires inside the loop, the break triggers the finally block and both task_aborted and outcome are recorded. This creates two trace shapes for the same abort reason, confusing the Digester.

## Evidence
runner.py:1548-1552 — early return skips finally block; runner.py:1782 where finally always fires after loop breaks.

## Risk
Low risk fix.

## Acceptance Criteria
1. A session that hits task_aborted(reason="event_limit") at any point emits exactly one outcome event.
2. The Digester processes the session without duplicate-terminal-event confusion.
3. A regression test plants two sessions: one hitting the limit before the loop, one hitting it inside, and asserts both have exactly one outcome event.

## ADR(s)
ADR-0010, ADR-0007
```

---

#### 6. fix(harness): remove stale example_skill from manifest skill_inventory

**Priority:** P1
**KPI Tags:** `kpi-improvement-rate`, `kpi-regression-rate`
**ADR Tags:** `ADR-0004`, `ADR-0012`
**Confidence:** 0.90
**Source:** Sub-agent 3 — Harness Layer

**Body:**
```
## Motivation
The skill_inventory array in harness/manifest.json lists example_skill as a declared skill alongside production tools. However, harness/skills/example_skill.json explicitly self-describes as "Not for production use." Having a non-production stub in the active production inventory misleads the Evolver and creates risk that the Critic benchmark suite exercises a stub skill rather than a real one.

## Evidence
harness/manifest.json lines 35-36: example_skill in skill_inventory; harness/skills/example_skill.json description: "Not for production use"

## Risk
Low.

## Acceptance Criteria
1. example_skill is removed from the skills array in manifest.json
2. example_skill.json is removed from harness/skills/
3. The Critic benchmark suite still passes.

## ADR(s)
ADR-0004, ADR-0012
```

---

#### 7. perf(trace): SQLite prune doesn't reclaim WAL space causing unbounded growth

**Priority:** P1
**KPI Tags:** —
**ADR Tags:** `ADR-0013`, `ADR-0022`
**Confidence:** 0.88
**Source:** Sub-agent 2 — Trace Store

**Body:**
```
## Motivation
TraceLogger.prune_sessions (logger.py:955) issues batch DELETE statements against the SQLite backend but never calls VACUUM. SQLite's WAL mode accumulates deleted pages in the -wal and -shm sidecar files indefinitely. With heavy pruning, the WAL can grow to multiple times the size of the actual data pages.

## Evidence
logger.py:969-980 (_prune_sqlite): DELETE without VACUUM

## Risk
Medium. VACUUM requires exclusive access and can take time on large DBs. Should be optional or offloaded to a background thread.

## Acceptance Criteria
1. foundry-trace prune (SQLite backend) accepts a --vacuum flag that runs VACUUM after deleting sessions.
2. Without --vacuum, the command completes with no extra latency.
3. A benchmark test creates 1000 sessions, prunes 900 with --vacuum, and asserts WAL file size is < 2x the main db file size.

## ADR(s)
ADR-0013, ADR-0022
```

---

#### 8. docs(evolver): add context-overflow to evolver_meta_prompt.txt failure taxonomy

**Priority:** P1
**KPI Tags:** `kpi-improvement-rate`
**ADR Tags:** `ADR-0011`, `ADR-0018`
**Confidence:** 0.90
**Source:** Sub-agent 1 — Evolution Engine

**Body:**
```
## Motivation
The evolver_meta_prompt.txt (ADR-0018 §Failure Taxonomy) documents only 5 failure classes but the actual taxonomy (ADR-0011) has 6 classes including context-overflow. The LLM Evolver reads this file to guide its edit strategy. When a context-overflow failure is processed, the meta-prompt offers no class-specific guidance.

## Evidence
harness/evolver_meta_prompt.txt:43-51 — Five-class table; docs/adr/0011-failure-report-class-taxonomy.md — Six classes including context-overflow; src/foundry_x/evolution/digester.py:59 — CONTEXT_OVERFLOW_CLASS = 'context-overflow'

## Risk
Minimal. Documentation-only change.

## Acceptance Criteria
1. evolver_meta_prompt.txt §Failure Taxonomy includes context-overflow with root cause guidance and edit strategy.
2. The table is updated to six rows.
3. The Class-Specific Guidance section includes a context-overflow entry.

## ADR(s)
ADR-0011, ADR-0018
```

---

#### 9. fix(runner): emit one tool_call event per skill execution

**Priority:** P1
**KPI Tags:** `kpi-improvement-rate`
**ADR Tags:** `ADR-0010`
**Confidence:** 0.95
**Source:** Sub-agent 4 — Runner/Model Adapter

**Body:**
```
## Motivation
In run_task (runner.py lines 1707–1718 and 1733–1744), every skill execution produces two tool_call trace events with the same call_id and step: one with duration_ms: 0 before _execute_skill, and one with duration_ms: <actual> after. Both are recorded via _record_and_count. A consumer deduplicating by call_id will randomly keep either the zero-duration record or the real one — corrupting per-call latency histograms.

## Evidence
runner.py:1707–1718 (first emission, duration_ms=0); runner.py:1733–1744 (second emission, real duration)

## Risk
Low.

## Acceptance Criteria
1. For each tool the model emits, run_task records exactly one tool_call event with the actual duration_ms and hook_overhead_ms.
2. The existing test test_run_task_records_hook_overhead_ms_with_delayed_hook continues to pass.
3. A new test exercises the tool-call event sequence and asserts event.kind == 'tool_call' appears exactly once per tool call per step.

## ADR(s)
ADR-0010
```

---

#### 10. feat(kpis): add per-skill and per-task-family improvement_rate and regression_rate slices

**Priority:** P1
**KPI Tags:** `kpi-regression-rate`, `kpi-cycle-time`, `kpi-improvement-rate`
**ADR Tags:** `ADR-0006`
**Confidence:** 0.85
**Source:** Sub-agent 5 — Phase 4 Discovery

**Body:**
```
## Motivation
kpi-improvement-rate and kpi-regression-rate are currently aggregate fractions across all critic_verdict events. When a regression is flagged, the operator must manually inspect which benchmark task failed to determine which skill is at fault. With 40+ benchmark tasks and 8 skills, diagnosing a regression requires nontrivial forensic work.

## Evidence
kpis.py:448–476 (_verdict_rates): computes aggregate rates with no skill or task-family dimension; BenchmarkTask (benchmarks/models.py): each task declares tags and difficulty_tier.

## Risk
Low. Purely additive to the KPI layer.

## Acceptance Criteria
1. compute_kpis accepts a group_by parameter ('skill', 'task_family', 'difficulty_tier', None for aggregate).
2. When group_by='skill', KpiSummary includes per_skill: dict[str, SkillKpiSlice].
3. SkillKpiSlice is a pydantic model at the module boundary.
4. foundry-kpis CLI gains --group-by flag.
5. compare_kpis supports per-slice delta computation.

## ADR(s)
ADR-0006
```

---

#### 11. feat(harness): add context-pruning and token-budget guidance to system_prompt

**Priority:** P1
**KPI Tags:** `kpi-token-budget`, `kpi-improvement-rate`
**ADR Tags:** `ADR-0004`
**Confidence:** 0.85
**Source:** Sub-agent 3 — Harness Layer

**Body:**
```
## Motivation
The FoundryAgent receives context_pruned trace events when the pruning hook drops historical events to stay under the token/event budget, but the current system prompt gives the agent zero guidance on how to interpret this signal or adapt its behavior. An agent that keeps producing verbose intermediate steps is unaware that context is being shed.

## Evidence
harness/system_prompt.txt: no mention of context_pruned, pruning, token budgets, or truncation; src/foundry_x/observability/kpis.py: token_budget_hit_rate is a tracked KPI.

## Risk
Adding prompt guidance is low-risk.

## Acceptance Criteria
1. The Evolver proposes a system_prompt edit that adds operating rules for context_pruned events.
2. The Evolver proposes guidance for task_aborted(reason="token_budget") — simplify outputs, avoid redundant tool calls.
3. The edit passes the Critic gate.
4. kpi-token-budget metric shows improvement in subsequent evolutions.

## ADR(s)
ADR-0004 (Evolver must generate), PHILOSOPHY.md §6
```

---

#### 12. fix(observability): cycle_time KPI excludes sessions that fail before Critic runs

**Priority:** P1
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** `ADR-0010`
**Confidence:** 0.78
**Source:** Sub-agent 2 — Trace Store

**Body:**
```
## Motivation
_cycle_time (kpis.py:394) computes mean wall-clock time from task_received to critic_verdict only for sessions that have both. Sessions that fail before producing a critic_verdict are silently excluded. This creates survivorship bias: the measured cycle time is only the cycle time of sessions that completed the full loop.

## Evidence
kpis.py:414-427: sessions without critic_verdict are excluded with no auxiliary tracking.

## Risk
Low. The fix is additive.

## Acceptance Criteria
1. KpiSummary gains an excluded_from_cycle_time field (int).
2. foundry-kpis renders the exclusion count when > 0.
3. _render_history_markdown includes the exclusion count.
4. compare_kpis includes the exclusion count delta.
5. A test plants one session without critic_verdict and asserts the exclusion count is 1.

## ADR(s)
ADR-0010
```

---

#### 13. feat(benchmark): validate internal suite against external coding evals (HumanEval+/SWE-bench)

**Priority:** P1
**KPI Tags:** `kpi-improvement-rate`, `kpi-regression-rate`
**ADR Tags:** `ADR-0005`
**Confidence:** 0.75
**Source:** Sub-agent 5 — Phase 4 Discovery

**Body:**
```
## Motivation
The benchmark suite under benchmarks/tasks/ (40+ tasks) is the sole proxy for kpi-improvement-rate and kpi-regression-rate. ADR-0005 deferred adoption of external standardized benchmarks until a "concrete limitation forced an ADR." That limitation now exists: we cannot answer whether improving improvement_rate on our internal suite translates to real coding tasks.

## Evidence
PRD §5 and ADR-0005 both acknowledge kpi-improvement-rate is measured against benchmarks/tasks/. No study has correlated internal pass rates with external coding benchmarks.

## Risk
Low. Read-only validation study.

## Acceptance Criteria
1. Run the benchmark suite against HumanEval+ (or a 20-task slice) under fx-runner harness.
2. Compute Pearson correlation between internal and external pass/fail across ≥30 task instances.
3. If correlation ≥ 0.7: document in ADR-00XX and confirm suite is valid proxy.
4. If correlation < 0.7: document gap, identify uncorrelated task families, recommend external framework adoption.

## ADR(s)
ADR-0005
```

---

#### 14. feat(infra): add automated llama-server health-check and restart to foundry-runner

**Priority:** P1
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** —
**Confidence:** 0.80
**Source:** Sub-agent 5 — Phase 4 Discovery

**Body:**
```
## Motivation
infra/llama-cpp/README.md shows a manual llama-server launch workflow. If llama-server crashes mid-benchmark, the operator must manually detect the failure, restart the server, and re-run — adding friction to every fx-runner cycle. This directly impacts kpi-cycle-time.

## Evidence
MODEL_CONFIG.md §4: llama-server launched manually; infra/llama-cpp/README.md: smoke test is one-shot; ADR-0020 §4 (FOUNDRY_TASK_TIMEOUT=600 s) implies long-running sessions vulnerable to mid-session server failure.

## Risk
Low. Pure foundry infrastructure, no harness changes.

## Acceptance Criteria
1. A FoundryServerManager class provides start(), stop(), is_healthy(), restart() with exponential-backoff retry.
2. FoundryServerManager is configurable via env vars (LLAMACPP_HOST, LLAMACPP_MODEL_PATH, FOUNDRY_SERVER_NGpuLayers, FOUNDRY_SERVER_CTXSize).
3. fx-runner calls FoundryServerManager.ensure_healthy() before each session.
4. If server becomes unhealthy mid-session, Runner records a server_unavailable trace event and calls restart().
5. foundry-kpis gains a server_restart_count auxiliary metric.

## ADR(s)
None
```

---

#### 15. feat(harness): add pre-tool argument validation hook slot

**Priority:** P1
**KPI Tags:** `kpi-improvement-rate`, `kpi-token-budget`
**ADR Tags:** `ADR-0004`, `ADR-0009`
**Confidence:** 0.75
**Source:** Sub-agent 3 — Harness Layer

**Body:**
```
## Motivation
The current hook stack operates on the boundary between tool execution and the model prompt, but there is no hook that validates tool-call arguments before they reach the execution layer. When the model produces malformed JSON for a tool call, the error surfaces in the runner's error-handling path, producing a tool-error failure classification and triggering an unnecessary evolver proposal.

## Evidence
src/foundry_x/evolution/digester.py: tool-error is the catch-all failure class; src/foundry_x/execution/runner.py: _parse_tool_arguments emits tool_argument_parse_error on malformed JSON.

## Risk
A new hook slot adds complexity.

## Acceptance Criteria
1. The Evolver proposes a new validate_arguments hook (or extends Hook protocol) that runs in pre_tool after existing pre_tool hooks.
2. The hook validates tool arguments against the skill's input_schema before execution.
3. On validation failure, the hook returns a ToolResult with error='argument_validation_failed:...'.
4. The edit passes the Critic gate including security-evals benchmarks (ADR-0009).

## ADR(s)
ADR-0004, ADR-0009
```

---

#### 16. docs: Add a getting-started tutorial for new contributors

**Priority:** P1
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** —
**Confidence:** 0.90
**Source:** Sub-agent 6 — Developer Experience

**Body:**
```
## Motivation
AGENTS.md requires new contributors to read 11 documents before writing code. This creates a high barrier to entry. There is no single guided walkthrough that takes a contributor from git clone to understanding their first trace.

## Evidence
README.md quick start only covers smoke test path; CONTRIBUTING.md covers PR workflow but not how to run a real task and interpret output; no document walks through the full observe → digest → propose → evaluate loop.

## Risk
Low.

## Acceptance Criteria
1. A new contributor can follow the tutorial end-to-end in under 30 minutes without external context.
2. The tutorial produces a trace session the contributor can inspect with foundry-x-trace show <session_id>.
3. The tutorial explains key trace events (task_received, model_request, tool_call, outcome).
4. Tutorial is linked from README.md quick-start and CONTRIBUTING.md.
5. Tutorial is tested by CI.

## ADR(s)
None
```

---

### P2 Issues

---

#### 17. feat(observability): session-summary and session-card lack --latest flag

**Priority:** P2
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** `ADR-0007`
**Confidence:** 0.85
**Source:** Sub-agent 2 — Trace Store

---

#### 18. perf(runner): record post-tool hook latency in tool_call event

**Priority:** P2
**KPI Tags:** `kpi-improvement-rate`
**ADR Tags:** `ADR-0010`
**Confidence:** 0.90
**Source:** Sub-agent 4 — Runner/Model Adapter

---

#### 19. feat(evolver): add cross-task-family pattern transfer to the evolver meta-prompt

**Priority:** P2
**KPI Tags:** `kpi-improvement-rate`, `kpi-cycle-time`
**ADR Tags:** `ADR-0018`, `ADR-0004`
**Confidence:** 0.65
**Source:** Sub-agent 5 — Phase 4 Discovery

---

#### 20. fix(runner): ttft_ms=None is ambiguous for tool-call-only streaming responses

**Priority:** P2
**KPI Tags:** `kpi-improvement-rate`
**ADR Tags:** `ADR-0010`
**Confidence:** 0.75
**Source:** Sub-agent 4 — Runner/Model Adapter

---

#### 21. docs: Add ADR for workspace-root and agent filesystem boundary concept

**Priority:** P2
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** `ADR-0010`
**Confidence:** 0.75
**Source:** Sub-agent 6 — Developer Experience

---

#### 22. cli: Improve error messages for common failure paths to guide recovery

**Priority:** P2
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** `ADR-0010`
**Confidence:** 0.70
**Source:** Sub-agent 6 — Developer Experience

---

#### 23. feat(runner): add session checkpoint and resumption via trace replay

**Priority:** P2
**KPI Tags:** `kpi-cycle-time`, `kpi-improvement-rate`
**ADR Tags:** `ADR-0007`, `ADR-0010`
**Confidence:** 0.60
**Source:** Sub-agent 4 — Runner/Model Adapter

---

### P3 Issues

---

#### 24. chore: Add spell-check and markdownlint to pre-commit hooks

**Priority:** P3
**KPI Tags:** `kpi-cycle-time`
**ADR Tags:** `ADR-0008`
**Confidence:** 0.85
**Source:** Sub-agent 6 — Developer Experience

---

## Summary Statistics

| Priority | Count |
|----------|-------|
| P0 | 3 |
| P1 | 13 |
| P2 | 7 |
| P3 | 1 |
| **Total** | **24** |

| KPI Tag | Issue Count |
|---------|-------------|
| `kpi-cycle-time` | 14 |
| `kpi-improvement-rate` | 14 |
| `kpi-regression-rate` | 5 |
| `kpi-token-budget` | 5 |

| ADR Coverage | Count |
|--------------|-------|
| ADR-0004 | 7 |
| ADR-0010 | 8 |
| ADR-0006 | 3 |
| ADR-0012 | 3 |
| ADR-0007 | 4 |
| ADR-0011 | 2 |
| ADR-0013 | 2 |
| ADR-0018 | 3 |
| ADR-0005 | 2 |
| ADR-0009 | 2 |
| ADR-0008 | 2 |
| ADR-0014 | 0 (mentioned but no issues) |
| ADR-0022 | 1 |

---

## Confidence Distribution

| Range | Count |
|-------|-------|
| ≥ 0.90 | 8 |
| 0.80–0.89 | 8 |
| 0.70–0.79 | 6 |
| < 0.70 | 2 |
