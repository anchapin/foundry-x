# KPI/QA/Security Audit Report

**Date:** 2026-07-18
**Auditor:** QA Specialist (big-pickle)
**Scope:** 20 scout proposals for GitHub Issue Discovery
**Documents referenced:** `docs/PRD.md`, `docs/SECURITY.md`, `docs/CONTEXT.md`, source code

---

## Summary

| Verdict | Count | Proposals |
|---------|-------|-----------|
| **PASS** | 6 | trace-01, execution-01, OBS-001, OBS-005, docs-01, docs-02 |
| **FLAG** | 5 | OBS-003, OBS-004, OBS-006, OBS-007, harness-01 |
| **BLOCK** | 4 | evolution-02, evolution-03, infra-01, security-01 |
| **CONDITIONAL PASS** | 5 | evolution-01, OBS-002, bench-A-01, bench-R-01, docs-03 |

---

## Detailed Findings

### trace-01: Delete shadowed `_query_events_sqlite` duplicate in logger.py — **PASS**

- **KPI alignment:** Indirectly advances `kpi-regression-rate` by removing dead code that could cause confusion during trace analysis. No direct KPI impact.
- **Security impact:** None. The second definition at line 877 shadows the first at line 839; Python silently uses the second. Deleting the first (or the duplicate) is safe.
- **Evidence:** Both definitions at `logger.py:839` and `logger.py:877` have identical signatures and body. The second definition overwrites the first. Classic Python class-body shadowing.
- **Test requirements:** Existing trace tests should cover `iter_events` with SQLite backend. Verify no test depends on the shadowed definition.
- **Benchmark impact:** None. Pure dead-code removal.
- **Verdict:** Safe, evidence-backed cleanup. Ship it.

---

### execution-01: Align post tool_call name with hook-mutated call in runner.py — **PASS**

- **KPI alignment:** Advances `kpi-regression-rate` by fixing trace inconsistency. The post-execution `tool_call` event records the pre-hook name (`tool_call.function.name`) while the execution itself uses the post-hook name (`call.name`). This makes regression analysis unreliable.
- **Security impact:** None. Fix aligns trace truth with actual behavior.
- **Evidence:**
  - Pre-execution emit (line ~1712): `"name": call.name` ✓ (post-hook)
  - Post-execution emit (line ~1739): `"name": tool_call.function.name` ✗ (pre-hook, inconsistent)
  - Execution call (line ~1729): `await _execute_skill(call.name, ...)` ✓ (post-hook)
  - The fix: change line ~1739 from `tool_call.function.name` to `call.name`.
- **Test requirements:** Add a test that registers a hook which renames the tool call and asserts both `tool_call` events record the mutated name.
- **Benchmark impact:** None. Trace-only change.
- **Verdict:** Real bug, clear fix, safe scope.

---

### evolution-01: Restore Critic subprocess timeout in critic.py — **CONDITIONAL PASS**

- **KPI alignment:** Advances `kpi-cycle-time` and `kpi-regression-rate`. Without a timeout, the Critic can hang indefinitely on stuck pytest/git subprocesses, blocking the evolution loop. Also prevents `approved=True` from being returned on a hung process that eventually gets killed externally.
- **Security impact:** Low. A missing Critic timeout violates SECURITY.md "Runaway detection" principle (threat #5: resource exhaustion). However, this is in the Critic sandbox, not the agent loop.
- **Evidence:** Docstrings at critic.py:568,601 reference `self.gate_timeout_s` and `subprocess.TimeoutExpired`, but `rg` for `subprocess.run`/`subprocess.call` and `gate_timeout` in critic.py returns zero code hits (only docstring references). The timeout is documented but not implemented.
- **Test requirements:** Unit test that spawns a Critic with a short `gate_timeout_s` and a deliberately slow subprocess; assert verdict is `approved=False` with `:timeout` suffix in `failed_checks`.
- **Benchmark impact:** None. Critic robustness improvement.
- **Verdict:** CONDITIONAL — confirm whether `gate_timeout_s` is actually unused in code or if the timeout is applied via an indirect mechanism. If truly missing, this is a security-relevant fix.

---

### evolution-02: Normalize `_run_loop_async` return arity in evolve_cli.py — **BLOCK**

- **KPI alignment:** Cannot assess — the target file does not exist.
- **Security impact:** N/A
- **Evidence:** `src/foundry_x/evolution/evolve_cli.py` does not exist (IOError). `_run_loop_async` does not exist anywhere in `src/foundry_x/`. The evolution CLI may have been refactored, renamed, or never implemented in this file path.
- **Test requirements:** N/A
- **Benchmark impact:** N/A
- **Verdict:** **BLOCK — phantom proposal.** The referenced file and function do not exist in the codebase. The scout likely hallucinated or referenced stale information. Reject and close. If the evolution CLI exists elsewhere, re-scout the actual location.

---

### evolution-03: Persist async Critic verdicts — **BLOCK**

- **KPI alignment:** Cannot assess — the feature does not appear to exist in the codebase.
- **Security impact:** N/A
- **Evidence:** `rg -rn 'async.*verdict|persist.*verdict|critic.*async' src/foundry_x/evolution/` returns zero results. No async Critic verdict persistence path exists to "restore" or "fix." The Critic subprocess runs synchronously via `run_in_executor` or similar, and verdicts are already persisted by `record_verdict` in `regression_report.py`.
- **Test requirements:** N/A
- **Benchmark impact:** N/A
- **Verdict:** **BLOCK — phantom proposal.** The referenced mechanism does not exist. Either the scout hallucinated the issue, or this was already addressed in a prior commit. Verify against the issue tracker before re-proposing.

---

### OBS-001: Add `model_error` to Digester FAILURE_KINDS — **PASS**

- **KPI alignment:** Advances `kpi-regression-rate` and `kpi-improvement-rate`. Without `model_error` in `FAILURE_KINDS`, the Digester cannot classify sessions with model errors as failures, making the evolution loop blind to model-level regressions.
- **Security impact:** Positive. Hidden model errors could mask prompt injection or adversarial inputs that cause the model to fail silently.
- **Evidence:** Runner emits `kind="model_error"` (confirmed by docstring: "On any model error the loop records a `model_error` event"). Digester `FAILURE_KINDS` at line 61 contains: `tool_error`, `task_failed`, `run_failed`, `agent_error`, `error` — but NOT `model_error`. The payload also does not match `FAILURE_PAYLOAD_KEYS` (`error`, `traceback`, `exception`) because the model_error payload uses `error_type` and `message`.
- **Test requirements:** Unit test asserting `Digester.digest()` classifies a trace containing a `model_error` event as a failure. Regression test against the Digester vocabulary contract (ADR-0004).
- **Benchmark impact:** None. Vocabulary addition to Digester, not to the agent.
- **Verdict:** Clear evidence-backed gap. Ship with vocab test.

---

### OBS-002: Add hook_registry_error to Digester FAILURE_KINDS — **CONDITIONAL PASS**

- **KPI alignment:** Advances `kpi-regression-rate`. When `hook_registry_error` fires, all hooks (including security-critical `InjectionFirewallHook`) are disabled. The Digester should classify these sessions as failures.
- **Security impact:** **HIGH.** A `hook_registry_error` disables the injection firewall. If the Digester doesn't classify this as a failure, the evolution loop could accept a harness edit that causes registry errors — silently disabling security hooks.
- **Evidence:** `hook_registry_error` IS tracked in kpis.py (`hooks_disabled_count`, `hooks_disabled_rate` via issue #585). But it is NOT in `FAILURE_KINDS`. The payload (`error_type`, `message`) does not match `FAILURE_PAYLOAD_KEYS`. So the Digester completely misses these security-critical failures.
- **Test requirements:** Unit test that `Digest` classifies `hook_registry_error` traces as failures. Also verify the `INJECTION_BLOCKED_KIND` aggregation path isn't affected.
- **Benchmark impact:** None.
- **Verdict:** CONDITIONAL — This should be BLOCKED until the security implications are fully assessed. Adding to `FAILURE_KINDS` is the right fix, but the Digester aggregation logic for `hook_registry_error` needs careful design: should it produce a `FailureReport` with a new `proposed_class` (e.g., `'hooks-disabled'`)? Recommend splitting into two sub-issues: (1) add to FAILURE_KINDS, (2) design the Digester aggregation class.

---

### OBS-003: Track `model_retry` events in KPIs — **FLAG**

- **KPI alignment:** Advancing observability. `model_retry` events (emitted at runner.py:1449) are useful for operator awareness but do not directly advance any of the three PRD KPIs. They are an auxiliary health metric (like `token_budget_hit_rate`).
- **Security impact:** None.
- **Evidence:** `model_retry` events are emitted via `_on_retry` callback (runner.py:1446-1451) with `event.model_dump()` payload. These are triggered on 408/429/5xx/ConnectError retries (issue #200). Currently not surfaced in `foundry-kpis` output.
- **Test requirements:** Unit test for a new `_model_retry_rate()` function in kpis.py. Test with zero retries (→ 0.0) and with retries (→ correct count/rate).
- **Benchmark impact:** None. KPI-only addition.
- **Verdict:** FLAG — Low risk, well-scoped. However, this is an auxiliary metric, not a PRD KPI. Ensure the issue is labeled appropriately (e.g., `kpi-observability` not `kpi-regression-rate`). The test coverage requirement is clear.

---

### OBS-004: Track `tool_argument_parse_error` events in KPIs — **FLAG**

- **KPI alignment:** Same as OBS-003 — auxiliary observability metric.
- **Security impact:** Low. Parse errors indicate malformed model output, not adversarial input. However, a spike could signal model degradation.
- **Evidence:** `tool_argument_parse_error` events are emitted (runner.py confirmed, tests confirmed at `test_execution_agent_loop.py`). NOT tracked in kpis.py. The event is emitted when `_parse_tool_arguments()` fails.
- **Test requirements:** Same pattern as OBS-003 — new function in kpis.py + unit test.
- **Benchmark impact:** None.
- **Verdict:** FLAG — Same advisory as OBS-003. Well-scoped auxiliary metric. Ensure correct labeling.

---

### OBS-005: Track event_limit aborts as separate KPI — **PASS**

- **KPI alignment:** Directly advances `kpi-regression-rate` and the tracked Token Budget Hit Rate metric. `event_limit` aborts are a different failure mode from `token_budget` aborts and should be tracked independently.
- **Security impact:** Positive. Event-limit aborts can mask runaway loops that aren't caught by token budget or wall-clock caps.
- **Evidence:** Runner emits `task_aborted(reason="event_limit")` at multiple points (lines 1530, 1721, etc.). SECURITY.md §Runaway detection documents this. However, kpis.py has NO tracking for `event_limit` — only `token_budget_abort_count` and `token_budget_hit_rate` exist (issue #466/704). The SECURITY.md itself at line 74-75 mentions `event_limit_hit` as a KPI consumer, but this is aspirational — it's not implemented.
- **Test requirements:** New `_event_limit_abort_count()` and `_event_limit_hit_rate()` functions in kpis.py + unit tests. Mirror the existing `token_budget_abort_count` pattern.
- **Benchmark impact:** None.
- **Verdict:** Clear evidence-backed gap. SECURITY.md promises this metric but kpis.py doesn't deliver it.

---

### OBS-006: Add windowed trend analysis to tool latency — **FLAG**

- **KPI alignment:** Indirectly supports `kpi-improvement-rate` by enabling operators to detect latency regressions over time.
- **Security impact:** None.
- **Evidence:** No windowed trend analysis exists in the codebase (`rg` returns zero hits for `windowed|trend|alerting`). This is a net-new feature with no existing code to modify.
- **Test requirements:** Requires a design spec before tests. What window sizes? What metric aggregation? What output format? The acceptance criteria are underspecified.
- **Benchmark impact:** Could impact `tool_call` event overhead if implemented in the hot path.
- **Verdict:** FLAG — The need is valid but the proposal is underspecified. It needs a design phase (issue → ADR or design doc) before implementation. Recommend scoping to a minimal viable version: mean/median/p95 latency over the last N sessions in `foundry-kpis` output.

---

### OBS-007: Add threshold-based alerting for observability metrics — **FLAG**

- **KPI alignment:** Indirect — operational alerting supports all KPIs by surfacing anomalies early.
- **Security impact:** None.
- **Evidence:** No alerting exists in the codebase. This is a net-new feature.
- **Test requirements:** Requires a design spec. What thresholds? What channels (stderr, trace event, webhook)? What metrics?
- **Benchmark impact:** Unknown — depends on implementation location.
- **Verdict:** FLAG — Same as OBS-006. Valid need, underspecified proposal. Needs a design phase. Recommend splitting into concrete sub-issues: (1) define alertable metrics + thresholds, (2) implement alerting channel, (3) integrate with `foundry-kpis`.

---

### bench-A-01: read_multiple_files skill benchmark — **CONDITIONAL PASS**

- **KPI alignment:** Advances `kpi-improvement-rate` by adding a benchmark task that the Critic can gate against.
- **Security impact:** None. Read-only skill.
- **Evidence:** `harness/skills/read_multiple_files.json` exists. No corresponding benchmark test in `benchmarks/tasks/`. The skill is currently untested at the benchmark level.
- **Test requirements:** Needs a proper `@pytest.mark.benchmark` test under `benchmarks/tasks/` that exercises the `read_multiple_files` skill against representative multi-file inputs. Must follow ADR-0005 conventions.
- **Benchmark impact:** This IS a benchmark addition. It must not regress existing benchmarks. The new benchmark must pass before being added to the Critic baseline.
- **Verdict:** CONDITIONAL — The skill exists but the benchmark task design needs review. What does "success" look like? Correct file contents returned? Performance threshold? Define the acceptance criteria before implementing.

---

### bench-R-01: rate_limit hook benchmark — **CONDITIONAL PASS**

- **KPI alignment:** Advances `kpi-improvement-rate` and `kpi-regression-rate` by ensuring the rate-limit hook behaves correctly under load.
- **Security impact:** Positive. The rate-limit hook is a SECURITY.md guardrail (§Rate limits). A benchmark ensures it cannot be accidentally disabled or degraded.
- **Evidence:** `harness/hooks/rate_limit.py` exists. No corresponding benchmark in `benchmarks/tasks/`. SECURITY.md lists rate limits as a guardrail but no benchmark task tests them.
- **Test requirements:** Benchmark task that verifies: (1) rate limiting triggers at the configured threshold, (2) the hook does not block requests under the threshold, (3) the hook degrades gracefully when the registry is unavailable.
- **Benchmark impact:** This IS a benchmark addition.
- **Verdict:** CONDITIONAL — High-value security benchmark. But design needs review: what scenarios does the benchmark cover? Consider the security-evals benchmark family (ADR-0009) as a pattern to follow.

---

### harness-01: Declare `TokenAwarePruningHook._phase = 4` — **FLAG**

- **KPI alignment:** Indirectly supports `kpi-regression-rate` by ensuring correct hook execution ordering.
- **Security impact:** Low. Incorrect hook phase ordering could cause the pruning hook to fire before the injection firewall, potentially pruning security-relevant events. However, the `_phase` attribute controls hook registration order, not execution order in the current implementation.
- **Evidence:** `ContextPruningHook` has `_phase = 2` (context_pruning.py:157). `TokenAwarePruningHook` (context_pruning.py:297) does NOT inherit from `ContextPruningHook` — it's a standalone class. It does NOT define `_phase`. The proposal says to set `_phase = 4` (presumably to run after the injection firewall at phase 3). The `TokenAwarePruningHook` is registered via `register_token_aware_into()` (runner.py:1494), not via the manifest-based hook system.
- **Test requirements:** Test that verifies `TokenAwarePruningHook._phase == 4` and that the hook executes after the injection firewall.
- **Benchmark impact:** None.
- **Verdict:** FLAG — The need is valid (explicit phase declaration for clarity), but the `_phase` attribute's effect depends on the hook registry implementation. Verify that `_phase` is actually read by the registry when hooks are registered via `register_token_aware_into()`, not just declared but unused.

---

### infra-01: Extend uv-pin test allowlist to all workflows — **BLOCK**

- **KPI alignment:** Indirect infra improvement. No direct KPI impact.
- **Security impact:** Low. Pin enforcement is a supply-chain security measure.
- **Evidence:** `rg -n 'uv.*pin|UV_PIN' .github/workflows/` returns zero results. No `uv-pin` test or allowlist exists in any CI workflow. The proposal references a non-existent mechanism. The CI workflows are: `audit.yml`, `ci.yml`, `critic.yml`, `docker.yml`, `lint.yml`, `pre-commit.yml`, `quantization-sweep.yml`, `real-llm.yml`, `rocm.yml`, `secrets.yml`, `test.yml`.
- **Test requirements:** N/A — the referenced mechanism doesn't exist.
- **Benchmark impact:** N/A
- **Verdict:** **BLOCK — phantom proposal.** No `uv-pin` test allowlist exists to extend. The scout may have hallucinated or referenced a removed/discarded mechanism. If the intent is to enforce `uv` lockfile pinning across CI, that's a valid issue but needs to be re-scouted from scratch.

---

### docs-01: Align FOUNDRY_TASK_TIMEOUT default in docs — **PASS**

- **KPI alignment:** No direct KPI impact. Documentation accuracy prevents operator confusion.
- **Security impact:** Low. Misleading timeout documentation could lead operators to set incorrect values, affecting runaway detection.
- **Evidence:** Clear discrepancy:
  - `docs/SECURITY.md:62`: default **300** seconds
  - `.env.example`: `FOUNDRY_TASK_TIMEOUT=600`
  - `docs/PHASE3-FINDINGS.md:148,160,181`: **600** seconds
  - `docs/adr/0020-phase-3-findings.md:82`: **600** seconds
  - `src/foundry_x/execution/runner.py`: reads from env, no hardcoded default visible (uses the env value)
  The actual runtime default depends on `.env.example` (600) and the code's fallback. SECURITY.md says 300.
- **Test requirements:** N/A — documentation-only change.
- **Benchmark impact:** None.
- **Verdict:** Clear discrepancy. SECURITY.md should be updated to say 600 (matching `.env.example` and Phase 3 findings), or the code default should be made explicit. Recommend updating SECURITY.md to match the actual runtime default.

---

### docs-02: Document hook registry degradation mode — **PASS**

- **KPI alignment:** Indirect. Proper documentation helps operators understand when security hooks are disabled.
- **Security impact:** **POSITIVE.** The hook registry degradation mode (when `hook_registry_error` fires, all hooks including the injection firewall are disabled) is a security-critical state. It IS documented in CONTEXT.md and the runner docstring, but operator-facing documentation in SECURITY.md or OPERATOR.md would improve awareness.
- **Evidence:** `hook_registry_error` events are emitted (runner.py via `_resolve_hook_registry`), tracked in kpis.py (`hooks_disabled_count/rate`), and documented in CONTEXT.md. SECURITY.md does not explicitly document the degradation path.
- **Test requirements:** N/A — documentation-only.
- **Benchmark impact:** None.
- **Verdict:** Valid documentation improvement. Ensure the docs explain: (1) when degradation occurs, (2) which hooks are affected, (3) what the operator should do (check hook imports, review harness edits).

---

### docs-03: Clarify evolution CLI --async vs --no-verify — **BLOCK**

- **KPI alignment:** N/A
- **Security impact:** N/A
- **Evidence:** `src/foundry_x/evolution/evolve_cli.py` does not exist. `_run_loop_async` does not exist. The evolution CLI may be implemented under a different name or may not exist yet. `rg -rn '_run_loop|evolve.*cli' src/foundry_x/` returns zero results.
- **Test requirements:** N/A
- **Benchmark impact:** N/A
- **Verdict:** **BLOCK — phantom proposal.** Same issue as evolution-02. The referenced file and functions do not exist. The scout hallucinated or referenced stale/incorrect information. Reject and close.

---

### security-01: Fix resolve_bash_path for `**` glob patterns — **BLOCK**

- **KPI alignment:** N/A
- **Security impact:** Would be relevant if the function existed — glob patterns could enable path traversal.
- **Evidence:** `rg -rn 'resolve_bash|bash_path|resolve_bash_path' src/ harness/ tests/` returns zero results. The function does not exist anywhere in the codebase. The bash skill (`harness/skills/bash.json`) delegates entirely to `subprocess.run` with `shell=False` and `shlex.split`. There is no glob/fnmatch handling in the skill or in the foundry code. The `**` glob pattern handling would be done by the shell itself (`bash -c "ls **"`) or by `glob.glob()` in user code, not by a `resolve_bash_path` function.
- **Test requirements:** N/A — the function doesn't exist.
- **Benchmark impact:** N/A
- **Verdict:** **BLOCK — phantom proposal.** The function `resolve_bash_path` does not exist in the codebase. The scout hallucinated. If there is a legitimate security concern about glob expansion in the bash skill, it needs to be re-scouted based on the actual implementation (subprocess with shell=False).

---

## Cross-Cutting Observations

### Phantom Proposals (4 of 20)
`evolution-02`, `evolution-03`, `infra-01`, and `security-01` reference files, functions, or mechanisms that do not exist in the codebase. This suggests the scouting process may be generating proposals from stale context or hallucinated code paths. **Recommendation:** Add a pre-validation step to the scouting workflow that confirms the existence of referenced files/functions before generating proposals.

### Security-Relevant Gaps (2 proposals)
`OBS-001` and `OBS-002` both address Digester blindness to security-relevant failure modes. `hook_registry_error` disabling the injection firewall without Digester classification is the more critical of the two. These should be prioritized.

### Documentation Discrepancies (2 proposals)
`docs-01` (FOUNDRY_TASK_TIMEOUT default mismatch) and `docs-02` (missing degradation mode docs) are straightforward fixes. `docs-01` is especially important because SECURITY.md is the security reference document and carries an incorrect default.

### Underspecified Proposals (2 proposals)
`OBS-006` (windowed trend analysis) and `OBS-007` (threshold-based alerting) are valid needs but lack design specifications. They should be converted into design-phase issues rather than implementation-ready issues.

---

## Recommendation

| Action | Proposals |
|--------|-----------|
| **Approve and merge** | trace-01, execution-01, OBS-001, OBS-005, docs-01, docs-02 |
| **Approve with conditions** | evolution-01 (verify timeout gap), OBS-002 (design Digester class), bench-A-01 (define success criteria), bench-R-01 (follow ADR-0009 pattern), harness-01 (verify _phase is consumed) |
| **Needs design phase** | OBS-003, OBS-004, OBS-006, OBS-007 |
| **Reject and close** | evolution-02, evolution-03, infra-01, security-01 |
