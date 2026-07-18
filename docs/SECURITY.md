# Security

FoundryX is a self-improving agent harness foundry. It edits code,
edits its own prompt, and runs LLM inference against arbitrary inputs.
This makes security a first-class concern, not a checkbox.

This document is a starting point. As the harness gains capabilities
(network access, shell access, self-modification of arbitrary files),
the threat model and guardrails must be revisited.

## Threat model

We design against these threats:

1. **Harness degradation.** The `Evolver` produces an edit that looks
   good on the benchmark but subtly degrades the agent on real tasks
   (regression, silent skill loss, prompt-injection by the harness
   itself).
2. **Prompt injection from traced content.** Tool-call results, file
   contents, or benchmark tasks contain adversarial text that the
   model then acts on.
3. **Supply-chain compromise.** A malicious or vulnerable dependency
   enters via `uv add` and runs inside the trace pipeline.
4. **Secret leakage.** API keys, tokens, or model weights committed
   to the repo or written into the trace store.
5. **Resource exhaustion.** A bad evolution edit causes the agent to
   enter a runaway loop, blowing up the GPU, the disk, or the wallet.
6. **Local privilege.** The harness runs on a host with user
   permissions; a buggy hook could read files outside the workspace.

## Guardrails

- **Critic gate (hard requirement, ADR-0004).** Every harness edit
  must pass the `Critic` benchmark gate before it is marked active.
  Enforced via `.github/workflows/critic.yml`. The CLI flag
  `--no-verify` (issue #888) skips the gate for local
  experimentation only — it cannot ship a harness edit to `main`.
  See [`--no-verify` and the Critic gate](#no-verify-and-the-critic-gate)
  below for the threat model.
- **Test gate.** Every change to `src/foundry_x/` must pass
  `uv run pytest`. CI blocks merges. Enforced via
  `.github/workflows/ci.yml`.
- **Lint gate.** Every change must pass `uv run ruff check .`. CI
  blocks merges. Enforced via `.github/workflows/ci.yml`.
- **Pre-commit hooks (recommended):** `ruff` and a `gitleaks`
  check to block common credential patterns.
- **Trace sanitization.** Trace writers MUST redact values matching
  secret-like patterns (`sk-...`, `Bearer ...`, PEM blocks) before
  persisting. See `src/foundry_x/trace/logger.py`.
- **Prompt-input firewall.** Tool-call results that will be
  re-injected into a prompt must be checked for injection markers
  (`ignore previous instructions`, role-tag injection, etc.) and
  either truncated or flagged for human review. Every block emits a
  ``firewall_exception`` trace event (issue #823) carrying
  ``hook_name``, ``pattern_matched``, and ``risk_score`` so firewall
  events are queryable via ``foundry-x-trace``.
- **Rate limits.** The `Evolver` is rate-limited: max N proposals per
  hour, max M lines of harness diff per proposal, max LLM calls per hour,
  and max cost per day. Defaults live in
  `src/foundry_x/evolution/evolver.py`:
  - Proposals: 10/hour
  - Diff lines: 200/proposal
  - LLM calls: 60/hour (shared across all Evolver instances)
  - LLM cost: $5.00/day (shared across all Evolver instances)
- **Runaway detection.** The runner enforces three resource caps per task:
  - ``FOUNDRY_TASK_TIMEOUT`` (wall-clock seconds, default 600): when exceeded,
    ``run_with_limits`` records ``task_aborted(reason="wall_clock")`` and
    raises ``asyncio.TimeoutError``.
  - ``FOUNDRY_TOKEN_BUDGET`` (total tokens, unset by default): when the
    running token total (accumulated from each ``ModelResponse.usage``) exceeds
    this cap, ``run_task`` records ``task_aborted(reason="token_budget")`` and
    terminates with ``outcome.status="failed"``, ``outcome.reason="token_budget"``.
  - ``FOUNDRY_MAX_EVENTS_PER_SESSION`` (event count, unset by default): when
    the accumulated event count reaches this cap, ``run_task`` records
    ``task_aborted(reason="event_limit")`` and terminates with
    ``outcome.status="failed"``, ``outcome.reason="event_limit"``.
  All caps emit a ``task_aborted`` trace event and set the terminal
  ``outcome`` accordingly so KPI consumers (``token_budget_abort_count``,
  ``token_budget_hit_rate``, ``wall_clock_abort_count``,
  ``event_limit_abort_count``) can observe the abort. Issue #869
  adds ``event_limit_abort_count`` so event-limit-driven failures
  are visible alongside the existing abort counters.
- **Sandbox.** Run benchmarks and evolution inside a Docker
  container with read-only mounts for the host filesystem (see
  `infra/`). The default local dev path runs unsandboxed but should
  be migrated to the container for any non-trivial evolution run.

## Secrets

- **Never commit** `.env`, API keys, model weights, or credentials.
- `.env.example` is the only `.env*` file in git.
- If a secret is committed by accident, rotate it immediately. Treat
  the commit as compromised even after removal: git history is
  forever.
- Trace stores MUST NOT contain raw API keys. Strip them at write
  time. Coverage includes AWS access key IDs, GCP service account
  emails / ADC paths / project ID env vars, GitHub PATs, JWTs,
  Stripe live keys, Slack tokens, and Bearer tokens.
- If a trace is found to hold a secret that slipped past the write-time
  scrubber, the `foundry-trace` CLI offers two operator remediation
  commands (issue #192, backed by the `TraceLogger.delete_session` and
  `TraceLogger.redact_event` helpers from issue #157):
  - `foundry-trace redact-session SESSION_ID` deletes the session and
    all its events and prints the count removed.
  - `foundry-trace redact-key SESSION_ID EVENT_INDEX KEY` overwrites a
    single payload field with `[REDACTED]`, exiting non-zero if the
    event index is out of range.
  Both accept `--out` to append a JSONL audit record of the action.

## Dependencies

- We pin via `uv` and commit the lockfile (ADR-0002).
- New dependencies require justification in the PR. High-risk
  packages (anything that shells out, anything with native
  extensions, anything in a maintenance dormancy) require an ADR.
- We run `uv pip audit` (or equivalent) in CI on every PR. Enforced
  via `.github/workflows/audit.yml`.

## Prompt injection

The trace pipeline reads tool outputs and feeds them to models.
Treat all such content as untrusted:

- Quote-wrap tool results before injecting into a prompt (or use a
  structured `tool_result` channel that the model cannot confuse with
  instructions).
- Strip or escape role-tag sequences (`system:`, `assistant:`,
  `<|...|>`).
- Reject evolution proposals whose diff is dominated by text that
  resembles instructions to the harness itself.

## `--no-verify` and the Critic gate

`foundry-evolve evolve --no-verify` (issue #888) is the only documented
way to skip the `Critic.evaluate(...)` gate from the CLI. The flag is
gated behind an explicit operator action, emits a prominent `stderr`
warning naming ADR-0004, and **never** silently produces a shippable
harness edit. The threat model for the flag is:

- **Local experimentation only.** The flag exists so an operator can
  inspect what the `Evolver` would propose against a failure without
  paying the Critic's pytest + benchmark cost. It is **not** a CI gate
  bypass: the orchestrator's own PRs go through normal CI, and the
  `Critic` workflow in `.github/workflows/critic.yml` still runs the
  full gate on every harness-touching PR.
- **Audit trail preserved.** The CLI records a synthetic
  `CriticVerdict(verdict=None, notes="--no-verify: skipped")` via
  `record_verdict`, so the `critic_verdict` trace event still fires.
  Downstream consumers treat `verdict=None` as a non-approval:
  - `analyze_regressions` / `generate_regression_report` count a
    skipped verdict as neither an approval nor a rejection, and the
    regression-pairing logic does not seed `prior_passed` from a skip
    (so an unrelated later failure is not mis-attributed as a
    regression).
  - The `foundry-kpis` approval/rejection counters see the skip in
    the total but not in either bucket.
- **Cannot ship to `main`.** A harness edit persisted with
  `verdict=None` is rejected by the Critic workflow on the next PR.
  The flag does not modify any harness DNA file (system prompt, hooks,
  skills) in place; it only persists the proposed edit and the
  synthetic verdict to the trace store and `ProposedEditStore` for
  operator review. Operators who want to actually apply the edit must
  still go through `foundry-evolve approve` → `foundry-evolve apply`,
  and the apply step still runs `git apply` against the harness
  directory.
- **Combined with `--background`.** When both flags are set, the
  background subprocess inherits the skip — the operator gets a
  non-blocking `Evolver` run with no Critic cost. The same audit-trail
  and cannot-ship-to-main guarantees apply.

Operators who see a `critic_verdict` event with `verdict=None` in the
trace store should treat it as evidence of a `--no-verify` run, not
evidence of an approval. The `notes` field always carries the string
`--no-verify: skipped` so the skip is grep-able:

```
foundry-trace events-grep <session_id> \
    --pattern critic_verdict \
    --db logs/traces.db
```

## Reporting a vulnerability

Please email **security@anchapin.dev** (or open a private security
advisory on GitHub). Do **not** file a public issue for suspected
vulnerabilities.

We aim to acknowledge reports within 72 hours and to publish a fix or
mitigation within 30 days for high-severity issues.

## Scope of this document

This is a starting point. As the harness gains capabilities, the
threat model and guardrails must be revisited.

The `Critic` gate (ADR-0004) carries a dedicated security-evals
BenchmarkTask family that pins each of the guardrails above to a
regression test. See [ADR-0009](adr/0009-security-evals-benchmark-family.md)
for the family definition and selection rules. The four tasks ship
under `benchmarks/tasks/`:

- `test_secret_redaction_evals.py` — `TraceLogger` scrubs the token
  set enumerated in §Secrets above (issue #3 + #121).
- `test_injection_firewall_evals.py` — `InjectionFirewallHook`
  truncates adversarial tool results before re-injection
  (issue #5 + #122).
- `test_hook_isolation_evals.py` — a thrown hook exception does not
  abort the chain (issue #21).
- `test_evolver_guardrail_evals.py` — `ProposedEdit` confines edits
  to `harness/{system_prompt.txt,hooks/,skills/}` and the `Evolver`
  enforces the §"Rate limits" cap.
- `test_foundation_token_budget.py` — token-budget mechanism is
  resilient to prompt-injection attempts embedded in task fixtures;
  the agent must not exfiltrate or manipulate the `FOUNDRY_TOKEN_BUDGET`
  value when an adversarial input is present in the task description
  (issue #822).

A regression in any of the five flips the Critic red before the
proposed harness edit ships. The Critic additionally persists a
regression baseline at `logs/critic_baseline.json` (ADR-0004 step 3,
issue #186): once a benchmark task is recorded as passing, any
later flip to failing rejects the gate with `regression:<task_name>`
in `failed_checks`.
