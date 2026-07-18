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
  Enforced via `.github/workflows/critic.yml`.
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
  - ``FOUNDRY_TASK_TIMEOUT`` (wall-clock seconds, default 300): when exceeded,
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
  ``token_budget_hit``, ``event_limit_hit``) can observe the abort.
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
