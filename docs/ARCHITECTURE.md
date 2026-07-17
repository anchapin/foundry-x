# Architecture

> This document is the authoritative map of FoundryX's runtime architecture.
> For terminology see [CONTEXT.md](./CONTEXT.md); for decisions see
> [adr/README](./adr/README.md).

## System overview

FoundryX has two layers:

- **`src/foundry_x/`** — the *foundry*: a Python package that wraps and
  evolves the agent harness. It owns the eval loop (Runner → TraceLogger →
  Digester → Evolver → Critic), the trace store, and the observability
  CLIs.
- **`harness/`** — the *artifact being evolved*: system prompt, hooks,
  and skills. Version-controlled; modified only through the Critic gate.

```
task → Runner → trace → Digester → Evolver → ProposedEdit
                                                   ↓
                                              Critic → accept | reject
                                                           ↓
                                                        harness (evolved)
```

## Components

### Runner (`src/foundry_x/execution/runner.py`)

Drives a single agent session. Reads the harness, calls the model via
`ModelAdapter`, fans tool calls through the `HookRegistry`, and records
every event to the `TraceLogger`. Produces the trace events described in
[CONTEXT.md §Event kinds](./CONTEXT.md#event-kinds).

### TraceLogger (`src/foundry_x/trace/logger.py`)

Ground-truth recorder. Persists every event to `logs/traces.db` (SQLite,
WAL mode) or a JSONL export. All FoundryX CLIs surface this store
without going through the agent.

### Digester (`src/foundry_x/evolution/digester.py`)

Reads a trace and produces a `FailureReport`: what failed, which step,
and the candidate root cause. Aligns its classifier against the `kind`
vocabulary in [CONTEXT.md §Event kinds](./CONTEXT.md#event-kinds).

### Evolver (`src/foundry_x/evolution/evolver.py`)

Meta-agent that consumes a `FailureReport` and emits a `ProposedEdit`
against the harness. Lives in the harness layer; not called by the
foundry directly.

### Critic (`src/foundry_x/evolution/critic.py`)

Gatekeeper. Runs the pytest suite and the benchmark suite against the
candidate harness and records a `critic_verdict`. A harness edit that
regresses any previously-passing benchmark is rejected.

## Trace store layout

```
logs/
  traces.db          # SQLite (WAL mode, one file per run)
  traces.db-shm      # shared memory file
  traces.db-wal      # WAL journal
```

The SQLite default is `logs/traces.db`. Use `--db logs/traces.jsonl` to
target the JSONL backend. Both backends share the same schema; see
[ADR-0003](./adr/0003-sqlite-as-trace-store.md) for the rationale.

---

## Debugging

When the eval loop misbehaves — evolver produces no `ProposedEdit`,
critic hangs, runner OOMs — use the `foundry-x-trace` CLI (also
available as `foundry-trace`) to inspect traces without reading source
code.

### Inspecting logs locally

List sessions:

```
foundry-trace session-list --db logs/traces.db
```

Show every event in a session as a timeline:

```
foundry-trace session-show <session_id> --db logs/traces.db
```

Print every event whose payload JSON matches a pattern:

```
foundry-trace events-grep <session_id> --pattern "tool_result" --db logs/traces.db
```

Render a failure report for a session:

```
foundry-trace render-failure <session_id> --db logs/traces.db
```

### Replaying a session

Replay is not a live re-run; it is a read-only inspection of a recorded
session. To replay, load a session by ID and walk the event timeline:

```
foundry-trace session-show <session_id> --db logs/traces.db
```

To replay the failureDigester pipeline against a specific session:

```
foundry-trace render-failure <session_id> --trace-path logs/traces.db
```

Export a session for sharing:

```
foundry-trace export <session_id> --out /tmp/session.jsonl --db logs/traces.db
```

### Common failure modes and trace signatures

| Failure | Trace signature | How to confirm |
|---|---|---|
| **Evolver produces no ProposedEdit** | Session ends after `task_completed` with no `critic_verdict` event following it. The `Digester` ran but the Evolver step was never reached. | `foundry-trace events-grep <sid> --pattern critic_verdict` returns nothing. Check that the upstream `task_completed` is present: `foundry-trace events-grep <sid> --pattern task_completed`. |
| **Critic hangs / timeout** | `task_aborted` event with `{"reason": "wall_clock", "timeout_s": N}`. No `critic_verdict` follows. | `foundry-trace events-grep <sid> --pattern task_aborted`. If `reason == "wall_clock"` the wall-clock cap fired; if `token_budget` is non-null the token budget cap fired. See [ADR-0010](../adr/0010-runner-agent-loop.md) §Termination. |
| **Runner OOM** | `task_failed` event with `{"error_type": "MemoryError"}` or similar, followed by `task_completed` in degraded state. Also look for `model_error` events with `step` incrementing toward `max_steps` without an `outcome`. | `foundry-trace events-grep <sid> --pattern MemoryError`. Check the session's last step count: `foundry-trace session-show <sid>` and look for the highest `step` value before the failure. |
| **No tool calls emitted (tool surface missing)** | `model_response` events with `tool_calls: []` on every step, ending in `outcome{status: "success", reason: "final_answer"}` after step 0. | `foundry-trace events-grep <sid> --pattern tool_calls`. If every match shows `tool_calls: []`, the model never saw the tool surface. Check that `harness/skills/*.json` files are loadable and the Runner loaded them at session start. |
| **Hook registry error** | `hook_registry_error` event with non-null `error_type`. The session continues in degraded mode (no hook fan-out). | `foundry-trace events-grep <sid> --pattern hook_registry_error`. If present, the `InjectionFirewallHook` did not run; treat the session as potentially compromised. See [AGENTS.md](../AGENTS.md) §2. |
| **Injection attempt** | `injection_blocked` events with `{"markers": [...], "tool": "...", "preview": "..."}`. Multiple such events in one session indicate an active adversarial attempt. | `foundry-trace events-grep <sid> --pattern injection_blocked`. Each event names the tool that was blocked and the suppressed text preview. |

### Pruning old sessions

Sessions accumulate in `logs/`. Enforce retention with:

```
# Keep the 10 most recent sessions
foundry-trace prune --keep-last 10 --db logs/traces.db

# Remove sessions older than 30 days
foundry-trace prune --older-than 30 --db logs/traces.db

# Dry run first
foundry-trace prune --older-than 30 --dry-run --db logs/traces.db
```

### Further reading

- [CONTEXT.md §Event kinds](./CONTEXT.md#event-kinds) — full payload
  contracts and failure-signal vocabulary
- [ADR-0003](./adr/0003-sqlite-as-trace-store.md) — trace store design
- [ADR-0010](./adr/0010-runner-agent-loop.md) — Runner loop semantics
- [SECURITY.md](./SECURITY.md) — secrets redaction and runaway detection
