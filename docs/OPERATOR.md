# OPERATOR.md

> A guide for human operators running the FoundryX harness against their tasks.
> For the visual loop diagram, see the [loop diagram in CONTEXT.md](./CONTEXT.md#the-loop).
> For the full glossary of terms, see [Glossary](#glossary) below.

## What is an operator?

An **operator** is a human who runs the FoundryX harness against their own
tasks. The operator observes traces, identifies failure patterns, and — when
the harness needs changes — routes a failure report through the Evolver ->
Critic loop rather than editing the harness directly by hand.

## The Evolution Loop

FoundryX runs a continuous improvement loop:

```
  task -> Runner -> trace
                 |
                 v
            Digester -> failure report
                            |
                            v
                        Evolver -> ProposedEdit
                                      |
                                      v
                                 Critic -> accept | reject
                                                |
                                                v
                                            harness (updated)
```

A single iteration is small. The value comes from running the loop many times
per day against a benchmark suite.

## Key Concepts

### Harness

The artifact being evolved. The harness is the product — it defines the
agent's persona, operating rules, hooks, and tool surface. See
[PHILOSOPHY.md](./PHILOSOPHY.md) §6.

### Evolver

The meta-agent that takes a failure report and proposes a ProposedEdit
against the harness. It is the component that proposes harness mutations.

### Critic

The gatekeeper. The Critic runs the proposed edit through the pytest suite
and benchmark suite, then rejects any edit that causes regressions.

### ProposedEdit

A structured pydantic model representing a proposed change to the harness.
It is the unit of work produced by the Evolver and consumed by the Critic.

### Regression Gate

The Critic's benchmark gate. Every harness edit must pass the regression gate
before it is marked active. If any benchmark flips red, the edit is rejected.

### Runner

The component that drives a single agent session against a task. The Runner
reads the harness, calls the model, and writes the trace.

### TraceLogger

The component that wraps an agent session and persists every prompt, tool
call, and outcome to the trace store. It is the ground-truth recorder.

### Digester

The component that parses traces and produces a failure report: what failed,
where, and the candidate root cause.

### Failure Report

The structured artifact produced by the Digester from a trace. It names what
failed, where, and the candidate root cause. The failure report is consumed
by the Evolver as the basis for a ProposedEdit.

### Hook

Middleware that runs around every tool call. Hooks live in
`harness/hooks/*.py` and are loaded by the Runner at session start.

### Benchmark Task

A standardised coding task marked with `@pytest.mark.benchmark`. The Critic
uses benchmark tasks to gate harness changes — a regression in any benchmark
flips the Critic red.

### Foundry

The Python package (`src/foundry_x/`) that wraps and evolves the harness.
Built and maintained primarily by humans. Distinct from the harness, which is
the artifact being evolved.

### FoundryAgent

The runtime coding agent persona as defined in the harness. Its persona and
operating rules live in `harness/system_prompt.txt`.

## The Operator Workflow

Operators follow a six-step cycle (see AGENTS.md §3):

1. **Observe** — read traces and existing code before proposing.
2. **Digest** — turn observations into a failure report.
3. **Propose** — produce a ProposedEdit (or, for a human, a PR).
4. **Evaluate** — run the test and benchmark gate (Critic).
5. **Commit** — atomic, conventional-commits change.
6. **Hand off** — open a PR and wait for review.

## Hook Registry Failure Degradation Mode

When `harness.hooks.get_registry()` raises during session start,
`Runner._resolve_hook_registry()` catches the exception, records a
`hook_registry_error` trace event, and returns `None`. The session
continues **with all hooks silently disabled**. This is the
"hook registry failure" degradation mode first surfaced by the
runner instrumentation in issue #260 and tracked as a KPI by
issue #585. Operators must treat any session that carries a
`hook_registry_error` event as **degraded** — the agent runs, but
its middleware layer (security controls, rate limits, context
pruning) is offline.

> A *missing* harness (`ImportError` in `_resolve_hook_registry`) is
> a legitimate degraded mode that returns `None` without comment.
> The case documented here is the *importable-but-broken* registry
> path — `get_registry()` is reachable but raises — which must be
> observed, not swallowed (AGENTS.md §2 — never silently swallow an
> exception).

### Affected security controls {#affected-hook-controls}

When the registry fails to load, **every** hook in
`harness/hooks/__init__.py` is unregistered for the lifetime of the
session. The high-impact controls are:

| Hook | Role | What you lose when disabled |
|------|------|------------------------------|
| `InjectionFirewallHook` | Quarantines tool-call results that carry prompt-injection markers before they are re-injected into the prompt. Mandated by `SECURITY.md` "Prompt-input firewall" and emits `firewall_exception` events (`issue #823`). | Adversarial tool output is passed straight to the model unfiltered. Untrusted content reaches the model verbatim. |
| `ContextPruningHook` / `TokenAwarePruningHook` | Caps the live context window at the thresholds in `harness/manifest.json` (`token_threshold`, `event_threshold`) and emits `context_pruned` events. Registered lazily by the runner. | Token- and event-budget enforcement drift off; long sessions can balloon and breach `FOUNDRY_TOKEN_BUDGET` without a `task_aborted` event to mark it. |
| `RateLimitHook` | Enforces the Evolver guardrail from `SECURITY.md` "Rate limits" (max proposals/hour, max diff lines/proposal, max LLM calls/hour, max daily cost). | The Evolver can exceed its cost and proposal budget, blowing past the runaway detection that `SECURITY.md` depends on. |

`harness/hooks/base.py` (the registry itself) is the load-bearing
piece — when it raises, the four hooks above cannot self-register,
and `register_into(...)` calls inside the runner cannot find a
registry to register into.

### Detecting the condition {#detecting-hook-registry-failure}

Three signals are available. Any one of them confirms the
degradation mode is active for the relevant time window.

1. **Trace event** — every failed registry load records exactly one
   `hook_registry_error` event with payload
   `{"error_type": <ExceptionClass>, "message": <str>}` (see
   `src/foundry_x/execution/runner.py:670-681`). Grep a single
   session:

   ```
   foundry-trace events-grep <session_id> \
       --pattern hook_registry_error \
       --db logs/traces.db
   ```

   Presence means **that session ran with the middleware layer
   disabled**. A non-empty `error_type` identifies the exception
   class (e.g. `KeyError`, `ImportError`, `AttributeError`) that
   `get_registry()` raised; the `message` field carries the
   str-formatted exception text.

2. **KPI `hooks_disabled_count`** — the cumulative count of
   `hook_registry_error` events across the trace store, paired with
   `hooks_disabled_rate`, the fraction of sessions that recorded at
   least one such event. Both are computed by
   `src/foundry_x/observability/kpis.py:_hook_registry_errors`
   (issue #585) and surfaced by `foundry-kpis`:

   ```
   foundry-kpis --db logs/traces.db
   ```

   `hooks_disabled_count > 0` is the canonical alarm;
   `hooks_disabled_rate > 0` confirms the proportion of affected
   sessions. A non-zero value on a fresh store is the operator's
   signal to investigate **before** trusting any subsequent trace.

3. **Digester classification** — once issue #867 lands, the
   `Digester` will tag sessions with `hook_registry_error` events
   under a dedicated failure class so the `Evolver` can route a
   `ProposedEdit` that repairs the harness manifest or hook loader.
   Until then, filter the regression report for
   `hook_registry_error` events the same way you would any other
   failure signature.

### Recovery {#recovering-from-hook-registry-failure}

Treat a non-zero `hooks_disabled_count` as a **session-affecting
incident**, not a metric blip. Walk this checklist in order:

1. **Confirm the harness manifest is intact.** Open
   `harness/manifest.json` and confirm the `hooks` array lists
   `["base", "injection_firewall", "context_pruning", "rate_limit",
   "token_aware_pruning"]`. A missing entry here causes the registry
   to raise because `harness/hooks/__init__.py` imports each hook
   module to activate its self-registration.
2. **Verify the hook files are present.** Every hook listed in the
   manifest must have a corresponding module under `harness/hooks/`:

   ```
   ls harness/hooks/{base,injection_firewall,context_pruning,rate_limit,token_aware_pruning}.py
   ```

   A missing file produces an `ImportError` at registry load time.
3. **Confirm the harness package is importable.** The registry
   resolver returns `None` silently when `import harness.hooks`
   raises `ImportError` — *that* path is a legitimate degraded mode
   and is not what this section covers. From the runner's Python
   environment, run:

   ```
   python -c "import harness.hooks; print(harness.hooks.get_registry())"
   ```

   A non-zero exit or traceback usually means `harness/` was not
   added to `sys.path` (the foundry uses lazy import — see
   `_resolve_hook_registry`'s docstring) or a sub-dependency failed
   to import. Address the underlying import error before restarting.
4. **Read the last `hook_registry_error` payload.** The `error_type`
   field names the exception class; the `message` field names the
   missing key, module, or attribute. Use those to drive the
   fix — **do not** speculative-edit the harness. Harness edits
   route through the `Critic` gate (ADR-0004, AGENTS.md §2).
5. **Restart the runner** and run a smoke session. Confirm the new
   session has no `hook_registry_error` event:

   ```
   foundry-trace events-grep <new_session_id> \
       --pattern hook_registry_error \
       --db logs/traces.db
   ```

   Presence on the new session means the fix did not take;
   absence confirms recovery. Re-run `foundry-kpis` and confirm
   `hooks_disabled_count` no longer increments.

Until recovery is confirmed, consider **all recent sessions in the
affected window as potentially compromised** by prompt injection
that the `InjectionFirewallHook` would normally have quarantined.
Do not feed untrusted external content into the agent during the
investigation window — see `SECURITY.md` §"Prompt injection". If a
`ProposedEdit` is needed to repair the harness loader, route it
through the standard `Evolver` -> `Critic` pipeline; do not
hand-edit `harness/hooks/*.py` (ADR-0004).

### Trace event reference {#hook-registry-error-event-reference}

| Field | Type | Meaning |
|-------|------|---------|
| `kind` | `"hook_registry_error"` | Event discriminant. Matches the entry in `CONTEXT.md` §Event kinds. |
| `payload.error_type` | `str` | Exception class name raised by `get_registry()` (e.g. `KeyError`, `ImportError`, `AttributeError`). |
| `payload.message` | `str` | The exception's `str()` output. Carries the missing key, module, or attribute that triggered the failure. |

See `CONTEXT.md §Event kinds` for the full payload contract,
`SECURITY.md` for the controls the affected hooks enforce, and
`ADR-0004` for why this failure mode cannot bypass the `Critic`
gate.

## Server Supervision (FoundryServerManager, issue #899)

The runner supervises the local model server (`llama-server` or any
OpenAI-compatible endpoint) through a `FoundryServerManager` that
probes `GET /health` before each session and triggers a bounded
exponential-backoff restart loop when the server becomes unhealthy
mid-session. The supervisor is **opt-in** via
`FOUNDRY_SERVER_AUTOSTART` — when set to `0` the manager degrades to
a passive health-checker that records `server_unavailable` trace
events but does not spawn or kill anything (preserves the existing
operator workflow of a manually-launched server on a remote box).

### Failure mode

When `is_healthy()` returns `False` at the start of an agent-loop
iteration, the runner records a `server_unavailable` trace event and
calls `restart()`. The restart loop performs up to 3 attempts with
exponential backoff (1 s, 2 s, 4 s; capped at 8 s). If the loop
re-establishes `/health` the agent loop continues; otherwise the
session terminates with `outcome.status="failed"` and
`outcome.reason="server_unavailable"`.

### Detecting the condition {#detecting-server-unavailable}

Three signals are available:

1. **Trace event** — every failed health probe records exactly one
   `server_unavailable` event with payload
   `{"step": int, "host": str, "health_url": str, "restart_attempted": bool}`.
   Grep a single session:

   ```
   foundry-trace events-grep <session_id> \
       --pattern server_unavailable \
       --db logs/traces.db
   ```

   Presence means **that session observed a llama-server failure
   mid-run**. The `restart_attempted` field names whether the
   supervisor tried to recover; check the session's `outcome.reason`
   to see if recovery succeeded (`"final_answer"` / `"max_steps"` /
   `"token_budget"`) or failed (`"server_unavailable"`).

2. **KPI `server_restart_count`** — the cumulative count of
   `server_unavailable` events across the trace store, surfaced by
   `foundry-kpis` and the regression baseline report. A non-zero
   value on a fresh store is the operator's signal to investigate
   llama-server reliability — typically GPU OOM, host reboot, or a
   crashed background process. The metric is intentionally separated
   from model-quality KPIs so an infrastructure regression does not
   get blamed on a harness edit.

3. **Runner-managed process lifecycle** — when
   `FOUNDRY_SERVER_AUTOSTART=1` the manager owns the subprocess and
   restarts it transparently; the PID is held inside
   `FoundryServerManager._proc` and torn down on `stop()` (or process
   exit). When `FOUNDRY_SERVER_AUTOSTART=0` the manager does not
   spawn or kill anything — the operator is responsible for the
   server process.

### Recovery {#recovering-from-server-unavailable}

Treat a non-zero `server_restart_count` as an
**infrastructure-affecting incident**, not a metric blip. Walk this
checklist in order:

1. **Confirm the model file is intact.** Open
   `LLAMACPP_MODEL_PATH` (or `FOUNDRY_MODEL_PATH`) and verify the
   GGUF file exists and has not been truncated. The supervisor
   refuses to launch when `LLAMACPP_MODEL_PATH` is unset.
2. **Check `rocm-smi` for VRAM pressure.** A Q5_K_M model needs
   ~5.5 GB VRAM at full offload; partial offload
   (`FOUNDRY_SERVER_NGpuLayers`) trades speed for VRAM. A GPU OOM
   manifests as the server crashing immediately after startup — the
   supervisor's restart loop then repeatedly hits the same failure.
3. **Verify the host:port pair matches `LLAMACPP_HOST`.** The
   supervisor's `host` field is the value of `LLAMACPP_HOST`. A
   misconfigured value (e.g. pointing at `0.0.0.0` instead of
   `127.0.0.1`) causes `/health` to return 404 forever.
4. **Restart `llama-server` out-of-band when autostart is disabled.**
   When `FOUNDRY_SERVER_AUTOSTART=0` the supervisor will not
   re-spawn the process; the operator must restart it manually
   (`./llama.cpp/build/bin/llama-server ...`) before the next
   `fx-runner` invocation.
5. **Re-run `fx-runner` and confirm the new session has no
   `server_unavailable` event:**

   ```
   foundry-trace events-grep <new_session_id> \
       --pattern server_unavailable \
       --db logs/traces.db
   ```

   Presence on the new session means the fix did not take;
   absence confirms recovery. Re-run `foundry-kpis` and confirm
   `server_restart_count` is no longer incrementing.

Until recovery is confirmed, treat the affected window as
**infrastructure-degraded**: cycle-time KPI may be inflated by
restart latency even when model output quality is unchanged. Do not
attribute a cycle-time regression to a harness edit without ruling
out `server_restart_count > 0` first.

### Trace event reference {#server-unavailable-event-reference}

| Field | Type | Meaning |
|-------|------|---------|
| `kind` | `"server_unavailable"` | Event discriminant. Matches the entry in `CONTEXT.md` §Event kinds. |
| `payload.step` | `int` | Agent-loop step index at which the unhealthy `/health` was observed. |
| `payload.host` | `str` | Resolved value of `LLAMACPP_HOST` at probe time. |
| `payload.health_url` | `str` | Absolute `/health` URL probed (scheme + netloc + `/health`). |
| `payload.restart_attempted` | `bool` | `true` when the supervisor invoked `restart()` (i.e. `FOUNDRY_SERVER_AUTOSTART=1`). `false` when the manager is a passive prober. |

See `CONTEXT.md §Event kinds` for the full payload contract,
`docs/MODEL_CONFIG.md §Server supervision` for the env-var
reference, and `infra/llama-cpp/README.md` for the underlying
`llama-server` launch workflow.

## Glossary {#glossary}

| Term | Definition |
|------|------------|
| **Benchmark Task** | A standardised coding task marked `@pytest.mark.benchmark`; the Critic gates harness changes against the full benchmark suite. |
| **Critic** | The gatekeeper that runs the proposed edit through the pytest suite and benchmark suite, rejecting any edit that causes regressions. |
| **Digester** | The component that parses traces and produces a failure report naming what failed, where, and the candidate root cause. |
| **Evolver** | The meta-agent that takes a failure report and proposes a ProposedEdit against the harness. |
| **Failure Report** | A structured artifact produced by the Digester; consumed by the Evolver as the basis for a ProposedEdit. |
| **Foundry** | The Python package (`src/foundry_x/`) that wraps and evolves the harness. Distinct from the harness itself. |
| **FoundryAgent** | The runtime coding agent persona as defined in `harness/system_prompt.txt`. |
| **Harness** | The artifact being evolved: the system prompt, hooks, and skills. The harness is the product. |
| **Hook** | Middleware that runs around every tool call; lives in `harness/hooks/*.py`. |
| **Meta-agent** | An agent that operates on another agent's artifacts rather than on the end task. In FoundryX the Evolver is the meta-agent. |
| **Operator** | The human who runs the FoundryX harness against their own tasks. |
| **ProposedEdit** | A structured pydantic model representing a proposed change to the harness; the unit of work produced by the Evolver and consumed by the Critic. |
| **Regression Gate** | The Critic's benchmark gate. Every harness edit must pass the regression gate before it is marked active. |
| **Runner** | The component that drives a single agent session against a task, reads the harness, calls the model, and writes the trace. |
| **Trace** | A recorded sequence of events (prompts, tool calls, outcomes) produced by the Runner and persisted by the TraceLogger. |
| **TraceLogger** | The component that wraps an agent session and persists every prompt, tool call, and outcome to the trace store. |
