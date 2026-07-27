# CONTEXT.md

> Glossary of terms used throughout FoundryX. Reading this is part of
> onboarding for both human and AI contributors (see
> [AGENTS.md](../AGENTS.md) section 1 and
> [PHILOSOPHY.md](./PHILOSOPHY.md)).
>
> This document is the source of truth for project vocabulary. If you
> introduce a new term, add it here in the same PR that introduces the
> concept.

## The product

- **FoundryX** — the framework as a whole: the runtime, the evolution
  loop, and the conventions that hold it together.
- **FoundryAgent** — the runtime coding agent persona as defined in the
  harness. Its persona and operating rules live in `harness/system_prompt.txt`.
  This is a *harness-layer term*: the src/foundry_x/ layer refers to "the
  agent" or "the runtime agent" without using this specific persona name.
- **Harness** — the artifact being evolved. Consists of the system
  prompt, hooks, and skills. Version-controlled, evolved by the
  `Evolver`, gated by the `Critic`. Per PHILOSOPHY.md §6, the harness
  is the product.
- **Foundry** — the Python package (`src/foundry_x/`) that wraps and
  evolves the harness. Built and maintained primarily by humans.

## Subsystems

- **TraceLogger** — wraps an agent session and persists every prompt,
  tool call, and outcome to a structured trace store. The ground-truth
  recorder. (`src/foundry_x/trace/logger.py`)
- **Runner** — drives a single agent session against a task. Reads the
  harness, calls the model, writes the trace.
  (`src/foundry_x/execution/runner.py`)
- **ModelAdapter** — the model-agnostic boundary used by the Runner to call
  OpenAI-compatible chat completion endpoints and normalize responses.
  (`src/foundry_x/execution/model_adapter.py`)
- **Digester** — parses a trace (or set of traces) and produces a
  failure report: what failed, where, and the candidate root cause.
  (`src/foundry_x/evolution/digester.py`)
- **Evolver** — a meta-agent (harness-layer role) that takes a failure report
  and proposes a `ProposedEdit` against the harness.
  (`src/foundry_x/evolution/evolver.py`)
- **Critic** — the gatekeeper. Runs the proposed edit through the
  pytest suite and benchmark suite; rejects regressions.
  (`src/foundry_x/evolution/critic.py`)
- **ProposedEdit** — a structured `pydantic` model representing a
  proposed change to the harness. The unit of work produced by the
  Evolver and consumed by the Critic.

## The loop

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

A single iteration is small. The value comes from running the loop
many times per day against a benchmark suite.

## Concepts

- **meta-agent** — an agent that operates on another agent's artifacts
  rather than on the end task; in FoundryX the `Evolver` is the
  meta-agent that turns failure reports into `ProposedEdit`s against
  the harness.  This is a *conceptual term* used in documentation; the
  src/foundry_x/ code uses "Evolver" as the proper class name.
- **failure report** — the structured artifact produced by the
  `Digester` from a trace, naming what failed, where, and the candidate
  root cause; consumed by the `Evolver` as the basis for a
  `ProposedEdit`.

## KPIs

The FoundryX KPIs are the project's definition of progress (PRD §5).
Canonical implementations live in `src/foundry_x/observability/kpis.py`.

The three PRD KPIs cannot move until the Runner and Critic are in place
(ADR-0010).

- **Cycle Time** — time from "Agent Failure" to "Harness Edit Proposal"
  (the operational proxy implemented in ``kpis.py`` measures
  ``task_received`` → ``critic_verdict``).
- **Regression Rate** — fraction of sessions with a `critic_verdict` in which
  a task previously seen in `passed_checks` later appears in `failed_checks`.
- **Improvement Rate** — fraction of `critic_verdict` events whose
  persisted payload has `approved: true`.

A fourth tracked metric is also exposed via `foundry-kpis` alongside the
three PRD KPIs:

- **Token Budget Hit Rate** — fraction of sessions that recorded at least
  one `task_aborted(reason="token_budget")` event. Signals whether the
  context-pruning hook is aggressive enough, or whether the model-context
  window is being misspent. The raw session count
  (`token_budget_abort_count`) is retained as an auxiliary signal.

## Event kinds

The vocabulary of `kind` values persisted by the `TraceLogger` onto
trace events. The `Digester` aligns its failure classifier against
these names (see `src/foundry_x/evolution/digester.py`). Adding a new
kind is a vocabulary change and must ship in the same PR as the code
that emits it.

The table below enumerates every `kind` value currently emitted by
the production code paths (Runner, Critic, hooks, and the trace store
itself), grouped by lifecycle phase. The "Failure signal?" column
marks kinds whose presence is itself an indication of failure; benign
kinds can still carry a failure signal in their payload (see
`FAILURE_PAYLOAD_KEYS` in `src/foundry_x/evolution/digester.py`).
The "Failure-signalling subset" subsection below cross-references the
`Digester`'s `FAILURE_KINDS` vocabulary.

### Session lifecycle

The events that bracket a single FoundryAgent task session. Every
`fx-runner --task "..."` invocation (see `Runner.main` in
`src/foundry_x/execution/runner.py`) opens with a `task_received` event
and closes with exactly one terminal marker: `task_completed` on the
success path or `task_failed` on any unhandled `Exception` (ADR-0007,
ADR-0010). This pair is the canonical FoundryAgent session-lifecycle
signal the Digester attributes a terminal status to; the rows below
pin their producer, payload contract, and failure-signal classification
(see `tests/docs/test_event_kinds_terminal_events.py`).

| Kind | Producer | Payload contract | Failure signal? |
| --- | --- | --- | --- |
| **`session_start`** | `TraceLogger.session` (`src/foundry_x/trace/logger.py`) | JSONL marker line `{"session_id", "started_at", "harness_version", "model_id", "metadata", "kind": "session_start"}`. In the sqlite backend the same data lives on the `sessions` row, not in the `events` table; the marker is part of the persisted vocabulary either way. ``metadata`` always contains ``harness_version_source`` whose value is ``"version_file"`` (VERSION file had content), ``"git_describe"`` (git described a tag), or ``"fallback"`` (fabricated ``"0.1.0"`` because VERSION was absent/blank and git failed or returned empty output). | no |
| **`session_end`** | `TraceLogger._end_session` (`src/foundry_x/trace/logger.py`) | JSONL marker line `{"session_id", "ended_at", "kind": "session_end"}`. In sqlite it updates `sessions.ended_at`. | no |
| **`task_received`** | `Runner.main` (`src/foundry_x/execution/runner.py`) | `{"prompt": str}` — the raw `--task` argument before the agent loop is opened. | no |
| **`task_completed`** | `Runner.main` (terminal, success path) | `{"duration_ms": int}` — wall-clock time of the entire `run_task` awaitable. | no |
| **`task_failed`** | `Runner.main` (terminal, exception path) | `{"error_type": str, "message": str, "duration_ms": int}` — exception class name, `str(exc)`, and wall-clock duration; stack frames are deliberately omitted to keep traces compact (ADR-0007). | **yes** (terminal) |
| **`task_aborted`** | `Runner.run_with_limits` (wall-clock cap, SECURITY.md "Runaway detection") | `{"reason": "wall_clock", "timeout_s": float \| null, "token_budget": int \| null}` — the cap that fired plus the active token budget at abort time. When `reason="token_budget"` the event contributes to the `token_budget_hit_rate` and `token_budget_abort_count` KPIs surfaced in the default `foundry-kpis` summary (issue #704). | **yes** (terminal) |

### Agent loop

| Kind | Producer | Payload contract | Failure signal? |
| --- | --- | --- | --- |
| **`user_prompt`** | `Runner.run_task` (issue #89, ADR-0010) | `{"content": str, "tool_count": int}` — the task as fed into the model plus the size of the tool surface the agent sees. | no |
| **`model_request`** | `Runner.run_task` (one per round-trip) | `{"step": int, "message_count": int, "tool_count": int}` — loop index, conversation length, and tool-surface size at request time. | no |
| **`model_response`** | `Runner.run_task` (one per round-trip) | `{"step": int, "finish_reason": str | null, "message": dict, "tool_calls": list[dict], "time_to_first_token_ms": int | null, "chunk_count": int, "total_stream_ms": int, "token_usage": dict | null, "tokens_used": int}` — the assistant message plus any tool calls the model emitted, plus streaming timing fields (issue #580). `time_to_first_token_ms` is measured from stream start to the first delta carrying text content OR a tool-call fragment — a tool-call-only response still counts as a payload delta (issue #905); the value is `null` only when the stream produced zero payload deltas of either kind. | no |
| **`model_error`** | `Runner.run_task` (on `adapter.complete` exception, and on the empty-response degenerate path) | `{"step": int, "error_type": str, "message": str}` — loop index plus exception class name and `str(exc)`. Paired with a `task_failed` terminal marker. Issue #931 added a second trigger: when the model stream returns no content, no tool calls, and `finish_reason=None` (e.g. `max_tokens=0`, content filter, premature connection close after HTTP 200), the Runner emits a `model_error` event with `error_type="EmptyResponse"` before classifying the outcome as `status="failed"`/`reason="model_error"`; the condition explicitly checks `finish_reason is None`, so `finish_reason="stop"` with empty content is unaffected. | **yes** |
| **`tool_call`** | `Runner.run_task` (exactly one per emitted tool call) | `{"step": int, "call_id": str, "name": str, "arguments": dict, "duration_ms": int, "hook_overhead_ms": int | null, "hook_post_overhead_ms": int | null}` — added in issue #173; per-tool-call latency for KPI slicing. `hook_overhead_ms` (issue #709) is wall-clock milliseconds spent in `HookRegistry.run_pre` before the tool executes; `null` when no hooks are registered. `hook_post_overhead_ms` (issue #903) is wall-clock milliseconds spent in `HookRegistry.run_post` after the tool executes (this includes the security-critical `InjectionFirewallHook` scan on the result before it is sent back to the model); `null` when no hooks are registered. Issue #893 consolidated the previous two-event emission (a pre-execution marker with `duration_ms=0` plus a post-execution event with the real duration) into this single event emitted after `run_post`, so per-call latency percentiles no longer see phantom zero-duration rows; the event carries the actual `duration_ms` plus both hook overheads. See ADR-0010 for the agent-loop structure. | no |
| **`tool_result`** | `Runner.run_task` (one per tool execution) | `{"step": int, "call_id": str, "name": str, "duration_ms": int, "output": Any \| null, "error": str \| null}` — non-null `error` flips the event onto the Digester's failure path via `FAILURE_PAYLOAD_KEYS`. | when `error` is non-null |
| **`outcome`** | `Runner.run_task` (always emitted in `finally`) | `{"status": "success" \| "truncated" \| "failed", "reason": "final_answer" \| "model_error" \| "max_steps", "steps": int}` — terminal status the Digester attributes to the session. | when `status == "failed"` |
| **`hook_registry_error`** | `Runner.run_task` via `_resolve_hook_registry` (issue #260) | `{"error_type": str, "message": str}` — emitted when `harness.hooks.get_registry()` raises after a successful lazy import. The session continues in degraded mode (`registry is None`, so no hook fan-out including the `InjectionFirewallHook`), but the event records that the firewall layer is off so the Digester and operator have a signal (AGENTS.md §2). | **yes** (security-critical hooks disabled) |
| **`server_unavailable`** | `Runner.run_task` via `_handle_server_unavailable` (issue #899) | `{"step": int, "host": str, "health_url": str, "restart_attempted": bool}` — emitted when the `FoundryServerManager` reports `GET /health` returning a non-200 status mid-session and triggers the supervisor's bounded exponential-backoff restart loop. The `restart_count` attribute on the manager (and the `server_restart_count` KPI consumer in `foundry_x.observability.kpis`) tracks actual successful supervisor recoveries, not detection count. The session continues only if `restart()` re-establishes `/health` within the configured retry budget. | **yes** (infrastructure) |
| **`model_response_chunk`** | `Runner._consume_model_stream` (issue #199) | `{"step": int, "delta_index": int, "content_so_far": str, "chunk_duration_ms": int}` — one event per SSE delta received from `adapter.stream()`, emitted via `ModelResponseChunkEvent` (`src/foundry_x/execution/runner.py:168`). `delta_index` is the zero-based chunk ordinal within the step; `chunk_duration_ms` is wall-clock milliseconds since the previous chunk (or stream start for delta 0). Enables KPI consumers to split model latency (time-to-first-token) from network latency (inter-chunk gaps) without waiting for the terminal `model_response`. Per-chunk events are excluded from the event-limit accounting so streaming telemetry cannot starve the loop budget (issue #790). | no |
| **`model_retry`** | `Runner.run_task` via `_on_retry` closure wired onto `OpenAICompatibleAdapter.on_retry` (issue #200) | `{"attempt": int, "error_type": str, "backoff_ms": int}` — 1-based index of the failed attempt, exception class name, and jittered backoff (ms) before the next attempt. Emitted via `ModelRetryEvent` (`src/foundry_x/execution/model_adapter.py:217`). Only `OpenAICompatibleAdapter` has retry logic; injected fakes/stubs are left untouched. If all retries are exhausted the terminal `model_error` event fires. The aggregate count is surfaced as the `model_retry_count` auxiliary KPI. | no |
| **`token_usage_missing`** | `Runner.run_task` (issue #580) | `{"step": int, "message": str}` — emitted when `response.usage` is `None`, i.e. the endpoint omitted the wire-level `usage` object. The runner counts zero tokens for that step and emits this event so the gap is observable without crashing the loop. The `model_response` event for the same step carries `token_usage: null` and `tokens_used` unchanged. | no |
| **`tool_argument_parse_error`** | `Runner.run_task` (issue #261, #872) | `{"step": int, "call_id": str, "name": str, "raw": str, "error": str}` — emitted when `_parse_tool_arguments` cannot JSON-decode the model's tool-call arguments string. `raw` is the unparsed arguments string; `error` is the parse error message (always non-null when the event is emitted). The aggregate count is surfaced as the `tool_argument_parse_error_count` auxiliary KPI. | when `error` is non-null (always; routed via `FAILURE_PAYLOAD_KEYS`) |

### Hooks

| Kind | Producer | Payload contract | Failure signal? |
| --- | --- | --- | --- |
| **`injection_blocked`** | `InjectionFirewallHook` (`harness/hooks/injection_firewall.py`, one per block) | `{"markers": list[str], "tool": str, "preview": str}` — sorted unique marker names, originating tool name, and the first 120 characters of the suppressed text with newlines folded to spaces (safe to persist; never re-injected into a prompt). The Digester aggregates every block in a session into one `FailureReport` with `proposed_class == 'injection-attempt'` and one entry per block in `failed_steps` so the Evolver sees the full adversarial surface. See issue #120. | **yes** (adversarial) |
| **`firewall_exception`** | `InjectionFirewallHook` (`harness/hooks/injection_firewall.py`, one per block) | `{"hook_name": str, "pattern_matched": str, "risk_score": int}` — hook class name, the aggregated marker name(s) that triggered the block, and a severity-weighted risk score (role-tag and unicode-confusable patterns score 2; instruction-override patterns score 1). Emitted alongside ``injection_blocked`` on every block so firewall events are queryable via ``foundry-x-trace`` independently of the Digester aggregation path. See issue #823. | **yes** (adversarial) |
| **`context_pruned`** | `ContextPruningHook` (event-count) and `TokenAwarePruningHook` (token-aware) (`harness/hooks/context_pruning.py`, opt-in via `harness/manifest.json`; ADR-0021) | Two payload shapes depending on which pruner is active. **Event-count pruner** (fires when `FOUNDRY_CONTEXT_TOKENS` is unset; issue #106, issue #491): `{"dropped": int, "threshold": int, "token_threshold": int}` — number of older non-preserved events dropped to bring the session back under the per-session event cap (`threshold`); `token_threshold` echoes the hook's config value from `harness/manifest.json` (`context_pruning.token_threshold`, default 8192). **Token-aware pruner** (fires when `FOUNDRY_CONTEXT_TOKENS` is set to a positive int; issue #465, ADR-0021): `{"dropped": int, "threshold_tokens": int, "session_tokens": int}` — `threshold_tokens` is the configured token ceiling and `session_tokens` is the cumulative `tokens_used` observed at prune time. Note: `token_threshold` is a config key in `harness/manifest.json` and a config-echo field in the event-count payload only; it is **not** present in the token-aware payload, so operators querying token-aware sessions for `token_threshold` get empty results — query `threshold_tokens`/`session_tokens` instead. | no |
| **`fetch_blocked`** | `WebFetchHook` (`harness/hooks/web_fetch.py`, one per block) | `{"url": str, "reason": "domain_not_in_allowlist", "allowed_domains": list[str]}` — the requested URL, the block reason, and the sorted set of domains the operator configured in `FETCH_ALLOWED_DOMAINS`. Emitted when the hook's `pre_tool` slot detects a `web_fetch` tool call whose URL host is not in the allowlist (or the allowlist is empty). The hook then clears the URL in the call arguments so the skill executor returns an error without issuing the HTTP request. See issue #1054. | **yes** (security policy enforcement) |
| **`fetch_success`** | `Runner.run_task` via `_execute_skill` (issue #1054) | `{"url": str, "status_code": int, "bytes_returned": int}` — the fetched URL, HTTP status code, and number of bytes returned in the response body. Emitted after a successful `web_fetch` skill execution (no error in the result). The `fetch_success` event is benign; a non-200 `status_code` is observable but does not by itself flip the event onto the Digester's failure path. | no |

### Critic pipeline

| Kind | Producer | Payload contract | Failure signal? |
| --- | --- | --- | --- |
| **`critic_verdict`** | `record_verdict` (`src/foundry_x/observability/regression_report.py`, constant `VERDICT_KIND = "critic_verdict"`) | `{"approved": bool, "passed_checks": list[str], "failed_checks": list[str], "notes": str}` — the persisted `VerdictRecord` shape (ADR-0006 boundary model). KPI and regression-report consumers reach this kind via `logger.iter_events(sid, kind="critic_verdict")`. | no (verdict is a structured summary, not a failure in itself; downstream regressions are derived from `failed_checks` history) |

### Failure-signalling subset

The `Digester` recognises a *failure signal* in two ways: the event's
`kind` is in `FAILURE_KINDS`, **or** the event's payload contains a
key from `FAILURE_PAYLOAD_KEYS`. The vocabulary below is the closed
set the Digester considers structural failure markers; treat it as a
subset of the broader kind vocabulary above.

- **`FAILURE_KINDS`** (constant in
  `src/foundry_x/evolution/digester.py:62-89`): `tool_error`,
  `task_failed`, `task_aborted`, `run_failed`, `agent_error`, `error`,
  `model_error`, `hook_registry_error`.
  `task_failed` and `task_aborted` are emitted by the production Runner:
  `task_failed` when the agent loop raises an exception, and
  `task_aborted` when the wall-clock cap fires (reason=`wall_clock`)
  or the token budget is exceeded (reason=`token_budget`). The remaining
  four are reserved vocabulary recognized by the Digester for
   compatibility with legacy producers and tests. `model_error` (issue
   #867) is emitted by the Runner when `adapter.complete` raises, paired
   with `outcome.reason="model_error"`; issue #931 added a second trigger
   on the empty-response degenerate path (no content, no tool calls,
   `finish_reason=None`) — the Runner emits `model_error` with
   `error_type="EmptyResponse"` and classifies the outcome as
   `status="failed"`/`reason="model_error"` so the first-failure walk sees
   sessions that previously silently landed in `"success"`;
   `hook_registry_error` (issue
  #867) is emitted by `Runner._resolve_hook_registry` when
  `harness.hooks.get_registry()` raises after a successful lazy import,
  leaving the session with every hook (including the
  `InjectionFirewallHook`) silently disabled. Without these two kinds
  in the set, the Digester's first-failure walk reports the later
  downstream failure as the root cause and the Evolver can miss the
  real signal (model fault or security-degraded session). Adding a new
  value here is a vocabulary change and must ship with both a producer
  and a regression test (ADR-0004).
- **`FAILURE_PAYLOAD_KEYS`** (constant in
  `src/foundry_x/evolution/digester.py:91-97`): `error`, `traceback`,
  `exception`. A `tool_result` whose payload has any of these keys is
  classified as a failure even though its `kind` is benign — the
  signal is on the payload, not on the kind. The same payload-key
  rule applies to every other kind in the table above.
- **`injection_blocked`** is *not* in `FAILURE_KINDS`: it is handled
  by a dedicated aggregation pass in `Digester.digest` that collects
  *every* block in the session (not just the first), so the generic
  first-failure walk would under-report. It is exposed as a separate
  constant `INJECTION_BLOCKED_KIND` so tests can pin the contract.
- **`tool_argument_parse_error`** is *not* in `FAILURE_KINDS`: its
  payload always carries a non-null `error` key, so it is already
  caught by `FAILURE_PAYLOAD_KEYS` and routed onto the Digester's
  failure path without needing a kind-level entry. Adding the kind to
  `FAILURE_KINDS` would add no new classification capability; it is
  documented in the Agent loop table for discoverability (issue #1049).

- **`model_response`** — emitted by `run_task` for every chat-completion
  round-trip the runner performs. Payload contract (issue #191, issue #580):
  `{"step": int, "finish_reason": str | null, "message": <serialized
  ModelMessage>, "tool_calls": list[dict], "time_to_first_token_ms": int | null,
  "chunk_count": int, "total_stream_ms": int, "token_usage": dict | null,
  "tokens_used": int}`.
  The `time_to_first_token_ms` field is the wall-clock milliseconds from
  stream start to the first content delta; `null` when the stream produced
  no payload. `chunk_count` is the number of SSE deltas received.
  `total_stream_ms` is the wall-clock duration of the entire stream.
  The `token_usage` field carries `{"prompt_tokens": int,
  "completion_tokens": int, "total_tokens": int}` when the
  `OpenAICompatibleAdapter` surfaces the wire-level `usage` object on
  the response, or `null` when the endpoint omits it. The runner also
  emits a `RuntimeWarning` on missing telemetry so the gap is
  observable in operator logs without crashing the loop. The Phase 3
  `Digester` reads `token_usage` to compute per-step token deltas, and
  the PRD "Improvement Rate" KPI uses the same field to attribute the
  cost of a harness edit. The timing fields (`time_to_first_token_ms`,
  `total_stream_ms`, `chunk_count`) enable the KPI's Improvement Rate to
  attribute slow responses to network latency vs. model generation time.
  Consumers MUST treat `null` as missing data, not as a zero reading.

## Artifacts on disk

- `harness/system_prompt.txt` — the agent's persona and operating rules.
- `harness/hooks/*.py` — middleware that runs around every tool call.
- `harness/skills/*.json` — tool definitions the agent can invoke.
- `logs/` — trace store (gitignored). Per-run SQLite databases plus
  exports. Manage unbounded growth with `foundry-trace prune` (issue #275):
  `--keep-last N` retains the N most recent sessions, `--older-than DAYS`
  drops aged ones; both support `--dry-run` and work on sqlite/jsonl.
  Add `--vacuum` (sqlite, issue #896) on a retention pass to reclaim the
  `logs/traces.db-wal` sidecar that heavy pruning otherwise grows
  unboundedly.
- `benchmarks/` — pytest-marked tasks the Critic uses to gate harness
  changes.
- `docs/adr/` — recorded architecture decisions, numbered sequentially.
- `docs/ideas/` — retired (issue #645). Was: design ideas not yet accepted.

## Roles

- **Operator** — the human running the harness against their tasks.
- **Engineer** — the human maintaining `src/foundry_x/` (the foundry).
- **Agent** — an AI collaborator (e.g., Claude, GPT, or a local
  model driven by FoundryX itself).
- **Critic** — see above; in role terms, also the regression-tester.

The Operator and Engineer may be the same person. The Agent is always
external to the runtime under test.

## Verbs

These verbs structure the workflow described in `AGENTS.md` §3 and
`CONTRIBUTING.md`. Use them in PR titles and commit bodies when
relevant.

- **Observe** — read traces and existing code before proposing.
- **Digest** — turn observations into a failure report.
- **Propose** — produce a `ProposedEdit` (or, for a human, a PR).
- **Evaluate** — run the test + benchmark gate.
- **Commit** — atomic, conventional-commits change.
- **Hand off** — open a PR and wait for review.
