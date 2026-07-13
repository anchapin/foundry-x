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
- **FoundryAgent** — the runtime coding agent wrapped by the harness.
  Its persona and operating rules live in `harness/system_prompt.txt`.
  This is a **harness-layer term**: the ``src/foundry_x/`` library does not
  reference ``FoundryAgent``; the term belongs to the artifact under
  evolution, not the foundry code that drives it.
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
  the harness.  This is a **harness-layer term**: the concept names a
  role within the artifact under evolution, not a class or function
  inside ``src/foundry_x/``.
- **failure report** — the structured artifact produced by the
  `Digester` from a trace, naming what failed, where, and the candidate
  root cause; consumed by the `Evolver` as the basis for a
  `ProposedEdit`.

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

| Kind | Producer | Payload contract | Failure signal? |
| --- | --- | --- | --- |
| **`session_start`** | `TraceLogger.session` (`src/foundry_x/trace/logger.py`) | JSONL marker line `{"session_id", "started_at", "harness_version", "model_id", "metadata", "kind": "session_start"}`. In the sqlite backend the same data lives on the `sessions` row, not in the `events` table; the marker is part of the persisted vocabulary either way. | no |
| **`session_end`** | `TraceLogger._end_session` (`src/foundry_x/trace/logger.py`) | JSONL marker line `{"session_id", "ended_at", "kind": "session_end"}`. In sqlite it updates `sessions.ended_at`. | no |
| **`task_received`** | `Runner.main` (`src/foundry_x/execution/runner.py`) | `{"prompt": str}` — the raw `--task` argument before the agent loop is opened. | no |
| **`task_completed`** | `Runner.main` (terminal, success path) | `{"duration_ms": int}` — wall-clock time of the entire `run_task` awaitable. | no |
| **`task_failed`** | `Runner.main` (terminal, exception path) | `{"error_type": str, "message": str, "duration_ms": int}` — exception class name, `str(exc)`, and wall-clock duration; stack frames are deliberately omitted to keep traces compact (ADR-0007). | **yes** (terminal) |
| **`task_aborted`** | `Runner.run_with_limits` (wall-clock cap, SECURITY.md "Runaway detection") | `{"reason": "wall_clock", "timeout_s": float \| null, "token_budget": int \| null}` — the cap that fired plus the active token budget at abort time. | **yes** (terminal) |

### Agent loop

| Kind | Producer | Payload contract | Failure signal? |
| --- | --- | --- | --- |
| **`user_prompt`** | `Runner.run_task` (issue #89, ADR-0010) | `{"content": str, "tool_count": int}` — the task as fed into the model plus the size of the tool surface the agent sees. | no |
| **`model_request`** | `Runner.run_task` (one per round-trip) | `{"step": int, "message_count": int, "tool_count": int}` — loop index, conversation length, and tool-surface size at request time. | no |
| **`model_response`** | `Runner.run_task` (one per round-trip) | `{"step": int, "finish_reason": str, "message": dict, "tool_calls": list[dict]}` — the assistant message plus any tool calls the model emitted. | no |
| **`model_error`** | `Runner.run_task` (on `adapter.complete` exception) | `{"step": int, "error_type": str, "message": str}` — loop index plus exception class name and `str(exc)`. Paired with a `task_failed` terminal marker. | **yes** |
| **`tool_call`** | `Runner.run_task` (one per emitted tool call) | `{"step": int, "call_id": str, "name": str, "arguments": dict, "duration_ms": int}` — added in issue #173; per-tool-call latency for KPI slicing. | no |
| **`tool_result`** | `Runner.run_task` (one per tool execution) | `{"step": int, "call_id": str, "name": str, "duration_ms": int, "output": Any \| null, "error": str \| null}` — non-null `error` flips the event onto the Digester's failure path via `FAILURE_PAYLOAD_KEYS`. | when `error` is non-null |
| **`outcome`** | `Runner.run_task` (always emitted in `finally`) | `{"status": "success" \| "truncated" \| "failed", "reason": "final_answer" \| "model_error" \| "max_steps", "steps": int}` — terminal status the Digester attributes to the session. | when `status == "failed"` |
| **`hook_registry_error`** | `Runner.run_task` via `_resolve_hook_registry` (issue #260) | `{"error_type": str, "message": str}` — emitted when `harness.hooks.get_registry()` raises after a successful lazy import. The session continues in degraded mode (`registry is None`, so no hook fan-out including the `InjectionFirewallHook`), but the event records that the firewall layer is off so the Digester and operator have a signal (AGENTS.md §2). | **yes** (security-critical hooks disabled) |

### Hooks

| Kind | Producer | Payload contract | Failure signal? |
| --- | --- | --- | --- |
| **`injection_blocked`** | `InjectionFirewallHook` (`harness/hooks/injection_firewall.py`, one per block) | `{"markers": list[str], "tool": str, "preview": str}` — sorted unique marker names, originating tool name, and the first 120 characters of the suppressed text with newlines folded to spaces (safe to persist; never re-injected into a prompt). The Digester aggregates every block in a session into one `FailureReport` with `proposed_class == 'injection-attempt'` and one entry per block in `failed_steps` so the Evolver sees the full adversarial surface. See issue #120. | **yes** (adversarial) |
| **`context_pruned`** | `ContextPruningHook` (`harness/hooks/context_pruning.py`, opt-in via `harness/manifest.json`) | `{"dropped": int, "threshold": int}` — number of older events the pruner dropped to bring the session back under the per-session cap, plus the threshold in effect. See issue #106. | no |

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
  `src/foundry_x/evolution/digester.py:60-68`): `tool_error`,
  `task_failed`, `run_failed`, `agent_error`, `error`. Of these,
  `task_failed` is the only kind currently emitted by the production
  Runner; the remaining four are reserved vocabulary recognized by
  the Digester for compatibility with legacy producers and tests.
  Adding a new value here is a vocabulary change and must ship with
  both a producer and a regression test (ADR-0004).
- **`FAILURE_PAYLOAD_KEYS`** (constant in
  `src/foundry_x/evolution/digester.py:70-76`): `error`, `traceback`,
  `exception`. A `tool_result` whose payload has any of these keys is
  classified as a failure even though its `kind` is benign — the
  signal is on the payload, not on the kind. The same payload-key
  rule applies to every other kind in the table above.
- **`injection_blocked`** is *not* in `FAILURE_KINDS`: it is handled
  by a dedicated aggregation pass in `Digester.digest` that collects
  *every* block in the session (not just the first), so the generic
  first-failure walk would under-report. It is exposed as a separate
  constant `INJECTION_BLOCKED_KIND` so tests can pin the contract.

- **`model_response`** — emitted by `run_task` for every chat-completion
  round-trip the runner performs. Payload contract (issue #191):
  `{"step": int, "finish_reason": str | null, "message": <serialized
  ModelMessage>, "tool_calls": list[dict], "token_usage": dict | null}`.
  The `token_usage` field carries `{"prompt_tokens": int,
  "completion_tokens": int, "total_tokens": int}` when the
  `OpenAICompatibleAdapter` surfaces the wire-level `usage` object on
  the response, or `null` when the endpoint omits it. The runner also
  emits a `RuntimeWarning` on missing telemetry so the gap is
  observable in operator logs without crashing the loop. The Phase 3
  `Digester` reads `token_usage` to compute per-step token deltas, and
  the PRD "Improvement Rate" KPI uses the same field to attribute the
  cost of a harness edit. Consumers MUST treat `null` as missing data,
  not as a zero reading.

## Artifacts on disk

- `harness/system_prompt.txt` — the agent's persona and operating rules.
- `harness/hooks/*.py` — middleware that runs around every tool call.
- `harness/skills/*.json` — tool definitions the agent can invoke.
- `logs/` — trace store (gitignored). Per-run SQLite databases plus
  exports. Manage unbounded growth with `foundry-trace prune` (issue #275):
  `--keep-last N` retains the N most recent sessions, `--older-than DAYS`
  drops aged ones; both support `--dry-run` and work on sqlite/jsonl.
- `benchmarks/` — pytest-marked tasks the Critic uses to gate harness
  changes.
- `docs/adr/` — recorded architecture decisions, numbered sequentially.
- `docs/ideas/` — design ideas not yet accepted into the project.

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
