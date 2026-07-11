# ADR-0010: Runner agent loop

## Status

Accepted. 2026-07-11.

## Context

`docs/ROADMAP.md` Phase 1 names the Runner as the "bridge between the agent
and the LLM," and `docs/CONTEXT.md` defines it as "drives a single agent
session against a task." Phase 2 (the evolution loop — Digester → Evolver →
Critic) consumes traces as ground truth, so the Runner must produce traces
the Digester can reason about. Three PRD KPIs cannot move until the Runner
emits real traces: cycle time has no failure to measure, regression rate has
no benchmark run to regress, improvement rate has no before/after to compare
(PRD §5).

The harness exposes three concerns the Runner must integrate:

1. **Tool surface.** The harness advertises what the model may call through
   `harness/skills/*.json` documents (issue #103, #104, #105, #168 — `bash`,
   `edit_file`, `write_file`, `list_dir`, `grep_search`, `read_file`). The
   Runner is the layer that turns those documents into the OpenAI-compatible
   `tools` array the model client sends.
2. **Hook fan-out.** `harness/hooks/base.py:62-76` exposes
   `HookRegistry.run_pre` (may rewrite the call) and `HookRegistry.run_post`
   (may rewrite the result). Per `docs/SECURITY.md` "Prompt-input firewall"
   and "Prompt injection" the `InjectionFirewallHook`
   (`harness/hooks/injection_firewall.py`) is the second-layer defense: every
   `ToolResult.output` must be scanned for adversarial markers **before** it
   is sent back to the model in the next turn.
3. **Termination.** The Runner must produce a terminal `outcome` event
   so the Digester has a reason to attribute success vs. truncation
   vs. failure (ADR-0007 — traces carry observable behavior). Without
   it, a session can be `task_completed` and still be a runaway loop the
   critic needs to flag.

The current `run_task` (`src/foundry_x/execution/runner.py:284-326`) is a
single-turn stub: it sends the user prompt and records `model_request` /
`model_response` events with `tools=[]`, then returns. There is no loop, no
skill surface, no hook fan-out. The integration is the missing link.

The Phase 1 milestones end at "Build the TraceLogger" — the Runner is a
prerequisite for Phase 2 but has no dedicated ADR yet (`docs/adr/` lacks a
`0010-runner-*` entry; issue #89 specifically requests one). The Critic gate
(ADR-0004) demands observable failure modes; a hand-rolled loop without an
ADR would foreclose the `tool_call` / `tool_result` event vocabulary the
Digester needs.

## Decision

The Runner implements an asyncio-based agent loop on top of the existing
`ModelAdapter` protocol (`src/foundry_x/execution/model_adapter.py`). For
each turn the Runner:

1. Reads `system_prompt.txt` and `harness/skills/*.json` from the
   resolved `harness_dir`, constructs the `ToolDefinition` surface
   (`ModelAdapter.complete(messages, tools=...)`), and records a
   `user_prompt` trace event (the harness-level marker; the
   `task_received` lifecycle event in `main()` remains untouched so
   existing terminal-event tests keep passing).
2. Sends the initial chat completion to the model adapter and
   records a `model_request` event.
3. On each response:
   - Append the assistant message (with `tool_calls`) to the running
     `messages` list so the next turn carries context.
   - If `response.tool_calls` is empty (the model returned a final
     assistant message), record `model_response`, terminate the loop
     with `outcome.status="success"` and `outcome.reason="final_answer"`.
   - For each `tool_call` in the response:
     1. Record a `tool_call` trace event with `name`, `arguments`
        (parsed JSON from the wire-format string), `call_id`, and the
        current step index.
     2. Fan through `HookRegistry.run_pre(call)` so `pre_tool`
        hooks can rewrite arguments (e.g. the prompt-injection firewall
        in `#5`, or a future rate-limiting hook).
     3. Execute the skill handler. The Runner carries a default
        `skill_executor(name, arguments) -> dict` that returns a
        stub `{"status": "ok", "skill": <name>}` envelope so the loop
        closes; a future PR hooks real `subprocess.run`-backed
        executors per skill via the same protocol. Tests inject a
        real executor through the `skill_executor=` keyword.
     4. Record a `tool_result` trace event with `name`, `call_id`,
        `duration_ms`, `output`, and `error` (None on success).
     5. Fan through `HookRegistry.run_post(call, result)` so
        `post_tool` hooks can rewrite the result **before** it lands
        back in the prompt — the security-critical position of the
        injection firewall.
     6. Append a `role="tool"` `ModelMessage` carrying the result so
        the next `complete()` call carries the result back to the
        model.
   - Increment the step counter; if it reaches `max_steps` (env
     `FOUNDRY_MAX_AGENT_STEPS`, default 16), terminate with
     `outcome.status="truncated"` and `outcome.reason="max_steps"`.
   - The wall-clock cap from `RunLimits.task_timeout_s`
     (`run_with_limits` in `runner.py:253`) is the outer guardrail
     (SECURITY.md "Runaway detection"); exceeding it aborts the
     session regardless of the loop state.
4. Records a single `outcome` trace event when the loop exits, with
   `status` (success | truncated | failed), `reason`
   (final_answer | max_steps | model_error), and `steps` (int) so the
   Digester can bucket sessions.

The Runner does **not** introduce a `harness_layout.py` or rename
`task_received`; both belong to other in-flight issues (#90, lifecycle
events respectively). The skill loading stays inline in `runner.py` to
respect AGENTS.md §2 ("Never widen scope").

## Consequences

- **Trace vocabulary** gains `user_prompt`, `tool_call`, `tool_result`,
  `outcome` events. Each is rendered with the redaction pipeline in
  `TraceLogger.record` (SECURITY.md §Secrets), so model output cannot
  leak raw credentials into the Digester's view. The evolution loop
  in Phase 2 sees the precise shape it needs: which tool was called,
  what arguments, what result, and a terminal reason.
- **Tool surface** is data-driven: a new `harness/skills/<name>.json`
  lands automatically; the existing `tests/harness/test_skills_load.py`
  schema checks remain the gate. A change that adds a skill with no
  JSON-schema fields surfaces as a model-validation failure, not a
  silent capability regression.
- **Hook fan-out** is mandatory per call: every tool call routes through
  `HookRegistry.run_pre` and `HookRegistry.run_post`. The
  InjectionFirewallHook self-registers on `import harness.hooks`
  (`harness/hooks/__init__.py`), so the firewall runs by default
  without test plumbing. The hook isolation contract
  (`harness/hooks/base.py:78-115`) — one buggy hook does not abort the
  chain — remains in force; a pre- or post-tool failure is logged
  through `HookRegistry._isolate_failure` and routed to the
  `on_error` callback the Runner installs (the
  `task_aborted`-shaped emission is the bridge to the trace store).
- **Critic gating (ADR-0004).** The Critic gate now has a concrete
  surface to gate against: a harness edit that disables the
  `run_pre` / `run_post` fan-out, drops `outcome` recording, or
  removes `max_steps` bounds all become observable regressions the
  integration test in `tests/test_execution.py` catches. The Critic
  benchmark family in `docs/adr/0009-security-evals-benchmark-family.md`
  is supplemented by a new task that drives `main()` with a stub
  ModelAdapter and asserts the event sequence
  `user_prompt → tool_call → tool_result → outcome`.
- **Termination semantics.** Three terminal reasons: `final_answer`
  (the model gave up tool calls), `max_steps` (loop cap reached —
  bounds blow-the-context-window risk on a degenerate harness),
  `model_error` (the adapter raised — `run_with_limits` still emits
  the wall-clock variant). `outcome.status` is `"success"` for the
  first, `"truncated"` for the second, `"failed"` for the third;
  each lands in the same `outcome` event so the Digester has one
  payload schema to read.
- **Out of scope.** `harness_layout.py` (#90) and real skill
  executors (the bash subprocess hook) stay in their own proposals.
  This ADR touches only the loop, its trace events, and the trace
  ordering; it deliberately stops short of "make `bash` actually run
  `subprocess.run`" — that's the next layer the Criti gate evaluates
  independently.
