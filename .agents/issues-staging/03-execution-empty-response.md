## Motivation

In `run_task`, when the model stream produces zero payload deltas — no content, no tool_calls, and `finish_reason=None` — the branch at `runner.py:1775-1781` enters `if not response.tool_calls:` and falls through to `outcome_reason = "final_answer"` because `response.finish_reason not in (None, "stop")` evaluates False (None IS in the exclusion set). The `outcome_status` variable remains its default `None`, which the finally block at `runner.py:1904` coerces to `"success"`.

Result: a session where the model returned absolutely nothing is classified as a successful final answer. Because no `model_error` event is emitted, the Digester's `FAILURE_KINDS` first-failure walk never sees these sessions, so the evolution loop has no signal to act on and operators see a false success in trace output.

This is a distinct residual slice of #750 (which handled only non-None `finish_reason` values — `finish_reason=None` was explicitly left in the "OK" set). Causes include endpoint misconfiguration (max_tokens=0), content filters returning empty, premature connection close after status 200, or API quirks.

## Evidence

- `src/foundry_x/execution/runner.py:1775-1781` — empty-response path sets `outcome_reason='final_answer'`; `outcome_status` stays None (becomes "success" in finally at :1904)
- `src/foundry_x/execution/runner.py:1693-1709` — `model_error` event only emitted on `_consume_model_stream` exception; a normal return with an empty response produces no failure event
- `src/foundry_x/evolution/digester.py:61-88` — `FAILURE_KINDS` includes `"model_error"` at line 80
- `tests/execution/test_runner_outcome_preservation.py:373-405` — test covers the success path (`content='done'`, `finish_reason='stop'`) but NOT the empty-response (`content=None`, `finish_reason=None`) path
- `tests/execution/test_runner_stream.py:346-364` — tests `_consume_model_stream` in isolation (asserts `ttft_ms` is None, `chunk_count==0`) but does NOT test `run_task`'s outcome classification for this case

## Risk

Low. ~55-line diff. Reuses existing `model_error` kind (already in `FAILURE_KINDS` at digester.py:80) and existing `outcome.reason="model_error"` vocabulary (CONTEXT.md:192-194). Does not add a new event kind. The condition explicitly checks `finish_reason is None`, so `finish_reason='stop'` with empty content is unaffected.

## Acceptance Criteria

1. When `run_task` receives a response with `content=None/empty`, `tool_calls=[]`, and `finish_reason=None`, the outcome event has `status="failed"` and `reason="model_error"`
2. A `model_error` trace event with `error_type="EmptyResponse"` is emitted before the outcome event
3. Existing success path (`content="done"`, `finish_reason="stop"`) still produces `status="success"`, `reason="final_answer"`
4. Existing truncation path (`finish_reason="length"`, no tool_calls) still produces `status="truncated"`, `reason="length"`
5. `CONTEXT.md` `model_error` row updated to document "empty-response degenerate path" as a trigger condition
6. A test exercises a `_StreamingScriptedAdapter` yielding zero chunks through `run_task`, asserting `outcome.status=="failed"` and a `model_error` event in the trace

## ADR(s)

ADR-0010 — advances: §Termination semantics defines `outcome.status` as `success|truncated|failed`; this correctly populates "failed" for the degenerate empty-response case that currently silently lands in "success"
