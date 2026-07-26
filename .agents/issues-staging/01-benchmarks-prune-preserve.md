## Motivation

`_PRESERVE_KINDS = frozenset({"tool_result", "user_prompt"})` (`harness/hooks/context_pruning.py:68`) is the invariant that prevents the context-pruning hooks from dropping the user's task description and tool execution results. Both `ContextPruningHook` (line 227) and `TokenAwarePruningHook` (line 383) pass this set to the pruner's `_drop` function, which builds a SQL `NOT IN (...)` clause to exclude those kinds.

The Evolver can edit `harness/hooks/context_pruning.py` (it is harness DNA). If an Evolver-produced diff removes `tool_result` or `user_prompt` from `_PRESERVE_KINDS`, the pruner would start dropping those critical events, silently degrading agent context.

No benchmark catches this regression. Both existing pruning benchmarks (`test_context_pruning_benchmark.py` #618 and `test_token_aware_pruning_benchmark.py` #732) plant ONLY non-preserved event kinds and assert `dropped >= 1`. Their `_plant` helpers explicitly document: "None of the planted events use tool_result or user_prompt, so every planted event is eligible for pruning." They test the DROP path but never the PRESERVE path.

A grep for `_PRESERVE_KINDS` across `tests/` and `benchmarks/` returns zero matches — the preservation invariant has no test coverage.

## Evidence

- `harness/hooks/context_pruning.py:68` — `_PRESERVE_KINDS = frozenset({"tool_result", "user_prompt"})`
- `harness/hooks/context_pruning.py:227` — `ContextPruningHook.pre_tool` passes `_PRESERVE_KINDS` to pruner
- `harness/hooks/context_pruning.py:383` — `TokenAwarePruningHook.pre_tool` passes `_PRESERVE_KINDS` to pruner
- `benchmarks/tasks/test_context_pruning_benchmark.py` — `_plant` docstring: "None of the planted events use tool_result or user_prompt" — only tests drop path
- `grep -rn '_PRESERVE_KINDS' tests/ benchmarks/` → zero matches

## Risk

Low. New benchmark task only; no production code change. Mirrors existing `_sqlite_pruner` pattern from #618.

## Acceptance Criteria

1. After pruning fires, events with `kind='tool_result'` are present in the trace DB
2. After pruning fires, events with `kind='user_prompt'` are present in the trace DB
3. `dropped` count > 0 (pruning actually fired)
4. Noise events (e.g. `model_request`) are reduced by the dropped count
5. Task has `@pytest.mark.benchmark` and a matching fixture directory (passes `test_hygiene.py`)
6. `uv run pytest benchmarks/tasks/test_prune_preserves_kinds.py -m benchmark` passes

## ADR(s)

ADR-0005 — advances (pytest-based regression target per the unified-eval contract)
ADR-0004 — advances (Critic gate gains a regression target for the preservation invariant)
