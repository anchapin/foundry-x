# ADR-0009: Security-evals BenchmarkTask family

## Status

Accepted. 2026-07-10.

## Context

[ADR-0004](0004-self-modification-guardrails.md) makes every harness edit
gated by the `Critic`, and [ADR-0005](0005-pytest-as-evaluation-framework.md)
makes the Critic's pass/fail signal come from `pytest` tasks marked
`@pytest.mark.benchmark`. The existing `benchmarks/tasks/` suite
(`test_fix_syntax_error.py`, `test_sort_a_list.py`,
`test_nth_fibonacci.py`, `test_reverse_string.py`,
`test_write_unit_test.py`, `test_stop_after_two_failures.py`) is entirely
coding-problem shaped: it exercises the agent's general capability but
says nothing about whether a harness edit re-introduced a security
regression. A change to `harness/hooks/base.py` that drops the
hook-failure isolation, or to `src/foundry_x/trace/logger.py` that
weakens the secret scrubber, or to `src/foundry_x/evolution/evolver.py`
that loosens the rate limit, would still pass every existing benchmark
and silently ship a regression.

`docs/SECURITY.md` §"Scope of this document" states the gap explicitly:
"The `Critic` itself should be extended to include security-relevant
test cases; track that work as an ADR." This ADR is that record, and it
supersedes the placeholder sentence in that section.

Each control we gate has a concrete surface today (otherwise a benchmark
target would be aspirational and the PR body would have to flag it):

- **Secret redaction on trace payloads** is implemented by
  `src/foundry_x/trace/logger.py` (`_redact` / `_redact_value`, called
  from both `TraceLogger.record` and `TraceLogger.session`'s
  `metadata` path; issues #3 and #121).
- **Prompt-injection blocking on tool outputs** is implemented by
  `harness/hooks/injection_firewall.py` (`InjectionFirewallHook` on
  the `post_tool` slot, with the `scan_for_injection` pure function as
  the Evolver-tunable surface; issues #5 and #122).
- **Hook-failure isolation** is implemented by
  `harness/hooks/base.py` (`HookRegistry.run_pre` / `run_post` route
  exceptions through `_isolate_failure`; issue #21).
- **Evolver rate-limit and `ProposedEdit` target confinement** is
  implemented by `src/foundry_x/evolution/evolver.py` (`Evolver._check_rate_limit`,
  `ProposedEdit._target_file_within_harness_tree`, `EvolverGuardError`).

## Decision

We seed four security-evals `BenchmarkTask` families under
`benchmarks/tasks/` (per ADR-0006, each declares a pydantic
`BenchmarkTask` with `tags` including `'security'` and
`difficulty_tier='medium'`), each carrying `@pytest.mark.benchmark` so
the Critic gate (ADR-0004) catches a regression:

- `benchmarks/tasks/test_secret_redaction_evals.py` — control:
  `TraceLogger` redacts `sk-...`, PEM blocks, and the modern-token set
  from event payloads and the `session()` metadata dict
  ([SECURITY.md §Secrets](../SECURITY.md), [ADR-0003](0003-sqlite-as-trace-store.md),
  issues #3 and #121).
- `benchmarks/tasks/test_injection_firewall_evals.py` — control:
  `InjectionFirewallHook.post_tool` truncates the re-injectable
  `ToolResult.output` and sets `error='injection_detected:...'` when
  an `INJECTION_PATTERNS` marker fires; clean results pass through
  identity ([SECURITY.md §"Prompt-input firewall" and §"Prompt injection"](../SECURITY.md);
  issues #5 and #122).
- `benchmarks/tasks/test_hook_isolation_evals.py` — control: a thrown
  exception in one registered hook is isolated by `HookRegistry` so
  the original `ToolCall`/`ToolResult` passes through unchanged and
  every other registered hook still runs ([ADR-0004](0004-self-modification-guardrails.md)
  hook-failure isolation, issue #21, `harness/hooks/base.py`).
- `benchmarks/tasks/test_evolver_guardrail_evals.py` — control:
  `ProposedEdit(target_file=...)` confines edits to
  `harness/system_prompt.txt`, `harness/hooks/...`, and
  `harness/skills/...`; `Evolver._check_rate_limit` raises
  `EvolverGuardError` past `max_proposals_per_hour`
  ([ADR-0004](0004-self-modification-guardrails.md),
  [SECURITY.md §"Rate limits"](../SECURITY.md)).

The four tasks ship as an additive family: the existing coding-task
suite is untouched, and the `test_stop_after_two_failures.py` task
from issue #111 keeps its `harness-rule` tag. The new tasks share the
`'security'` tag so the Critic can select or exclude the family
selectively (`uv run pytest -m benchmark -k security`). Each task
runs against the existing harness + foundry code directly; no
subprocess, no Runner, and no Evolver invocation is required, so the
family stays within the "<30 s on a developer laptop" budget that
ADR-0005 inherits from the existing suite.

`docs/SECURITY.md` §"Scope of this document" is updated to remove the
"track that work as an ADR" placeholder and to cross-reference this
ADR plus the four task filenames, so the document stays honest about
what is enforced.

## Consequences

- The Critic gate (ADR-0004) now has regression targets for four
  distinct security controls; a regression in any of them surfaces at
  PR review, not at deployment.
- The benchmark suite grows from six coding-shaped tasks (issue #30)
  plus one rule-shaped task (issue #111) to ten tasks. Selection by
  marker (ADR-0005) still scopes a sub-suite trivially
  (`uv run pytest -m benchmark -k security`).
- The four tasks import `harness.*` and `foundry_x.*` directly; the
  existing `pythonpath = ["."]` setting in `pyproject.toml` already
  exposes both, so no `pyproject.toml` change is required (and
  ADR-0008 forbids bundling one in this PR).
- Adding new security controls later is a single-PR change: drop a
  new `BenchmarkTask` + `@pytest.mark.benchmark` test under
  `benchmarks/tasks/`, list the file in this ADR's "Decision" section,
  and the Critic gates it automatically.
- A regression that weakens the `INJECTION_PATTERNS` set, removes a
  `_redact` regex, drops the `_isolate_failure` try/except, or lowers
  `max_proposals_per_hour` below operational need will turn one of
  the new benchmarks red. That is the regression signal this ADR
  exists to create.
- See [ADR-0004](0004-self-modification-guardrails.md) for the Critic
  contract, [ADR-0005](0005-pytest-as-evaluation-framework.md) for the
  pytest contract, and [ADR-0006](0006-pydantic-for-module-boundaries.md)
  for the `BenchmarkTask` schema.
