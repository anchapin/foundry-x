# ADR-0011: Digester `FailureReport` class taxonomy

## Status

Accepted. 2026-07-11. Updated 2026-07-17: `context-overflow` removed
from "pending" — shipped in issue #576 (Phase 3).

## Context

The `Digester` (`src/foundry_x/evolution/digester.py`) is the first link
of the evolution loop (Runner → trace → Digester → Evolver → Critic).
It walks an ordered list of `TraceEvent`s and emits a
`FailureReport` whose `proposed_class` is the unit the `Evolver`
consumes. [ADR-0007](0007-trace-driven-development.md) makes the trace
ground truth; [ADR-0006](0006-pydantic-for-module-boundaries.md) fixes
the `FailureReport` schema; [ADR-0010](0010-runner-agent-loop.md)
extended the trace vocabulary with `user_prompt`, `tool_call`,
`tool_result`, and a terminal `outcome` event carrying
`status`/`reason`/`steps`.

The taxonomy of `proposed_class` values, however, lives only in code:

- The four code-defined classes and their keyword buckets are encoded
  in `src/foundry_x/evolution/digester.py:93-152`
  (`_CLASS_KEYWORDS`); the catch-all `tool-error` is intentionally
  last.
- The structured `injection_blocked` event (issue #120) is *not* in
  `FAILURE_KINDS`; it is handled by a dedicated aggregation pass
  (`_aggregate_injection_blocks`) that emits
  `proposed_class='injection-attempt'` with one entry per block.
- The two non-failure sentinel values (`"clean"` when no failure is
  detected; the default `"unknown"` on a fresh `FailureReport`) are
  defined inline on the model and the digest return path.

Every lock-in test in `tests/test_digester.py` (around lines 321-329
and the parametrised keyword cases that follow) restates one rule in
its docstring: **"new keywords land in an ADR, not a test."** Today,
that rule is enforced only by reviewer habit. There is no ADR to point
at, so adding a new class (or a new keyword bucket) would mean
breaking the docstring contract with nothing to justify the change.

`docs/ROADMAP.md` Phase 3 ("Optimize context management") introduces a
new first-class failure mode: the Runner agent loop can
`outcome.status="truncated"` / `outcome.reason="max_steps"`
([ADR-0010](0010-runner-agent-loop.md)) when the context budget is
exhausted. The Phase-3 context-pruning work needs a fifth class to
bucket that event so the `Evolver` can write a `ProposedEdit` against
the pruning hook rather than against some generic `tool-error`
catch-all. That fifth class is the missing piece this ADR codifies so
the follow-up code change has a stable target.

## Decision

`FailureReport.proposed_class` is closed under the following set of
strings. Every classifier, generator, or consumer that ever assigns
to `proposed_class` MUST emit one of these values. New values land in
this ADR (and the index in `docs/adr/README.md`) before they land in
code or in a test.

### The five failure classes

1. **`wrong-tool`** — Agent invoked a tool or command that is not
   registered (or whose name does not match any `harness/skills/*.json`
   schema). Keyword bucket, in priority order
   (`src/foundry_x/evolution/digester.py:93-106`):

   ```text
   no such tool | tool not found | unknown tool | invalid tool
   | no tool named | no command named | unknown function
   | is not a valid tool
   ```

   Cause template
   (`digester.py:157-160`): the agent should review the
   available-tool list in the prompt.

2. **`bad-prompt`** — The task prompt is ambiguous, under-specified,
   or unparseable. Keyword bucket
   (`digester.py:107-120`):

   ```text
   missing context | under-specified | underspecified
   | malformed prompt | cannot parse prompt | contradictory
   | ambiguous | vague | unparseable
   ```

   Cause template
   (`digester.py:161-165`): tighten the prompt with concrete
   acceptance criteria.

3. **`state-leak`** — Execution hit stale, leaked, or contaminated
   state from a prior session. Keyword bucket
   (`digester.py:121-135`):

   ```text
   no such file | file not found | already exists
   | unexpected state | race condition | dirty tree
   | is not empty | leftover | stale | leak
   ```

   Cause template
   (`digester.py:166-170`): check sandbox reset/cleanup
   between steps.

4. **`tool-error`** — Generic catch-all for any failing event whose
   payload carries a structural signal (`kind` in `FAILURE_KINDS` or a
   `payload_key` in `FAILURE_PAYLOAD_KEYS`) but no specific-mode
   keyword matched. Keyword bucket
   (`digester.py:136-151`):

   ```text
   traceback | exception | timed out | timeout | exit code
   | segmentation fault | segfault | broken pipe
   | command not found | error | failed
   ```

   Cause template
   (`digester.py:169-172`): inspect the failing call's
   traceback. **This class is intentionally last in `_CLASS_KEYWORDS`;
   the three specific classes above always win over it.**

5. **`injection-attempt`** — The `InjectionFirewallHook` suppressed
   one or more tool results for prompt-injection markers (issue
   #120). The Digester recognises the emitting event kind
   (`injection_blocked`, exposed as the module constant
   `INJECTION_BLOCKED_KIND`) via a dedicated aggregation pass
   (`_aggregate_injection_blocks`); the generic first-failure walk
   never produces this class. Every block in the session is listed
   under `failed_steps`, with the first block's marker list as the
   `{match}` evidence. Cause template
   (`digester.py:177-182`): treat the agent as compromised for
   this session; consider tightening the firewall patterns or the
   upstream tool-result scrubbing policy.

### The fifth keyword class: `context-overflow`

The Phase-3 context-pruning work introduced a class for sessions where
the Runner agent loop terminates via
`outcome.status="truncated"` / `outcome.reason="max_steps"`
([ADR-0010](0010-runner-agent-loop.md) §Termination semantics).

The classification change shipped in issue #576
(`src/foundry_x/evolution/digester.py:347-431`), with keyword-bucket
and parametrised tests in `tests/test_digester.py` (lines 666-793),
following the same "ADR + keyword tuple + aggregation pass + test"
pattern as the four existing classes.

- **Class name:** `context-overflow`.
- **Trigger event:** any `TraceEvent` with
  `kind='outcome'`, `payload['status'] == 'truncated'`, and
  `payload['reason'] == 'max_steps'` ([ADR-0010](0010-runner-agent-loop.md)
  §Termination semantics). The wall-clock variant
  (`run_with_limits` aborting in
  `src/foundry_x/execution/runner.py:253`) is a separate guardrail
  and is **out of scope** for this class — it stays a `tool-error` or
  a future distinct class declared here in a later amendment.
- **Implementation:** `_aggregate_max_steps` in `Digester.digest()`
  (`digester.py:347-431`), which short-circuits the generic walk so a
  max-steps truncation is reported as `context-overflow` even when a
  later `tool_error` also fires. Tests at
  `tests/test_digester.py:687-793`.
- **Cause template:** "Agent loop reached `max_steps` (steps=N)
  before producing a final answer; the context budget was exhausted.
  Review the pruning hook and the model's tendency to repeat tool
  calls."

### Two non-failure sentinels

- **`clean`** — Returned by `Digester.digest` when no failure event
  is present in the session
  (`digester.py:381-386`). `failed_steps` is `[]` and
  `suspected_causes` is `[]`. Not a class of failure; the absence of
  one.
- **`unknown`** — The default value of `FailureReport.proposed_class`
  before classification runs
  (`digester.py:42`). Producers MUST overwrite it before returning.
  Consumers MUST treat it the same as `clean` for routing purposes.

### Class invariants

These are the rules reviewers enforce when an ADR amendment adds (or
removes) a class:

- `proposed_class` is one of: `wrong-tool`, `bad-prompt`,
  `state-leak`, `tool-error`, `injection-attempt`, `context-overflow`,
  `clean`, `unknown`. No other strings reach `proposed_class`.
- The four code-defined classes are encoded in `_CLASS_KEYWORDS` in
  this fixed priority order, *most-specific first*:
  `wrong-tool` → `bad-prompt` → `state-leak` → `tool-error`. The
  catch-all is intentionally last so a payload that contains a
  generic token (`"error"`, `"failed"`, `"traceback"`) alongside a
  more specific phrase still resolves to the specific class
  (`tests/test_digester.py::test_specific_keyword_beats_tool_error_catchall`).
- Each keyword tuple, in turn, is ordered most-specific-first so the
  recorded `{match}` evidence string is deterministic (not
  hash-ordered).
- `INJECTION_BLOCKED_KIND` is intentionally **not** in
  `FAILURE_KINDS`: its aggregation lives outside the generic walk,
  and adding it there would under-report adversarial surface.
- `clean` is returned only by `digest`; it never appears in any
  keyword or aggregation rule. `unknown` is a model default only;
  `digest` always overwrites it.
- A new class ships in this ADR (with its keyword bucket, trigger,
  and cause template spelled out) **in the same PR** as the
  `_CLASS_KEYWORDS` change (or aggregation rule) and the
  parametrised test in `tests/test_digester.py`. The existing
  docstring rule *"new keywords land in an ADR, not a test"* is the
  practical enforcement; this ADR is now the artefact it points to.

## Consequences

- **Vocabulary, not code.** The five failure classes and the two
  sentinels are now vocabulary, codified here. A reviewer who reads
  this ADR and the `_CLASS_KEYWORDS` table can confirm that the
  code reflects the contract without diff archaeology.
- **Closed class set.** Adding a class is a vocabulary change and
  ships in this ADR per ADR-0001 ("new ADRs land in the same PR as
  the change they justify when practical"). Removing a class is a
  breaking change for any downstream `Evolver` routing logic and
  requires an ADR that supersedes this one.
- **Critic gating (ADR-0004) inherits stability.** The lock-in tests
  in `tests/test_digester.py` (parametrised keyword, specificity/
  precedence, structural-signal, injection aggregation) already pin
  every class declared here. A refactor that silently swaps a
  keyword bucket surfaces as a test failure; a refactor that adds a
  class *without* an ADR amendment is rejected at PR review.
- **`context-overflow` shipped.** Issue #576 implemented the
  `_aggregate_max_steps` aggregation pass in `Digester.digest()`
  (`digester.py:347-431`), making `context-overflow` a live failure
  class. The implementation follows the same "ADR + aggregation pass +
  parametrised test" pattern as `injection-attempt`, with tests at
  `tests/test_digester.py:687-793`. The "out of scope" list in the
  original proposal (issue #190's `FAILURE_KINDS` change and the
  pruning-hook wiring) remain separate follow-up items.
- **`injection-attempt` stays special.** Its aggregation rule
  deviates from the keyword-bucket pattern by design — adversarial
  blocks should always land as one report with the full surface in
  `failed_steps`, not be picked off one-by-one by the generic walk.
  A future reviewer who proposes "fold `injection-attempt` into
  `_CLASS_KEYWORDS`" must rebut that argument here.
- **Trace vocabulary alignment is unchanged.** Per ADR-0006 the
  vocabulary of `kind` values is a `pydantic` enum at module
  boundaries; this ADR does not modify that enum. The
  `outcome`/`truncated`/`max_steps` taxonomy is owned by
  [ADR-0010](0010-runner-agent-loop.md); the `_aggregate_max_steps`
  aggregation pass in issue #576 turned it into `context-overflow`.
- **Cross-references.** See
  [ADR-0006](0006-pydantic-for-module-boundaries.md) (FailureReport
  is a pydantic model),
  [ADR-0007](0007-trace-driven-development.md) (trace content
  drives the report — every keyword and cause template points back
  to a trace string),
  [ADR-0010](0010-runner-agent-loop.md) (source of the
  `outcome.status=truncated` event the fifth class will bucket),
  [`docs/CONTEXT.md`](../CONTEXT.md) (Digester and failure-report
  glossary entries), and issue #120 (origin of
  `injection_blocked` and the `injection-attempt` class).
