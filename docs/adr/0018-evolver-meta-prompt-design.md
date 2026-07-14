# ADR-0018: Evolver Meta-Prompt Design

## Status

Accepted. 2026-07-14.

## Context

The `Evolver` (the meta-agent defined in `CONTEXT.md` §Concepts) is the
second link of the evolution loop after the `Digester`. It takes a
`FailureReport` and produces `ProposedEdit` objects that the `Critic`
then evaluates. Issue #476 identified that the Evolver's original
implementation used only fixed templates (`_PROPOSED_CLASS_EDIT_TEMPLATES`
in `src/foundry_x/evolution/evolver.py:41-77`) and could not produce
novel prompt instructions.

The goal of this ADR is to design the meta-prompt — the system prompt
that guides the Evolver-as-meta-agent when it generates harness edits —
so that future implementations can be LLM-driven rather than
template-driven.

## Decision

The Evolver meta-prompt is a standalone file,
`harness/evolver_meta_prompt.txt`. It is external to the foundry
(`src/foundry_x/`) so it can be evolved by the Evolver itself (闭环)
without requiring a separate self-modification guardrail.

### Documented Contract

The meta-prompt encodes four pieces of ground truth:

1. **Harness schema** — the closed set of editable files:
   `system_prompt.txt`, `manifest.json`, `hooks/*.py`, `skills/*.json`.
   This is the ADR-0004 path confinement rendered as an explicit
   enumeration rather than embedded in validator code.

2. **Failure taxonomy** — the five `proposed_class` values from
   ADR-0011 (`wrong-tool`, `bad-prompt`, `state-leak`, `tool-error`,
   `injection-attempt`) plus the two sentinels (`clean`, `unknown`),
   with class-specific guidance on what kind of edit each class calls
   for. This bridges the Digester vocabulary and the Evolver's edit
   strategy.

3. **ADR-0004 path confinement rules** — rate limit (10 proposals /
   hour), diff size cap (200 lines), Critic gate (full pytest + benchmark
   suite), and human-review requirement for hand-edits. These are
   already in prose in `SECURITY.md`; the meta-prompt renders them as
   operational constraints the Evolver must obey.

4. **Unified diff format** — the `--- a/<path>` / `+++ b/<path>` header
   requirement, the `@@` hunk header requirement, and the 200-line cap.
   This is enforced at the `ProposedEdit` validator
   (`_unified_diff_has_git_apply_headers`) but is also documented in the
   meta-prompt so the Evolver cannot generate edits that fail at the
   boundary.

### Good vs Bad Edit Examples

The meta-prompt includes three pairs of examples:

- A **good** `wrong-tool` edit (reinforce tool adherence).
- A **good** `injection-attempt` edit (tighten firewall patterns).
- Three **bad** edits: one that escapes the harness tree, one that omits
  git headers, and one that exceeds the diff size cap.

The bad examples are explicit negative evidence: they pin the
path-confinement and diff-format invariants in a way that prose alone
cannot.

### Class-Specific Guidance Table

The meta-prompt uses a table mapping each `proposed_class` to the
typical root cause and a one-sentence edit strategy. This gives the
Evolver a deterministic starting point while leaving room for novel
proposals that are nonetheless grounded in the taxonomy.

### ADR-0004 Relationship

ADR-0004 establishes the Critic gate. This ADR does not modify ADR-0004;
it *renders* it as operational guidance inside the meta-prompt so the
Evolver can reason about its own constraints.

## Consequences

- The Evolver meta-prompt is itself part of the harness and subject to
  the same Critic gate as any other harness edit.
- The meta-prompt is intentionally plain text (`.txt`) rather than JSON or
  Python so it can be read by any model without an additional parsing step.
- The meta-prompt does not include the `ProposedEdit` pydantic schema;
  that schema lives in `src/foundry_x/evolution/evolver.py` and is
  enforced by the validator. The meta-prompt describes intent; the schema
  enforces structure.
- The fixed templates in `_PROPOSED_CLASS_EDIT_TEMPLATES` remain as a
  fallback for non-LLM Evolver implementations. They are not removed.
- Adding a new `proposed_class` value (per ADR-0011 §Class invariants)
  requires amending this ADR and updating the meta-prompt in the same PR.

## Alternatives Considered

### Schema in the meta-prompt

Representing the `ProposedEdit` JSON schema in the meta-prompt was
rejected because the schema lives at the pydantic boundary (`ADR-0006`)
and must be kept in sync with the code. Encoding it in the meta-prompt
would create a second source of truth prone to drift.

### Separate meta-prompt ADR

Creating a separate ADR for each sub-section (harness schema, failure
taxonomy, etc.) was rejected in favor of a single document because the
acceptance criteria for issue #476 are a unit: the meta-prompt cannot
work without all four pieces, and an ADR for each would create
cross-reference churn.

### In-code prompt template

Storing the meta-prompt as a Python string constant in `evolver.py` was
rejected because the meta-prompt must be evolvable by the Evolver without
requiring a code change and a foundry release. A standalone file in the
harness tree can be edited and reviewed like any other harness artifact.

## Cross-References

- [ADR-0004](./0004-self-modification-guardrails.md): Critic gate and
  self-modification guardrails.
- [ADR-0006](./0006-pydantic-for-module-boundaries.md): `ProposedEdit`
  as a pydantic model at the module boundary.
- [ADR-0011](./0011-failure-report-class-taxonomy.md): Failure class
  vocabulary (`wrong-tool`, `bad-prompt`, `state-leak`, `tool-error`,
  `injection-attempt`).
- [ADR-0012](./0012-manifest-json-as-evolver-target.md): Expansion of
  allowed Evolver targets to include `manifest.json`.
- [`src/foundry_x/evolution/evolver.py`](../../src/foundry_x/evolution/evolver.py):
  `ProposedEdit` model and `_confine_to_harness_tree` validator.
- [`harness/evolver_meta_prompt.txt`](../../harness/evolver_meta_prompt.txt):
  The meta-prompt itself.
