# ADR-0007: Trace-driven development is the default

## Status

Accepted. 2026-07-10.

## Context

FoundryX's central claim is that observed agent behavior, recorded
as traces, is the most reliable input for improving the agent. If
that is true for the product, it should also be true for how we
develop the product. Speculation-based PRs ("this feels cleaner",
"the agent probably needs...") are the cheapest to write and the
most expensive to merge.

## Decision

The trace store under `logs/` is the ground truth of what our agents
and our contributors do. Default workflow:

1. Before proposing a change to behavior, read the relevant traces.
2. State the observed behavior in the PR description, with a trace
   excerpt or a reproduction command.
3. State the expected behavior after the change, with the test or
   benchmark that proves it.
4. PR review checks the diff *and* the evidence link.

Speculation is allowed for refactors that do not change behavior
("improve readability of X"), but never for behavior changes.

## Consequences

- Traces must be inspectable. The CLI under
  `src/foundry_x/trace/cli.py` is part of the developer surface,
  not an internal utility.
- PRs without evidence require reviewer pushback or rejection.
- The CONTRIBUTING.md "evidence-led" rule and the AGENTS.md
  "Observe first" step both follow from this ADR.
- When traces are not available (e.g., a brand-new subsystem), a
  failing test that the change makes pass is acceptable evidence.
- PHILOSOPHY.md §1 ("Evidence over opinion") restates this rule in
  normative form.
