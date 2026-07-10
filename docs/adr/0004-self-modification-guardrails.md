# ADR-0004: Self-modification guardrails via the Critic gate

## Status

Accepted. 2026-07-10.

## Context

FoundryX edits its own `harness/` files. Without gatekeeping, a bad
proposal could degrade the agent or silently introduce regressions.
The PRD calls for a `Critic`; the question is what the Critic must
enforce and how strictly.

## Decision

Every harness edit, whether produced by the `Evolver` or proposed by
a human, must pass through the `Critic` gate before it is marked
active. The gate runs:

1. The full pytest suite.
2. The benchmark suite (`benchmarks/`) at a configurable size.
3. A regression check: no previously-passing benchmark task may
   newly fail.

Hand-edits to `harness/` are not blocked but require a PR with
explicit justification and a second human reviewer.

## Consequences

- The Critic is a hard dependency of the Evolver. Removing it is a
  breaking change and requires its own ADR.
- The Critic itself is versioned and tested. Changes to the Critic
  require their own ADR.
- Evolution runs are slower but bounded; failure modes are
  observable in traces.
- See `docs/SECURITY.md` for the broader threat model this guard
  exists within.
- See `AGENTS.md` for the operational rule that forbids bypassing
  this gate.