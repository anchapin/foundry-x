# ADR-0035: Parallel Sub-Agent Issue Generation

**Status:** Proposed
**Date:** 2026-07-27
**Deciders:** FoundryX team

## Context

Phase 1–3 are shipped. To continue advancing the PRD KPIs (cycle time,
regression rate, improvement rate) and the token budget metric, the project
needs a repeatable process for generating high-quality, non-overlapping GitHub
issues that are distributed across subsystems to enable parallel implementation.

Manual single-agent issue writing is slow and tends to produce
broad, overlapping issues. A parallel sub-agent approach — each agent
investigating one subsystem with a shared context and structured output format —
maximizes coverage and enables the work to be split across 9 subsystems
simultaneously.

## Decision

Adopt a shared prompt template (`.agents/prompts/parallel-issue-generation.md`)
that is instantiated per subsystem and dispatched to parallel sub-agents.
Each sub-agent:
1. Reads the subsystem's source, ADRs, and CONTEXT.md entries
2. Proposes 3–5 issues scoped to that subsystem
3. Returns results as a structured JSON array

A companion dispatch script (`.agents/scripts/dispatch-issue-generation.sh`)
handles the parallel spawning and result collection.

## Dispatched Subsystems

1. `trace_store_and_observability`
2. `execution_runner`
3. `evolution_pipeline`
4. `critic_and_benchmarking`
5. `harness_dna`
6. `context_pruning_and_token_management`
7. `model_adapter_and_server`
8. `cli_and_operator_experience`
9. `testing_and_ci`

## Issue Schema

Each issue must contain:
- `title`: `subsystem: short description` (≤72 chars)
- `body`: What / Why it matters (KPI impact) / How to verify / Acceptance criteria
- `labels`: at minimum one of `kpi-cycle-time`, `kpi-regression-rate`,
  `kpi-improvement-rate`, or `tracked-metric`; plus a `kind-*` label

## Consequences

- Issues will be well-scoped to one subsystem, avoiding cross-cutting scope creep
- Parallel dispatch reduces total wall-clock time for issue generation
- The dispatch script must be updated if new subsystems are added
- Harness-layer changes (`harness/hooks/`, `harness/skills/`, `system_prompt.txt`)
  are explicitly excluded — those are evolved by the loop only

## References

- `.agents/prompts/parallel-issue-generation.md` — the shared prompt template
- `.agents/scripts/dispatch-issue-generation.sh` — the dispatch script
- docs/PRD.md — KPI definitions
- docs/CONTEXT.md — subsystem vocabulary
