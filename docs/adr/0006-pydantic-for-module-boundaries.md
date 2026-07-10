# ADR-0006: Use pydantic for structured data at module boundaries

## Status

Accepted. 2026-07-10.

## Context

FoundryX is built from a small set of cooperating subsystems
(Tracer, Runner, Digester, Evolver, Critic). They exchange structured
payloads: trace events, failure reports, proposed edits, benchmark
results. Implicit schemas (duck typing, dicts, kwargs) work at first
but erode quickly when the data crosses subsystem boundaries, gets
serialized to JSON, or is persisted to the trace store.

## Decision

We use `pydantic` v2 models for every structured payload that crosses
a module boundary:

- Trace events (`src/foundry_x/trace/`)
- Failure reports (`src/foundry_x/evolution/digester.py`)
- `ProposedEdit` and friends (`src/foundry_x/evolution/evolver.py`)
- Critic verdicts (`src/foundry_x/evolution/critic.py`)
- Benchmark task definitions (`benchmarks/`)
- Anything that will be serialized, persisted, or read by an
  external process.

Internal function-local data may stay untyped; the rule applies at
the import boundary. `Any` is allowed only with an explanatory
comment.

## Consequences

- One dependency (`pydantic>=2.6`) for schema, validation, and JSON
  serialization. Already in `pyproject.toml`.
- Validation errors are caught early and carry actionable messages,
  which the evolution loop can inspect.
- Schemas are reviewable in PR diffs in the same way logic is.
- Pydantic is a hard runtime dependency; changing the schema layer
  requires an ADR.
- AGENTS.md §4 codifies this for AI collaborators ("No `Any` without
  a comment explaining why").