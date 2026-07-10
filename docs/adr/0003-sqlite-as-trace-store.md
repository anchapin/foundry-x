# ADR-0003: SQLite as the default trace store

## Status

Accepted. 2026-07-10.

## Context

The `TraceLogger` records every prompt, tool call, and outcome to
produce the evidence that drives the evolution loop. Options
considered:

- JSONL files
- SQLite
- DuckDB
- PostgreSQL

## Decision

The default trace store is SQLite, accessed via `sqlite-utils`. Each
run gets a fresh database file under `logs/`. JSONL is supported as
an export format for portability, not as the primary store.

## Consequences

- Zero-setup local development: no external service required.
- Single-file inspection with `sqlite-utils` CLI and the `sqlite3`
  shell.
- Adequate for our scale (single user, single host). When we cross
  ~10M rows or need concurrent writers, revisit (see ADR-0001).
- The trace schema is defined as `pydantic` models in
  `src/foundry_x/trace/` and serialized via `sqlite-utils` helpers.
  No raw SQL strings in business logic.
- Sensitive values are redacted at write time (see
  `docs/SECURITY.md`).
