# ADR-0022: Store implementation selection criteria

## Status

Accepted. 2026-07-17. Clarifies and expands the evaluation rationale
first recorded in [ADR-0003](./0003-sqlite-as-trace-store.md); does not
supersede ADR-0003 (which covers the SQLite decision specifically).

## Context

The `TraceLogger` persists every prompt, tool call, and outcome to an
on-disk store that the Digester, KPI engine, and Critic can query.
Choosing the right store implementation affects:

- **Developer experience**: zero-setup vs. managing external services.
- **Query latency**: analytical reads vs. raw event throughput.
- **Operational surface**: backup, replication, schema migration.
- **Evolution-loop interactivity**: the store is read by long-lived
  background processes (Digester, KPI) while the Runner is still writing.

ADR-0003 selected SQLite as the default. This ADR documents the
evaluation process — the criteria, alternatives, and the conditions
that would trigger a migration.

## Alternatives Considered

### 1. PostgreSQL (via `asyncpg` or `psycopg3`)

**Summary:** Full client/server relational database.

**Pros:**
- Mature concurrency model: true parallel readers and writers.
- Network access: multi-host setups, connection pooling, managed cloud
  offerings (RDS, Neon, Supabase).
- Rich query language: window functions, CTEs, full-text search.
- Operational tooling: `pg_dump`, logical replication, `pg_bouncer`.

**Cons:**
- **External service required.** Every developer must have PostgreSQL
  running or have network access to a shared instance. Local-first
  iteration is gated on infrastructure.
- **Connection management complexity.** `asyncpg` requires a connection
  pool; connection strings, auth, and SSL add config surface.
- **Overkill for the trace-access pattern.** The trace store is written
  by one owner (the Runner) and read by one or two consumers
  (Digester, KPI). Concurrent writer contention is not the primary
  bottleneck — see ADR-0013 for how SQLite's WAL mode addressed the
  reader/writer case.
- **Migration of existing traces.** Switching to PostgreSQL would require
  a schema-export path from the existing SQLite files under `logs/`.

**Verdict at time of decision (2026-07-10):** Rejected for local-first
developer experience. Revisit if concurrent multi-host writers become
a documented requirement (see Migration Triggers below).

---

### 2. SQLite via `aiosqlite` (async SQLite driver)

**Summary:** SQLite accessed asynchronously via `aiosqlite`, preserving
the single-file simplicity but enabling non-blocking I/O in async
contexts.

**Pros:**
- All SQLite advantages retained: single file, zero-setup, portable.
- Non-blocking I/O: better fit for `async`-first code paths.

**Cons:**
- **`aiosqlite` is a thin wrapper.** It still serializes disk I/O
  through the OS; it does not give SQLite true async write concurrency.
  The performance benefit is marginal for our write-once-per-event
  pattern.
- **Added dependency.** `aiosqlite` is an extra package; `sqlite3` is
  in the stdlib.
- **No concurrent writer support.** `aiosqlite` does not change
  SQLite's single-writer limitation.

**Verdict at time of decision (2026-07-10):** Rejected. The async
benefit is small for the trace-write pattern, and stdlib `sqlite3`
with WAL mode (ADR-0013) already solves the reader/writer concurrency
problem without a new dependency.

---

### 3. In-memory store (Python `dict` / `list`, or `imesync`)

**Summary:** Ephemeral store that lives in the Runner process memory and
is not persisted to disk.

**Pros:**
- Fastest possible writes: no disk I/O at all.
- Simplest possible implementation.

**Cons:**
- **No persistence.** A crashed Runner loses all trace data for that
  run. The evolution loop depends on having trace data available after
  the run completes (for the Critic, KPI, and Digester to read).
- **No cross-process sharing.** The Digester runs in a separate process;
  it cannot access the Runner's memory. A separate IPC or file-based
  bridge would be needed, negating the simplicity gain.
- **No queryable history.** In-memory data is gone after the process
  exits. The KPI and trend analysis require querying historical runs.

**Verdict at time of decision (2026-07-10):** Rejected. Persistence and
cross-process access are hard requirements for the evolution loop.

---

### 4. DuckDB

**Summary:** Column-oriented OLAP database designed for analytical
workloads; supports SQLite-compatible storage and Python native bindings.

**Pros:**
- Excellent for analytical queries: columnar storage, vectorized execution.
- SQLite compatibility mode: can read `.db` files produced by
  `sqlite-utils`.
- Pure Python: no external service, `pip install duckdb`.

**Cons:**
- **Larger binary/dependency.** DuckDB is a substantial library (~100 MB);
  `sqlite3` is in the stdlib.
- **Overkill for write-heavy trace logging.** DuckDB's strengths are
  analytical reads; the trace store is write-once, read-occasionally
  during the evolution loop. The columnar format adds write overhead
  for marginal read benefits at our scale.
- **Ecosystem familiarity.** `sqlite-utils` has first-class support for
  `sqlite3`; DuckDB tooling (migration, backup, inspect) is less
  familiar to the team.

**Verdict at time of decision (2026-07-10):** Rejected for now. DuckDB
is a strong candidate if analytical query latency on large trace sets
becomes a documented bottleneck.

---

### 5. JSONL files (one file per run)

**Summary:** One `run-YYYYMMDDTHHMMSS.jsonl` file written append-only
with `json.dumps` per event.

**Pros:**
- Zero-dependency: plain text, readable in any editor.
- Trivially portable: `cp` works, `grep` works, `jq` works.
- No schema enforcement.

**Cons:**
- **No incremental query.** Reading a single run requires scanning the
  entire file. The Digester's `query_events` filter (by session, time
  range, outcome) would need to scan all lines every time.
- **No structured schema.** Drift in event field names goes undetected;
  `pydantic` validation at write time (which `sqlite-utils` wrappers
  provide for SQLite) is absent.
- **No concurrency primitives.** Appending to the same JSONL from
  multiple processes risks partial writes. No equivalent to WAL or
  file locking.
- **Large file sizes.** A medium-length run produces a multi-MB JSONL
  that must be fully loaded to filter.

**Verdict at time of decision (2026-07-10):** Rejected. JSONL is used
as an export/portability format (supported via `log --format jsonl`) but
is not the primary store. The query and schema-enforcement requirements
of the evolution loop make SQLite the better default.

## Decision

SQLite (via stdlib `sqlite3` + `sqlite-utils` helpers) is retained as
the primary trace store. The evaluation criteria applied were:

| Criterion | Weight | SQLite | PostgreSQL | aiosqlite | In-memory | DuckDB | JSONL |
| --------- | ------ | ------ | --------- | --------- | --------- | ------ | ----- |
| Zero-setup local dev | High | ✅ | ❌ | ✅ | ✅ | ✅ | ✅ |
| Cross-process reader access | High | ✅ WAL | ✅ | ✅ | ❌ | ✅ | ⚠️ IPC bridge |
| Query selectivity (filter by session/time) | High | ✅ | ✅ | ✅ | ❌ | ✅ | ❌ |
| Schema enforcement at write time | Medium | ✅ | ✅ | ✅ | ❌ | ⚠️ | ❌ |
| No external dependency | High | ✅ | ❌ | ✅ | ✅ | ⚠️ large dep | ✅ |
| Concurrent writer support | Low | ❌ | ✅ | ❌ | ❌ | ❌ | ❌ |
| Analytical read performance | Low | ⚠️ adequate | ✅ | ⚠️ adequate | ❌ | ✅ | ❌ |

The "Low" weights on concurrent writer support and analytical read
performance reflect the 2026-07-10 scale: single-host, single-user,
single-writer. These may change — see Migration Triggers below.

## Consequences

- All consequences listed in [ADR-0003](./0003-sqlite-as-trace-store.md)
  and [ADR-0013](./0013-wal-mode-and-connection-reuse.md) apply.
- The `logs/*.db` files remain the canonical trace store. Use
  `log --format jsonl` for portability only.
- New contributors evaluating a store change should use this ADR as the
  reference: check the criteria table, the Migration Triggers below,
  and open a proposed ADR before implementing.

## Migration Triggers

The following conditions would justify a migration away from SQLite:

1. **Concurrent multi-host writers.** If the Runner itself runs on
   multiple hosts writing to a shared store, SQLite's single-writer
   model breaks. Escalate to PostgreSQL (or a managed vector store)
   per [ADR-0001](./0001-record-architecture-decisions.md).
2. **Trace set exceeds ~10M rows and analytical query latency exceeds
   5 seconds.** At that scale, DuckDB's columnar reads become worth the
   operational cost. Benchmark first.
3. **Managed service requirement.** If the team adopts a hosted
   data stack (e.g., Supabase, Neon), PostgreSQL becomes the natural
   backing store with no external-service overhead.

These triggers do not expire; they are conditions this ADR explicitly
calls out for future evaluation.
