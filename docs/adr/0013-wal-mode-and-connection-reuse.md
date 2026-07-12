# ADR-0013: WAL mode and a single reused SQLite connection

## Status

Accepted. 2026-07-12. Supersedes the single-writer note of
[ADR-0003](./0003-sqlite-as-trace-store.md) (its "revisit when concurrent
writers" trigger has been crossed).

## Context

`TraceLogger` (issue #3) opened a brand-new `sqlite3.connect()` on every
operation — 11 call sites in `src/foundry_x/trace/logger.py`, 0 WAL pragmas.
Two consequences motivated this decision:

1. **Per-event connection overhead.** Each `record()` paid a fresh
   `sqlite3.connect` (page-cache priming, lock acquisition, schema read) on
   every event. Issue #199 streams per-chunk `model_response` events, which
   multiplies that overhead across a run.
2. **Reader/writer contention without WAL.** In the default rollback-journal
   mode a Digester or KPI reader on a separate connection reading while a
   Runner write is in flight can raise `SQLITE_BUSY`. ADR-0003 stated the
   store was adequate "for our scale" and to "revisit when concurrent
   writers." Phase-3 scale (multiple sessions/day, a live Digester→Critic
   feedback path) crosses that revisit trigger: there is now a long-lived
   reader process consuming the trace while the runner is still writing it.

## Decision

Two coupled changes to the SQLite backend:

1. **Reuse one connection.** `TraceLogger.__init__` opens a single
   `sqlite3.Connection` and every method (writes and reads) reuses it for
   the logger's lifetime. Per-operation transactions are preserved by
   wrapping writes in `with self._conn:`, which commits on success and
   rolls back on exception — identical semantics to the previous
   `with sqlite3.connect(...) as conn:` blocks, minus the reconnect. A
   `close()` method releases the connection deterministically; existing
   callers that never call it keep working via garbage collection.

2. **Enable WAL.** The same `__init__` runs `PRAGMA journal_mode=WAL` on
   the reused connection. WAL is a persistent database property (stored in
   the file header), so raw `sqlite3.connect` readers opened against the
   same file later inherit WAL automatically. With WAL, writers do not
   block readers and readers do not block writers: a reader on a separate
   connection sees the last committed snapshot even while the runner holds
   an open write transaction — the property that removes the `SQLITE_BUSY`
   window the Digester was exposed to.

The on-disk schema is unchanged. The `-wal` and `-shm` sidecar files WAL
introduces are added to `.gitignore` so they never leak into version
control.

## Consequences

- **No `SQLITE_BUSY` for concurrent readers.** A Digester/KPI reader
  process can `iter_events` / `query_events` against the trace file while
  the Runner is mid-session. This is the property issue #274's acceptance
  criteria lock in with a regression test.
- **Lower per-event overhead.** One connection amortizes page-cache and
  lock setup across the whole run instead of paying it 11 times per
  operation. This is the precondition issue #199's streaming needs to land
  cheaply.
- **Single writer still.** WAL allows exactly one writer at a time; it
  does **not** turn SQLite into a multi-writer database. The trace store
  remains single-writer (the `TraceLogger` instance owned by the Runner).
  Multiple-writer workloads would still require the ADR-0001 escalation
  to a client/server database.
- **New sidecar files.** Each `logs/*.db` now carries `logs/*.db-wal` and
  `logs/*.db-shm` siblings while a connection is open. These are
  gitignored and are checkpointed back into the main file on clean close.
- **ADRs and code drift together.** ADR-0003's "revisit when concurrent
  writers" note is satisfied by this ADR; ADR-0003 is kept and
  cross-referenced rather than rewritten (ADR-0008 §2: superseded ADRs
  are kept).
