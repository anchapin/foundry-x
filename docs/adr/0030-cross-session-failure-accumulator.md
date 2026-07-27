# ADR-0030: Cross-session failure-pattern accumulator

## Status

Accepted.

## Context

Issue #1038: the `Evolver` currently proposes edits based on single-session
failure analysis — the `Digester` output for one `session_id`. There is no
mechanism to accumulate failure patterns across multiple sessions. If the same
failure class (`proposed_class`) appears across 5 sessions with similar
context, the `Evolver` treats each independently rather than recognizing the
recurring pattern.

The consequence is a reactive, session-local editing loop that cannot
distinguish between:

- A one-off idiosyncratic failure (spurious, no structural edit needed)
- A systemic pattern that has appeared in N > 1 sessions and warrants a
  targeted, high-confidence edit

This ADR introduces a **FailurePatternStore** — a durable, queryable
accumulator of failure events across sessions — and specifies how the
`Digester` and `Evolver` query it to inform edit generation.

## Decision

### 1. FailurePatternStore schema

The store is a SQLite table in the trace store (`logs/traces.db` or the
configured `FOUNDRY_TRACE_BACKEND` path). It lives in the same database as
the trace events so it can share the connection and transaction model.

```sql
CREATE TABLE IF NOT EXISTS failure_patterns (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    proposed_class   TEXT    NOT NULL,          -- e.g. 'wrong-tool', 'context-overflow'
    session_id       TEXT    NOT NULL,          -- source session
    timestamp        TEXT    NOT NULL,          -- ISO-8601, from trace event
    context_hash     TEXT    NOT NULL,          -- sha256 of the context bucket
    resolution       TEXT    NOT NULL DEFAULT '',-- 'pending' | 'proposed' | 'approved' | 'rejected'
    edit_id          TEXT    NOT NULL DEFAULT '',-- FK to proposed_edits.id once resolved
    created_at       TEXT    NOT NULL           -- when the row was inserted
);

CREATE INDEX IF NOT EXISTS idx_fp_class_hash
    ON failure_patterns(proposed_class, context_hash);
CREATE INDEX IF NOT EXISTS idx_fp_session
    ON failure_patterns(session_id);
```

**Schema invariants:**

- `proposed_class` is one of the closed class set defined in
  [ADR-0011](./0011-failure-report-class-taxonomy.md).
- `context_hash` is a `sha256` hex digest of a **context hash bucket** (see §2).
- `resolution` is the state of the edit that addressed this failure event.
  `"pending"` means no edit has yet been proposed for this `(class, bucket)`.
  Once a `ProposedEdit` is created, it transitions to `"proposed"`. The
  `ReviewStateMachine` drives the remaining transitions.
- `edit_id` is populated once an edit is linked to this pattern entry, enabling
  the store to answer "which pattern did this edit address?"

### 2. context_hash_bucket algorithm

Raw context is too granular to cluster similar failures across sessions. Two
failures with identical `proposed_class` but different task prompts should
still be considered the "same" pattern if they share the same structural
failure surface (e.g., same tool name, same error type, same step index).

A **context hash bucket** is derived from a fixed set of structural fields
extracted from the `FailureReport`. The algorithm is deterministic and
pure (no I/O).

```
context_hash_bucket(failure: FailureReport) -> str  # sha256 hex, 64 chars

Inputs from FailureReport:
  - failure.proposed_class           (e.g. 'wrong-tool')
  - failure.failed_steps[0].get('kind')       (e.g. 'tool_error')
  - failure.failed_steps[0].get('payload', {}).get('name')
                                          (e.g. tool name or null)
  - failure.failed_steps[0].get('payload', {}).get('error_type')
                                          (e.g. 'FileNotFoundError' or null)
  - failure.failed_steps[0].get('signal')
                                          (e.g. 'kind:tool_error')

Canonical form:
  kind = failure.failed_steps[0]['kind']
  tool_name = failure.failed_steps[0].get('payload', {}).get('name', '')
  error_type = failure.failed_steps[0].get('payload', {}).get('error_type', '')
  signal = failure.failed_steps[0].get('signal', '')

  bucket_input = '|'.join([
      failure.proposed_class,
      kind or '',
      tool_name or '',
      error_type or '',
      signal or '',
  ])
  # Normalize: strip whitespace, lower-case, collapse multiple '|' delimiters
  normalized = re.sub(r'\s+', ' ', bucket_input).strip().lower()
  normalized = re.sub(r'\|+', '|', normalized).strip('|')

Return sha256(normalized.encode('utf-8')).hexdigest()
```

**Why this works:** The bucket captures the structural "shape" of the failure
—not the task-specific text. Two sessions that both fail with
`proposed_class='wrong-tool'`, `kind='tool_error'`, and `name='bash'`
produce the same bucket regardless of what the task prompt said. A session
that fails with `proposed_class='wrong-tool'` but `name='git'` produces a
different bucket, correctly distinguishing the two tool-selection failures.

**Stability:** The algorithm intentionally excludes `session_id`,
`timestamp`, `event_id`, and any free-text field (`summary`,
`suspected_causes`) to avoid hashing session-specific noise into the bucket.

### 3. Pattern detection thresholds

A **pattern** is a `(proposed_class, context_hash_bucket)` pair that has
accumulated failure events across multiple sessions.

```
PATTERN_MIN_SESSIONS = 3        # minimum distinct sessions to be "recurring"
PATTERN_AGE_CUTOFF = 30 days    # ignore sessions older than this for pattern detection
```

**Query: detect a pattern before generating a template edit**

```python
def find_cross_session_pattern(
    store: FailurePatternStore,
    failure: FailureReport,
    min_sessions: int = PATTERN_MIN_SESSIONS,
    age_cutoff: datetime | None = None,
) -> CrossSessionPattern | None:
    bucket = context_hash_bucket(failure)
    age_cutoff = age_cutoff or (datetime.now(UTC) - timedelta(days=PATTERN_AGE_CUTOFF))

    rows = store.query(
        """
        SELECT proposed_class, context_hash,
               COUNT(DISTINCT session_id)          AS session_count,
               MAX(timestamp)                      AS latest_timestamp,
               GROUP_CONCAT(session_id)             AS session_ids
        FROM   failure_patterns
        WHERE  proposed_class = ?
          AND  context_hash    = ?
          AND  timestamp      >= ?
        GROUP BY proposed_class, context_hash
        HAVING COUNT(DISTINCT session_id) >= ?
    """,
        [failure.proposed_class, bucket, age_cutoff.isoformat(), min_sessions],
    )

    if not rows:
        return None
    row = rows[0]
    return CrossSessionPattern(
        proposed_class=row["proposed_class"],
        context_hash=row["context_hash"],
        session_count=row["session_count"],
        latest_timestamp=row["latest_timestamp"],
        session_ids=row["session_ids"].split(","),
    )
```

**Pattern confidence tiers:**

| session_count | Confidence label | Evolver action |
|---|---|---|
| 1 | isolated | Use standard per-session template |
| 2 | emerging | Use standard template; note cross-session signal |
| 3–5 | recurring | Generate targeted edit with higher confidence |
| > 5 | chronic | Generate targeted edit; flag for operator review |

The `CrossSessionPattern` model carries `session_count` and `session_ids` so
the `Evolver` can include a evidence snippet in the `ProposedEdit.rationale`
(e.g., "observed across 4 sessions: sessions s1, s2, s3, s4").

### 4. API for storing and querying patterns

```python
class FailurePatternStore:
    """SQLite-backed store for cross-session failure pattern accumulation."""

    def __init__(self, path: str | Path) -> None: ...

    def record(self, failure: FailureReport, resolution: str = "pending") -> None:
        """Insert a failure event into the pattern store.

        Called by the Digester after producing a FailureReport (issue #1038).
        Idempotent: the same (class, bucket, session_id, timestamp) row is
        inserted once; re-calling record() for the same event is a no-op.
        """

    def find_pattern(
        self,
        failure: FailureReport,
        min_sessions: int = PATTERN_MIN_SESSIONS,
        age_cutoff: datetime | None = None,
    ) -> CrossSessionPattern | None:
        """Query for a cross-session pattern matching the given failure."""

    def link_edit(self, pattern_id: int, edit_id: str, resolution: str) -> None:
        """Link a ProposedEdit to a pattern entry and update resolution."""

    def get_pattern_history(
        self,
        proposed_class: str,
        context_hash: str,
        limit: int = 10,
    ) -> list[FailurePatternEntry]:
        """Return the most recent pattern entries for audit/review."""
```

```python
class CrossSessionPattern(BaseModel):
    """Result of a cross-session pattern query."""

    proposed_class: str
    context_hash: str
    session_count: int
    latest_timestamp: str
    session_ids: list[str]


class FailurePatternEntry(BaseModel):
    """A single row in the failure_patterns table."""

    id: int
    proposed_class: str
    session_id: str
    timestamp: str
    context_hash: str
    resolution: str
    edit_id: str
    created_at: str
```

### 5. How accumulated patterns influence edit generation

The `Evolver` queries the store **before** generating a `ProposedEdit`.
The existing `propose()` / `propose_async()` flow is unchanged for isolated
failures (session_count = 1). For patterns with session_count ≥ 3:

```
1. Evolver.propose() or propose_async() is called with a FailureReport.
2. store.record(failure) is called to accumulate this session's failure.
3. store.find_pattern(failure) is called to check for cross-session signal.
4. If no pattern found (session_count < PATTERN_MIN_SESSIONS):
       Proceed with existing per-session flow (template or LLM).
   If pattern found (session_count >= PATTERN_MIN_SESSIONS):
       a. Retrieve CrossSessionPattern with session_ids and session_count.
       b. Inject cross-session evidence into the LLM prompt (or template):
          - "This failure class+context has been observed in N sessions:
            [session_ids]. The pattern is recurring and warrants a targeted edit."
       c. The Evolver generates a ProposedEdit with a richer rationale that
          references the pattern evidence.
       d. store.link_edit() is called to associate the edit with the pattern.
```

**Prompt augmentation (LLM path):**

The `_build_llm_prompt` method is extended to accept an optional
`cross_session_pattern: CrossSessionPattern | None`. When provided, the
prompt section "FAILURE REPORT" is preceded by:

```
CROSS-SESSION PATTERN DETECTED
==============================
This (failure class, context) combination has been observed in {session_count}
sessions: {session_ids[0]}, {session_ids[1]}, ... {session_ids[-1]}.
The pattern is classified as "{confidence_label}" and warrants a targeted,
high-confidence edit.
```

**Template path:** The template-based edit generator does not use the LLM
prompt, but the `rationale` field of the `ProposedEdit` is augmented:
`"[cross-session pattern: {session_count} sessions] {original_rationale}"`.

### 6. Store lifecycle and data retention

- **Recording:** `store.record()` is called by the `Digester` after each
  `digest()` call that returns a non-`clean` `FailureReport`.
- **Pattern query:** `store.find_pattern()` is called by the `Evolver`
  before each edit generation.
- **Retention:** Pattern rows are retained indefinitely but `age_cutoff`
  in `find_pattern()` limits which rows are considered for pattern detection.
  A background prune (operator-triggered via `foundry-trace prune`) drops
  rows older than the cutoff.
- **No deletion on resolution:** Pattern rows are immutable once written.
  `resolution` and `edit_id` fields capture the edit lifecycle; the raw
  pattern record persists for audit.

## Consequences

- **Proactive cross-session editing:** The `Evolver` now distinguishes
  between isolated failures (one session) and recurring patterns (≥3
  sessions). Recurring patterns generate higher-confidence edits backed by
  multi-session evidence.
- **New store dependency:** `FailurePatternStore` is a new caller on the
  SQLite trace store connection. It is initialized lazily on first use.
- **Backwards compatibility:** The per-session flow is unchanged when no
  pattern is detected. The ADR does not alter the existing `FailureReport`,
  `ProposedEdit`, or `ReviewStateMachine` schemas.
- **Trace store schema migration:** The `failure_patterns` table is created
  with `CREATE TABLE IF NOT EXISTS` so existing databases are not migrated.
  No destructive schema changes are introduced.
- **Pattern detection is advisory:** The `Evolver` always generates an edit
  for a non-`clean` `FailureReport`. The `CrossSessionPattern` augments the
  edit rationale and prompt but does not gate edit generation.
- **Cross-references:** See [ADR-0011](./0011-failure-report-class-taxonomy.md)
  for the `proposed_class` vocabulary. See `src/foundry_x/evolution/store.py`
  for the existing `ProposedEditStore` pattern that this new store mirrors.
  See `src/foundry_x/evolution/digester.py` for the `FailureReport` schema.
  See `src/foundry_x/evolution/evolver.py` for the `Evolver.propose()` flow
  this ADR extends.

## Implementation notes (non-normative)

The following are the recommended implementation steps but are not part of
this ADR's binding decision:

1. Add `FailurePatternStore` to `src/foundry_x/evolution/store.py`.
2. Add `context_hash_bucket()` as a pure function in `digester.py`.
3. Extend `Digester.digest()` to call `store.record(failure)` after producing
   a non-`clean` report (defer to issue #1038's implementation PR).
4. Extend `Evolver.propose_async()` to call `store.find_pattern()` and
   inject the result into `_build_llm_prompt()` and the template rationale.
5. Add `link_edit()` and `get_pattern_history()` to `FailurePatternStore`.
6. Add a `foundry-trace pattern-summary` CLI command to inspect accumulated
   patterns (post-implementation).
