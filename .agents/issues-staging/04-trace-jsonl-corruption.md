## Motivation

Five JSONL read-path methods in `TraceLogger` call `json.loads()` without try/except. A single corrupted line (partial write from SIGKILL, OOM, or disk-full during append) raises an unhandled `JSONDecodeError` that propagates to the caller, making the ENTIRE JSONL store unreadable. The operator cannot run `foundry-x-trace show <sid>`, `events-grep`, `session-stats`, `foundry-kpis`, or `fx-trace regression-report` against the store.

In contrast, the three write/rewrite methods in the same file already wrap `json.loads` in `try/except JSONDecodeError` and continue past bad lines. This asymmetry means the trace store can survive corruption when rewriting itself but crashes when an operator merely tries to read it.

This is a data-integrity issue: when the JSONL backend is corrupted, all three PRD KPIs become unmeasurable because `compute_cycle_time` (kpis.py:676) and `compute_regression_report` (regression_report.py:118) both call `query_events`, which would crash on a corrupted line. `kpi-cycle-time` is the primary KPI because cycle-time is computed directly from the timestamps in trace events that become inaccessible.

## Evidence

- `src/foundry_x/trace/logger.py:761` — `_list_sessions_jsonl`: bare `json.loads(line)`, no try/except
- `src/foundry_x/trace/logger.py:932` — `_load_session_jsonl`: bare `json.loads(stripped)`, no try/except
- `src/foundry_x/trace/logger.py:1216` — `_redact_event_jsonl`: bare `json.loads(stripped)`, no try/except
- `src/foundry_x/trace/logger.py:1247` — `_iter_events_jsonl`: bare `json.loads(line)`, no try/except
- `src/foundry_x/trace/logger.py:1279` — `_query_events_jsonl`: bare `json.loads(line)`, no try/except
- `src/foundry_x/trace/logger.py:1024-1028` — `_prune_jsonl`: DOES handle `JSONDecodeError` with try/except (establishes expected pattern)
- `src/foundry_x/trace/logger.py:1077-1081` — `_delete_session_jsonl`: DOES handle `JSONDecodeError` (same pattern)
- `src/foundry_x/trace/logger.py:1132-1136` — `compact()`: DOES handle `JSONDecodeError` (same pattern)
- `src/foundry_x/observability/kpis.py:676` — `compute_cycle_time` calls `logger.query_events(kind='task_received')` (would crash on corrupted JSONL)
- `src/foundry_x/observability/regression_report.py:118` — `compute_regression_report` calls `logger.query_events(kind=VERDICT_KIND)` (would crash)
- No test in `tests/test_trace*.py` exercises a corrupted/partial JSONL line against any read-path method

## Risk

Low. Corrupted lines are skipped (not deleted); no data is destroyed. The corrupted line remains in the file — read methods skip it, they do not rewrite. This matches the existing write-side precedent.

## Acceptance Criteria

1. `list_sessions()` on a JSONL file containing one corrupted line returns sessions from valid lines without raising
2. `load_session(sid)` on a JSONL file containing one corrupted line returns events from valid lines without raising
3. `iter_events(sid)` on a JSONL file containing one corrupted line yields events from valid lines without raising
4. `query_events()` on a JSONL file containing one corrupted line yields events from valid lines without raising
5. `redact_event(sid, 0, key)` on a JSONL file containing one corrupted line succeeds on a valid event without raising
6. A test writes a valid event, appends a corrupted line, writes a second valid event, and asserts all five read-path methods return the two valid events without raising

## ADR(s)

ADR-0007 — advances: "traces must be inspectable" — read paths become resilient to partial writes
ADR-0003 — complies: JSONL backend contract unchanged; only error handling added
