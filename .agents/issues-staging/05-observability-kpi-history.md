## Motivation

`append_kpi_history` (kpis.py:1633) persists the full `KpiSummary` model dump to the JSONL history log — including the scalar auxiliary signals `model_retry_count`, `tool_argument_parse_error_count`, `event_limit_abort_count`, `server_restart_count`, `token_budget_abort_count`, and `token_budget_hit_rate`. However, `KpiHistoryEntry` (kpis.py:351) only declares 10 fields. Because pydantic's default `extra='ignore'` policy silently drops unknown keys, `read_kpi_history` returns entries where all 6 auxiliary signals are missing. **The data is written to disk and never read back.**

This was verified live: a `KpiSummary` with `model_retry_count=7` and `event_limit_abort_count=3` was persisted via `append_kpi_history`; the JSONL line on disk carries both keys, but `read_kpi_history` returns a `KpiHistoryEntry` where neither field exists in `model_fields`. The existing round-trip test (`test_append_kpi_history_round_trips_through_kpi_summary`, test_kpi_history.py:108) masks the bug because it parses the raw JSONL line as `KpiSummary` (which has all fields) rather than going through `read_kpi_history` -> `KpiHistoryEntry` (the actual CLI path).

`render_history_markdown` (kpis.py:1743) renders only the 3 PRD KPI columns, so even the signals that DO survive the round-trip (`hooks_disabled_*`) are invisible in the trend table. An operator running `foundry-kpis --from-history kpi-history.jsonl` cannot see whether model retries, event-limit aborts, or server restarts are trending up across harness edits — exactly the diagnostic view these signals were designed to provide (issues #869, #871, #872, #899).

Fixing this round-trip improves measurement fidelity: when `cycle_time_seconds` drifts up in the trend table, the operator can correlate it with reliability signals (retries, aborts, restarts) to distinguish infrastructure regression from model-quality regression. Those signals are currently computed and persisted but silently lost on read-back, blinding cycle-time diagnosis.

## Evidence

- `src/foundry_x/observability/kpis.py:351-382` — `KpiHistoryEntry` declares only 10 fields; the 6 auxiliary scalar fields are absent from `model_fields`
- `src/foundry_x/observability/kpis.py:1633-1687` — `append_kpi_history` calls `summary.model_dump(exclude={...})`; the 6 auxiliary fields are NOT excluded and are written to disk
- `src/foundry_x/observability/kpis.py:1690-1713` — `read_kpi_history` parses each line via `KpiHistoryEntry.model_validate_json`, which drops unknown keys via pydantic `extra='ignore'`
- `src/foundry_x/observability/kpis.py:1743-1812` — `render_history_markdown` renders only Cycle Time / Regression Rate / Improvement Rate columns
- `tests/test_kpi_history.py:108` — round-trip test parses through `KpiSummary` (all fields) instead of `read_kpi_history` -> `KpiHistoryEntry` (drops fields), masking the bug
- Live verification 2026-07-26: `KpiSummary(model_retry_count=7, event_limit_abort_count=3)` -> `append_kpi_history` -> `read_kpi_history` — JSONL line on disk contains both keys; `KpiHistoryEntry` read-back has neither field

## Risk

Low. Adding fields to `KpiHistoryEntry` with defaults is backward-compatible — old JSONL entries simply lack the fields and they default to 0/0.0. Two dead fields (`injection_blocks`, `wall_clock_abort_count`) are removed from `KpiHistoryEntry` because they are in the exclude set and never written; any consumer already sees defaults.

## Acceptance Criteria

1. `read_kpi_history` on a JSONL line with `model_retry_count=7` returns a `KpiHistoryEntry` whose `model_retry_count == 7` (not dropped)
2. `read_kpi_history` on a JSONL line with `event_limit_abort_count=3` returns a `KpiHistoryEntry` whose `event_limit_abort_count == 3`
3. A round-trip test calls `append_kpi_history` then `read_kpi_history` (NOT `KpiSummary.model_validate_json`) and preserves all 6 auxiliary fields
4. `render_history_markdown` includes a "Reliability Signals" section when at least one entry has a non-zero auxiliary signal
5. `render_history_markdown` with all-zero auxiliary signals omits the section (compact when clean)
6. `context_pruned_count` (a per-session dict) no longer appears in the JSONL history line (added to exclude set)
7. Existing trend-table output (3 PRD KPI columns + sparklines) is byte-identical when no auxiliary signals are non-zero

## ADR(s)

ADR-0007 — advances: trace-driven development requires that persisted observations are faithfully retrievable; silently dropping written KPI data on read-back violates the "trace store is ground truth" principle
