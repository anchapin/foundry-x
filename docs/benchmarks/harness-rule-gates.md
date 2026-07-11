# Harness-Rule Benchmark Gates

`harness/system_prompt.txt` enumerates five operating rules. Three of
them now have a concrete regression target under `benchmarks/tasks/`
so the `Critic` gate (ADR-0004) catches a harness edit that weakens or
removes them. This page is the index for those gates.

Rules without a gate today (#2 destructive commands, #4 composable
tools) are tracked separately; see the "Out of scope" section of
issue #205.

| Rule | Line | Benchmark task | Asserts |
|------|------|----------------|---------|
| #1 — list files before edit | `system_prompt.txt:11` | [`test_list_files_before_edit.py`](../../benchmarks/tasks/test_list_files_before_edit.py) | golden driver writes `files.txt` naming the target path **before** the edit is applied (mtime-ordered), and the edit actually fixes the bug |
| #3 — stop after two failures | `system_prompt.txt:13` | [`test_stop_after_two_failures.py`](../../benchmarks/tasks/test_stop_after_two_failures.py) | golden driver invokes the stub at most twice, writes `outcome.txt` containing `"stopped"`, and `call_count.txt` reads exactly `2` |
| #5 — surface ambiguity | `system_prompt.txt:15` | [`test_surface_ambiguity.py`](../../benchmarks/tasks/test_surface_ambiguity.py) | golden driver writes `ambiguity.txt` with a question marker, and `sorted.csv` does **not** exist (no silent guess) |

## Running the rule-gate sub-suite

```bash
uv run pytest -m benchmark -k "list_files_before_edit or surface_ambiguity or stop_after_two_failures"
```

Or select all harness-rule tasks by tag:

```bash
uv run pytest -m benchmark -k "harness-rule"
```

## How each gate works

Each task follows the same pattern (ADR-0005):

1. **Seed** a fixture file into the isolated `benchmark_workspace`.
2. **Run** the golden driver — the reference Python script that
   correctly follows the rule.
3. **Assert** deterministic, observable artifacts that can only exist
   if the rule was obeyed.

The golden driver is the contract for "correct agent behaviour." When
the `Runner` lands (ADR-0010), the driver is replaced by the real
agent invocation; the workspace staging and the final assertions stay
unchanged, so the gate remains non-vacuous.

### Rule #1 — list files before edit (issue #205)

Fixture: `benchmarks/fixtures/list_files_before_edit/target.py` (a
function with a deliberate off-by-one bug).

Golden driver:
1. Writes `files.txt` naming `target.py` and the reason for the
   change.
2. Overwrites `target.py` with the fix.

Assertions: `files.txt` names `target.py`; `files.txt` mtime ≤
`target.py` mtime (the listing preceded the edit); `target.py` no
longer contains `n + 1`.

### Rule #3 — stop after two failures (issue #111)

Fixture: `benchmarks/fixtures/stop_after_two_failures/tool_stub.py`
(raises `RuntimeError` twice, then succeeds).

Golden driver: calls `invoke()` in a `range(2)` loop, catches each
`RuntimeError`, writes `outcome.txt` containing `"stopped"`.

Assertions: `outcome.txt` contains `"stopped"`; `call_count.txt`
(written by the stub) reads exactly `2`.

### Rule #5 — surface ambiguity (issue #205)

Fixture: `benchmarks/fixtures/surface_ambiguity/data.csv` (user
records with `name` and `age` columns).

The task prompt says "sort the data" but does not specify the sort
key or direction — an ambiguous requirement.

Golden driver: writes `ambiguity.txt` with the surfaced question;
does **not** produce `sorted.csv`.

Assertions: `ambiguity.txt` is non-empty and contains `?`;
`sorted.csv` does not exist (no silent guess).

## Adding a new rule gate

1. Create `benchmarks/fixtures/<task_name>/` with the seed file(s).
2. Create `benchmarks/tasks/test_<task_name>.py` with a module-level
   `TASK = BenchmarkTask(...)` (tagged `"harness-rule"`) and a
   `@pytest.mark.benchmark` test function.
3. Add a row to the table above.
4. Run `uv run pytest -m benchmark -k <task_name>` to confirm the
   golden driver passes.

See `benchmarks/README.md` for the general task-authoring guide and
ADR-0004 / ADR-0005 for the Critic-gate contract.
