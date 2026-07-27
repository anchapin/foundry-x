# ADR-0034: Define the "smoke" DifficultyTier

## Status

Accepted. 2026-07-27.

## Context

ADR-0005 defines the benchmark framework and ADR-0006 established the
`BenchmarkTask` schema.  `benchmarks/models.py` defines
`DifficultyTier = Literal["smoke", "easy", "medium", "hard"]`.

ADR-0028 defined the **hard** tier.  No ADR defined what **smoke** means,
even though five tasks (`web_fetch_skill`, `server_unavailable`,
`smoke_marker_and_fixture_resolve`, `implementation_fizzbuzz`,
`quantization_v4_sweep`) declared `difficulty_tier="smoke"` — a mix of
infrastructure hygiene checks, server health checks, and a trivial
algorithm task that had nothing in common.

Issue #1119 identified this gap: without a definition, the smoke tier
cannot be reliably sliced in `foundry-kpis --group-by difficulty_tier`,
cannot be excluded from the improvement-rate KPI (ADR-0005 §Consequences),
and tasks can be assigned to it arbitrarily.

## Decision

### 1. Define what makes a task "smoke"

A task belongs to the **smoke** tier when it satisfies **either** of:

1. **Pure infrastructure / hygiene check** — the task runs in under 5
   seconds, performs no model inference, and verifies that a core
   runtime component (trace system, hook wiring, manifest registration,
   benchmark infrastructure) is correctly configured.  These tasks are
   executed by the Critic gate itself before any agent session starts.

2. **Single-function smoke test** — a deterministic, self-contained
   check that exercises exactly one function or code path with a known
   input and a binary pass/fail outcome.  No multi-step reasoning, no
   tool-call sequencing, no fixture seeding beyond a trivial input file.

Smoke tasks are **excluded from the improvement-rate KPI** (ADR-0005)
because they do not exercise agent capability.  A harness that passes
all smoke tasks but fails all easy/medium/hard tasks has not improved.
Smoke tasks **are** included in the **regression-rate KPI** because a
smoke-task failure indicates a broken pipeline, not an agent regression.

### 2. Tier boundary: smoke vs. easy

| Property | smoke | easy |
|---|---|---|
| Agent inference | None | Full model invocation |
| Typical runtime | < 5 s | 30 – 120 s |
| Tool-call rounds | 0 | 1 – 3 |
| Multi-step reasoning | No | No |
| KPI: improvement-rate | Excluded | Included |
| KPI: regression-rate | Included | Included |
| Fixture complexity | None / trivial | Seeded workspace |

### 3. Smoke-tier task registry

| Task name | Reason it is smoke | Remains smoke? |
|---|---|---|
| `server_unavailable` | No agent; exercises Runner abort path via stub | Yes |
| `web_fetch_skill` | Offline unit test of hook allowlist logic | Yes |
| `smoke_marker_and_fixture_resolve` | pytest discovery + fixture hygiene only | Yes |
| `quantization_v4_sweep` | Infrastructure sweep; no agent | Yes |
| `implementation_fizzbuzz` | Trivial algorithm; **should be easy** | **Reclassified → easy** |

### 4. Consequences for existing tasks

`implementation_fizzbuzz` is reclassified to `easy` (issue #1119) because
it requires a full agent session with tool calls to read `input.txt` and
write `output.txt`, even though the algorithm is trivial.  It exercises the
read/write tool loop, not infrastructure wiring.

## Consequences

- **Positive**: Smoke tier now has an explicit definition enabling correct
  KPI slicing (`foundry-kpis --group-by difficulty_tier` produces a valid
  smoke-tier slice).
- **Positive**: Smoke tasks are documented as excluded from the
  improvement-rate KPI per ADR-0005 §Consequences, eliminating the
  conflation of pipeline failures with capability regressions.
- **Positive**: Infrastructure health is now a first-class tier,
  making the benchmark suite's tier ladder complete (smoke → easy →
  medium → hard).
- **Negative**: Reclassifying `implementation_fizzbuzz` from smoke to easy
  changes the easy-tier baseline; existing easy-tier pass rates will
  shift slightly.

## Implementation plan

1. Add this ADR (ADR-0034).
2. Update the `DifficultyTier` docstring in `benchmarks/models.py` to
   document all four tiers.
3. Reclassify `implementation_fizzbuzz` to `difficulty_tier="easy"` in
   `benchmarks/tasks/test_implementation_fizzbuzz.py`.
4. Verify `foundry-kpis --group-by difficulty_tier` produces a valid
   smoke-tier slice.
