# ADR-0005: Use pytest as the unified evaluation framework

## Status

Accepted. 2026-07-10.

## Context

The project needs two things that look different but are
structurally the same:

- unit and integration tests for the *machinery* in
  `src/foundry_x/`
- benchmark tasks for the *agent* in `harness/`

Maintaining separate runners doubles the harness surface and the
maintenance cost.

## Decision

We use `pytest` for both. Conventional unit and integration tests
live under `tests/`. Benchmark tasks live under `benchmarks/` as
pytest test cases with a custom marker
(`@pytest.mark.benchmark`) so they can be selected or excluded.

The `Critic` invokes pytest under the hood (ADR-0004); humans invoke
pytest for local checks.

## Consequences

- One mental model: "everything is a pytest."
- The benchmark suite scales with pytest's existing parallelism
  (`pytest-xdist`, not yet enabled).
- Benchmark failures are indistinguishable from test failures in CI;
  this is intentional — a failing benchmark is a failing test.
- We do not adopt a separate benchmark framework (e.g., `inspect`,
  `promptfoo`) unless we hit a concrete limitation, in which case
  we write an ADR explaining why.