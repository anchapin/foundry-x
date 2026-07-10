# CONTEXT.md

> Glossary of terms used throughout FoundryX. Reading this is part of
> onboarding for both human and AI contributors (see
> [AGENTS.md](../AGENTS.md) section 1 and
> [PHILOSOPHY.md](./PHILOSOPHY.md)).
>
> This document is the source of truth for project vocabulary. If you
> introduce a new term, add it here in the same PR that introduces the
> concept.

## The product

- **FoundryX** — the framework as a whole: the runtime, the evolution
  loop, and the conventions that hold it together.
- **FoundryAgent** — the runtime coding agent wrapped by the harness.
  Its persona and operating rules live in `harness/system_prompt.txt`.
- **Harness** — the artifact being evolved. Consists of the system
  prompt, hooks, and skills. Version-controlled, evolved by the
  `Evolver`, gated by the `Critic`. Per PHILOSOPHY.md §6, the harness
  is the product.
- **Foundry** — the Python package (`src/foundry_x/`) that wraps and
  evolves the harness. Built and maintained primarily by humans.

## Subsystems

- **TraceLogger** — wraps an agent session and persists every prompt,
  tool call, and outcome to a structured trace store. The ground-truth
  recorder. (`src/foundry_x/trace/logger.py`)
- **Runner** — drives a single agent session against a task. Reads the
  harness, calls the model, writes the trace.
  (`src/foundry_x/execution/runner.py`)
- **Digester** — parses a trace (or set of traces) and produces a
  failure report: what failed, where, and the candidate root cause.
  (`src/foundry_x/evolution/digester.py`)
- **Evolver** — a meta-agent that takes a failure report and proposes
  a `ProposedEdit` against the harness.
  (`src/foundry_x/evolution/evolver.py`)
- **Critic** — the gatekeeper. Runs the proposed edit through the
  pytest suite and benchmark suite; rejects regressions.
  (`src/foundry_x/evolution/critic.py`)
- **ProposedEdit** — a structured `pydantic` model representing a
  proposed change to the harness. The unit of work produced by the
  Evolver and consumed by the Critic.

## The loop

```
  task -> Runner -> trace
                   |
                   v
              Digester -> failure report
                              |
                              v
                          Evolver -> ProposedEdit
                                            |
                                            v
                                       Critic -> accept | reject
                                                       |
                                                       v
                                                   harness (updated)
```

A single iteration is small. The value comes from running the loop
many times per day against a benchmark suite.

## Concepts

- **meta-agent** — an agent that operates on another agent's artifacts
  rather than on the end task; in FoundryX the `Evolver` is the
  meta-agent that turns failure reports into `ProposedEdit`s against
  the harness.
- **failure report** — the structured artifact produced by the
  `Digester` from a trace, naming what failed, where, and the
  candidate root cause; consumed by the `Evolver` as the basis for a
  `ProposedEdit`.

## Artifacts on disk

- `harness/system_prompt.txt` — the agent's persona and operating rules.
- `harness/hooks/*.py` — middleware that runs around every tool call.
- `harness/skills/*.json` — tool definitions the agent can invoke.
- `logs/` — trace store (gitignored). Per-run SQLite databases plus
  exports.
- `benchmarks/` — pytest-marked tasks the Critic uses to gate harness
  changes.
- `docs/adr/` — recorded architecture decisions, numbered sequentially.
- `docs/ideas/` — design ideas not yet accepted into the project.

## Roles

- **Operator** — the human running the harness against their tasks.
- **Engineer** — the human maintaining `src/foundry_x/` (the foundry).
- **Agent** — an AI collaborator (e.g., Claude, GPT, or a local
  model driven by FoundryX itself).
- **Critic** — see above; in role terms, also the regression-tester.

The Operator and Engineer may be the same person. The Agent is always
external to the runtime under test.

## Verbs

These verbs structure the workflow described in `AGENTS.md` §3 and
`CONTRIBUTING.md`. Use them in PR titles and commit bodies when
relevant.

- **Observe** — read traces and existing code before proposing.
- **Digest** — turn observations into a failure report.
- **Propose** — produce a `ProposedEdit` (or, for a human, a PR).
- **Evaluate** — run the test + benchmark gate.
- **Commit** — atomic, conventional-commits change.
- **Hand off** — open a PR and wait for review.
