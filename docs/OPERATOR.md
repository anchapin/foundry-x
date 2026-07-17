# OPERATOR.md

> A guide for human operators running the FoundryX harness against their tasks.
> For the visual loop diagram, see the [loop diagram in CONTEXT.md](./CONTEXT.md#the-loop).
> For the full glossary of terms, see [Glossary](#glossary) below.

## What is an operator?

An **operator** is a human who runs the FoundryX harness against their own
tasks. The operator observes traces, identifies failure patterns, and — when
the harness needs changes — routes a failure report through the Evolver ->
Critic loop rather than editing the harness directly by hand.

## The Evolution Loop

FoundryX runs a continuous improvement loop:

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

A single iteration is small. The value comes from running the loop many times
per day against a benchmark suite.

## Key Concepts

### Harness

The artifact being evolved. The harness is the product — it defines the
agent's persona, operating rules, hooks, and tool surface. See
[PHILOSOPHY.md](./PHILOSOPHY.md) §6.

### Evolver

The meta-agent that takes a failure report and proposes a ProposedEdit
against the harness. It is the component that proposes harness mutations.

### Critic

The gatekeeper. The Critic runs the proposed edit through the pytest suite
and benchmark suite, then rejects any edit that causes regressions.

### ProposedEdit

A structured pydantic model representing a proposed change to the harness.
It is the unit of work produced by the Evolver and consumed by the Critic.

### Regression Gate

The Critic's benchmark gate. Every harness edit must pass the regression gate
before it is marked active. If any benchmark flips red, the edit is rejected.

### Runner

The component that drives a single agent session against a task. The Runner
reads the harness, calls the model, and writes the trace.

### TraceLogger

The component that wraps an agent session and persists every prompt, tool
call, and outcome to the trace store. It is the ground-truth recorder.

### Digester

The component that parses traces and produces a failure report: what failed,
where, and the candidate root cause.

### Failure Report

The structured artifact produced by the Digester from a trace. It names what
failed, where, and the candidate root cause. The failure report is consumed
by the Evolver as the basis for a ProposedEdit.

### Hook

Middleware that runs around every tool call. Hooks live in
`harness/hooks/*.py` and are loaded by the Runner at session start.

### Benchmark Task

A standardised coding task marked with `@pytest.mark.benchmark`. The Critic
uses benchmark tasks to gate harness changes — a regression in any benchmark
flips the Critic red.

### Foundry

The Python package (`src/foundry_x/`) that wraps and evolves the harness.
Built and maintained primarily by humans. Distinct from the harness, which is
the artifact being evolved.

### FoundryAgent

The runtime coding agent persona as defined in the harness. Its persona and
operating rules live in `harness/system_prompt.txt`.

## The Operator Workflow

Operators follow a six-step cycle (see AGENTS.md §3):

1. **Observe** — read traces and existing code before proposing.
2. **Digest** — turn observations into a failure report.
3. **Propose** — produce a ProposedEdit (or, for a human, a PR).
4. **Evaluate** — run the test and benchmark gate (Critic).
5. **Commit** — atomic, conventional-commits change.
6. **Hand off** — open a PR and wait for review.

## Glossary {#glossary}

| Term | Definition |
|------|------------|
| **Benchmark Task** | A standardised coding task marked `@pytest.mark.benchmark`; the Critic gates harness changes against the full benchmark suite. |
| **Critic** | The gatekeeper that runs the proposed edit through the pytest suite and benchmark suite, rejecting any edit that causes regressions. |
| **Digester** | The component that parses traces and produces a failure report naming what failed, where, and the candidate root cause. |
| **Evolver** | The meta-agent that takes a failure report and proposes a ProposedEdit against the harness. |
| **Failure Report** | A structured artifact produced by the Digester; consumed by the Evolver as the basis for a ProposedEdit. |
| **Foundry** | The Python package (`src/foundry_x/`) that wraps and evolves the harness. Distinct from the harness itself. |
| **FoundryAgent** | The runtime coding agent persona as defined in `harness/system_prompt.txt`. |
| **Harness** | The artifact being evolved: the system prompt, hooks, and skills. The harness is the product. |
| **Hook** | Middleware that runs around every tool call; lives in `harness/hooks/*.py`. |
| **Meta-agent** | An agent that operates on another agent's artifacts rather than on the end task. In FoundryX the Evolver is the meta-agent. |
| **Operator** | The human who runs the FoundryX harness against their own tasks. |
| **ProposedEdit** | A structured pydantic model representing a proposed change to the harness; the unit of work produced by the Evolver and consumed by the Critic. |
| **Regression Gate** | The Critic's benchmark gate. Every harness edit must pass the regression gate before it is marked active. |
| **Runner** | The component that drives a single agent session against a task, reads the harness, calls the model, and writes the trace. |
| **Trace** | A recorded sequence of events (prompts, tool calls, outcomes) produced by the Runner and persisted by the TraceLogger. |
| **TraceLogger** | The component that wraps an agent session and persists every prompt, tool call, and outcome to the trace store. |
