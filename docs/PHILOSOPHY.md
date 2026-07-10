# Philosophy

This document states the beliefs that govern FoundryX. It is short on
purpose: if you find yourself violating a principle, stop and ask
whether the violation is justified — and if it is, propose changing the
principle first.

See also:

- [AGENTS.md](../AGENTS.md) — operational rules for AI collaborators.
- [CONTRIBUTING.md](../CONTRIBUTING.md) — operational rules for humans.
- [SECURITY.md](./SECURITY.md) — threat model and guardrails.
- [docs/adr/](./adr/) — recorded decisions.

## 1. Evidence over opinion

Every change to this codebase is justified by a trace, a benchmark, a
test, or a documented user need. "It feels cleaner" is not sufficient.
"The existing approach fails on this benchmark by 23%" is.

The trace store under `logs/` is the ground truth of what our agents
actually do. Read it before you speculate about what they should do.

## 2. Reversibility by default

Prefer changes that are cheap to undo. Small commits, gated rollouts,
no destructive migrations. The `Critic` exists to keep the harness
honest because the harness is the most expensive thing in this repo
to roll back.

If a change cannot be made reversible, the PR must say so explicitly
and include a rollback runbook.

## 3. Evaluation before change

No harness edit ships without passing the benchmark gate. No API
change ships without a regression test. We do not merge-and-pray; we
measure. See ADR-0004 and ADR-0005.

## 4. Humans and agents are peers here

This project is built by humans and AI agents working together. Both
contribute code, both file issues, both write docs. The difference is
in *what each is best at*:

- **Humans:** set direction, write ADRs, judge whether a change
  matches the philosophy, take responsibility when the harness
  misbehaves, and gate merges.
- **Agents:** read large surfaces fast, propose mechanical refactors,
  write tests, draft ADRs, summarize traces.

Both must follow the rules in `AGENTS.md` and `CONTRIBUTING.md`. The
harness enforces these on agents; humans enforce them on themselves.

## 5. Local-first, model-agnostic

The default target is the user's own hardware — an AMD RX 6600 XT, a
Ryzen 5 5600G, whatever you have. We test against `llama.cpp` because
it is the most permissive local runtime. We support any
OpenAI-compatible endpoint because lock-in is a tax on the user's
future options.

If a feature can only work with a closed hosted model, it goes behind
an opt-in flag and the default path stays local.

## 6. The harness is the artifact

In most repos, the code is the product and the docs are scaffolding.
Here it is the inverse: `harness/` (system prompt + hooks + skills)
is the product, and `src/foundry_x/` is the machinery that builds it.
Optimize accordingly:

- The harness is versioned, benchmarked, and evolved by machine.
- The machinery is versioned, tested, and evolved by humans.
- Mixing the two — e.g., hand-editing the harness to dodge a
  machinery bug — is a smell. Fix the machinery.

## 7. Optimize for evolvability, not cleverness

A piece of code that is elegant but impossible to evolve is a
liability. A piece of code that is plain but easy for the `Evolver`
to modify is an asset. When in doubt, choose the option that gives
the next change more room to move.

Concretely:

- Prefer data-driven configuration over hard-coded logic.
- Prefer explicit schemas (pydantic) over duck typing at module
  boundaries.
- Prefer flat hierarchies over deep inheritance.
- Prefer named, composable functions over metaprogramming.

## 8. The optimism budget is finite

We are excited about this project. Excitement produces scope creep.
Every new feature must answer: does this serve the PRD's KPIs (cycle
time, regression rate, improvement rate)? If not, it belongs in
`docs/ideas/`, not in `src/`.

## 9. The doctor is in the loop

This is a project that builds agents that write agents. If you ever
feel you cannot explain what a piece of code does in two sentences,
it is too clever. Simplify.
