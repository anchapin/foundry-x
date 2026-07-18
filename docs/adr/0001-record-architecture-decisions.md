# ADR-0001: Record architecture decisions

## Status

Accepted. 2026-07-10.

## Context

Decisions about dependency choice, schema design, evolution policy,
and runtime architecture accumulate quickly. Without a record, the
same arguments are re-litigated and the rationale for past choices
is lost when the people involved move on.

## Decision

We record every non-trivial architecture decision as an ADR in
`docs/adr/`, numbered sequentially with a short kebab-case slug:

```
docs/adr/NNNN-kebab-case-title.md
```

Each ADR follows this lightweight template:

- **Status**: `Proposed` | `Accepted` | `Superseded` | `Deprecated`
- **Context**: the situation that forced the choice.
- **Decision**: what we chose.
- **Consequences**: trade-offs, follow-ups, what this forecloses.

Lightweight MADR-style. We do not use `adr-tools` or a generator —
files are plain Markdown and reviewable in any diff.

### ADR writing conventions

All ADRs MUST use concrete names from the actual codebase:

- **File paths**: real paths like `src/foundry_x/trace/logger.py`, not
  `src/foo/bar.py`
- **Class/function names**: actual symbols like `TraceLogger`, `HookRegistry`,
  `FailureReport`, not `Foo`, `Bar`, `baz()`
- **Code examples**: real code from the repo, not invented snippets
- **Line references**: actual line numbers from the source, verified at write time

This ensures ADRs serve as accurate documentation — a reader can grep the
codebase and find exactly what the ADR describes. Abstract or placeholder
examples create a disconnect between documentation and reality that misleads
future contributors.

## Process

- **Proposed→Accepted**: mark `Accepted` in the same PR that ships the
  implementation. An ADR MUST NOT remain `Proposed` after its change
  lands.
- New ADRs land in the same PR as the change they justify when
  practical. Otherwise the ADR precedes the change.
- Superseded ADRs are not deleted; their `Status` is updated and a
  pointer is added to the replacement.
- Reading the `docs/adr/` directory should give a new contributor a
  faithful map of how this codebase came to be.
- ADR review is human-only — agents may draft ADRs but a human must
  approve them.

## Consequences
