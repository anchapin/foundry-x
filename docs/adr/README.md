# Architecture Decision Records

This directory contains the architecture decision records (ADRs) for
FoundryX. ADRs capture *why* a decision was made, not just *what* was
decided, so future contributors can understand the constraints that
shaped the codebase.

See [ADR-0001](./0001-record-architecture-decisions.md) for the
format and process.

## Index

| Number | Title | Status |
| ------ | ----- | ------ |
| [0001](./0001-record-architecture-decisions.md) | Record architecture decisions | Accepted |
| [0002](./0002-uv-for-dependency-management.md) | Use `uv` for dependency management | Accepted |
| [0003](./0003-sqlite-as-trace-store.md) | SQLite as the default trace store | Accepted |
| [0004](./0004-self-modification-guardrails.md) | Self-modification guardrails via the Critic gate | Accepted |
| [0005](./0005-pytest-as-evaluation-framework.md) | Use pytest as the unified evaluation framework | Accepted |
| [0006](./0006-pydantic-for-module-boundaries.md) | Use pydantic for structured data at module boundaries | Accepted |
| [0007](./0007-trace-driven-development.md) | Trace-driven development is the default | Accepted |
| [0008](./0008-conventional-commits-and-adr-discipline.md) | Conventional Commits and ADR discipline | Accepted |

When adding a new ADR, append it to this table in the same PR.
