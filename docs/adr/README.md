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
| [0009](./0009-security-evals-benchmark-family.md) | Security-evals BenchmarkTask family | Accepted |
| [0010](./0010-runner-agent-loop.md) | Runner agent loop | Accepted |
| [0011](./0011-failure-report-class-taxonomy.md) | Digester `FailureReport` class taxonomy | Accepted |
| [0012](./0012-manifest-json-as-evolver-target.md) | Expand Evolver target confinement to include `manifest.json` | Accepted |
| [0013](./0013-wal-mode-and-connection-reuse.md) | WAL mode and a single reused SQLite connection | Accepted |
| [0014](./0014-model-abstraction.md) | Model abstraction layer for model-swapping milestone | Proposed |

When adding a new ADR, append it to this table in the same PR.
