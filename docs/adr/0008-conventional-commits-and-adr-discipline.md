# ADR-0008: Conventional Commits and ADR discipline

## Status

Accepted. 2026-07-10.

## Context

A self-improving project generates a high volume of small changes,
many of them produced by the Evolver. Without disciplined commit
messages, the git history becomes unsearchable and automated tooling
(changelog generation, release tagging, evolution auditing) breaks.
Without recorded decisions, the same debate recurs every quarter.

## Decision

Two rules, applied to every PR:

1. **Conventional Commits.** Subject line is `type(scope): summary`
   in the imperative mood, no trailing period, 50 chars or fewer.
   Allowed types: `feat`, `fix`, `chore`, `docs`, `refactor`,
   `test`, `perf`, `build`, `ci`. Breaking changes append `!` after
   the scope and a `BREAKING CHANGE:` footer.
2. **ADRs for non-trivial decisions.** Any decision that affects
   architecture, dependencies, harness policy, or public API gets
   an ADR in `docs/adr/`. The ADR may land in the same PR as the
   code it justifies, or in a preceding PR.

Lightweight enforcement:

- The PR template reminds contributors of both rules.
- Reviewers block merges on clear violations.
- The pre-commit framework does not enforce these rules; they are
  review-time gates, not pre-commit gates.

## Consequences

- Git history is grep-able by type and scope.
- ADRs and code drift together; superseded ADRs are kept and
  cross-referenced.
- Agents writing PRs must follow both rules. AGENTS.md §5 codifies
  this for AI collaborators.
- Automated changelog generation becomes trivial later without
  retroactive history rewriting.
- See ADR-0001 for the ADR format itself.