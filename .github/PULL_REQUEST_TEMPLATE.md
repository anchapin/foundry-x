<!--
  FoundryX — Pull Request template

  Enforces the two review-time gates from ADR-0008 and the evidence
  rule from ADR-0007. Fill in every section; reviewers block merges
  on gaps. See AGENTS.md §5 and CONTRIBUTING.md for the full rules.

  ADR-0008: subject is Conventional Commits; ADRs for non-trivial choices.
  ADR-0007: behavior changes cite a trace excerpt / test / benchmark.
-->

## Motivation

<!-- Why is this change needed? What problem does it solve? -->

Resolves #

## Evidence

<!--
  ADR-0007: behavior-changing PRs must include a trace excerpt, a
  failing test that this change makes pass, or a benchmark result.
  Pure refactors that do not change behavior may write "n/a: refactor".
-->

## Change

<!-- One paragraph: what the diff does, and why this approach. -->

## Risk

<!-- What could break? Is the change reversible? Rollback path? -->

## ADRs

<!--
  ADR-0008: list every ADR this PR advances, supersedes, or creates.
  Write "none" if this is a trivial change that needs no decision record.
-->

## Checklist

- [ ] Commit subject is Conventional Commits — `type(scope): summary`,
      imperative mood, no trailing period, 50 chars or fewer (ADR-0008)
- [ ] ADR recorded for any non-trivial decision (ADR-0008)
- [ ] Evidence section cites a trace / test / benchmark (ADR-0007)
- [ ] Diff is focused — one concern, ideally under 400 lines (AGENTS.md §5)
- [ ] No secrets, no `.env` contents (docs/SECURITY.md)
- [ ] This PR does **not** hand-edit `harness/system_prompt.txt`,
      `harness/hooks/*`, or `harness/skills/*` (route harness changes
      through the Evolver -> Critic loop; ADR-0004)
