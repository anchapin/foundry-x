# Ideas

This directory holds design ideas, proposals, and explorations that
have not yet been accepted into the project. They live here so that
we preserve reasoning without committing to implementation.

## Why this exists

Per [PHILOSOPHY.md](../PHILOSOPHY.md) §8 ("The optimism budget is
finite"), new ideas must answer: does this serve the PRD's KPIs
(cycle time, regression rate, improvement rate)? If not, the idea
belongs here, not in `src/`.

## How to add an idea

1. Create `NNNN-short-title.md` (any short slug; sequential numbering
   is not enforced, but please avoid collisions).
2. Fill in the sections below. Keep it short — a paragraph per
   section is plenty for most ideas.
3. Open a PR. An idea PR is a discussion surface, not a commitment.
   The bar is "would a reasonable engineer want to read this?" not
   "do we agree with it?"

### Template

```markdown
# Idea NNNN: <short title>

## Author
<you, with date>

## Status
Proposed | Approved | Rejected

## Problem
What is missing or broken today?

## Proposal
What would we change? At what cost?

## KPIs affected
Which of cycle time / regression rate / improvement rate does this
move, and in which direction?

## Open questions
What would have to be true for this to be a good idea?
```

## How ideas graduate

When an idea is approved:

1. Update the `Status` field in the idea file to `Approved`.
2. Add or update an ADR in `../adr/` capturing the decision.
3. Either move the idea file into `../adr/` (rename and merge) if
   the ADR supersedes it, or delete it if the ADR fully captures
   the intent.
4. Implement in a follow-up PR.

When an idea is rejected:

1. Update the `Status` field in the idea file to `Rejected`.
2. Leave the file in place so future contributors see it was
   considered.

The ideas index in this README is updated automatically by
`scripts/update_ideas_index.py` (run as a pre-commit hook).

## Ideas index

<!-- Do not edit this table by hand. It is updated by
     `scripts/update_ideas_index.py`, which runs automatically as a
     pre-commit hook and can be run manually with:
       python scripts/update_ideas_index.py -->

| Number | Title | Status |
| ------ | ----- | ------ |
<!-- entries inserted by scripts/update_ideas_index.py -->
