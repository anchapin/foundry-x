# Ideas

## Status

**Retired.** This pipeline was documented in 2026 but never exercised.
No idea files were ever created. Keeping an unused process in the tree
creates maintenance burden and misleads contributors into thinking the
pipeline is active. See issue #645.

The directory is preserved in read-only form so any existing links to it
remain valid. To propose a design idea, open a GitHub issue instead.

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

1. Add or update an ADR in `../adr/` capturing the decision.
2. Either move the idea file into `../adr/` (rename and merge) if
   the ADR supersedes it, or delete it if the ADR fully captures
   the intent.
3. Implement in a follow-up PR.

When an idea is rejected:

1. Update its status to `Rejected` with a one-line reason.
2. Leave the file in place so future contributors see it was
   considered.

## Ideas index

_None yet._
