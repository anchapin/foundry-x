# ADR-0012: Expand Evolver target confinement to include `manifest.json`

## Status

Accepted. 2026-07-11.

## Context

ADR-0004 confines `ProposedEdit` targets to exactly three harness entries:
`system_prompt.txt`, `hooks/*`, and `skills/*`. When the Evolver adds a
new skill file, it cannot also update `harness/manifest.json` to declare
that skill in the `"skills"` array. The manifest drifts on every evolution
that touches the skills tree, requiring manual sync (tracked in
issue #202). Additionally, when the Evolver adds, reorders, or removes a
hook, manifest.json's `"hooks"` array falls out of sync.

The manifest is analogous to `system_prompt.txt`: a single leaf file that
describes the harness configuration. The auto-sync approach used in #202
is brittle[^1] because it inspects the filesystem and assumes the Evolver
meant to register every file it wrote. Letting the Evolver edit the
manifest directly is more honest and more robust.

[^1]: See issue #279 -> Evidence links in the trace store.

## Decision

Add `harness/manifest.json` to the set of leaf files the Evolver may
propose edits to, alongside `harness/system_prompt.txt`. The
`_confine_to_harness_tree` validator treats it identically to the prompt
file:

- `harness/manifest.json` → accepted.
- `harness/manifest.json/subpath` → rejected (leaf files are not
  directories).

The existing error message is updated to list all four allowed targets
instead of the previous three.

The ADR-0004 guardrail text is not changed; the Critic continues to run
the full test suite and benchmark suite on every proposal, including
manifest edits.

## Consequences

- The Evolver can now propose edits that keep `manifest.json` in sync
  with the skills and hooks trees in a single proposal.
- Manifest edits are subject to every existing guardrail: rate limits,
  diff size caps, the Critic gate, and the human-review requirement for
  hand-edits.
- Because manifest.json is a JSON file with structural constraints,
  the Critic should pay *extra* attention to manifest edits — a
  malformed JSON manifest could break harness loading. The Critic's
  evaluation prompt will be updated to flag JSON syntax errors in
  manifest edits (see ADR-0009 for Critic prompt structure).
- The next human who reviews an Evolver PR touching manifest.json must
  verify that the manifest remains valid JSON and matches the on-disk
  state of `hooks/` and `skills/`.
