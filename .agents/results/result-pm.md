# PM Result — Parallel GitHub-Issue Discovery Prompt

- **Status:** delivered
- **Session:** standalone (not orchestrated)
- **Summary:** Authored a self-contained, ready-to-run **master prompt** that
  launches read-only sub-agents **in parallel**, validates their findings in a
  second parallel wave, and returns a ranked, deduplicated, KPI-aligned
  portfolio of copy-paste-ready GitHub issue proposals for `anchapin/foundry-x`.
  The prompt is tuned to the repository's *current* state — **all three roadmap
  phases shipped (commit `929b327`) and zero open issues** — so discovery
  pivots from "build planned features" to "advance KPIs, harden, fix
  docs/code drift, and gather evidence."
- **Files changed:**
  - `.agents/prompts/parallel-issue-discovery.md` — **the deliverable prompt**
    (new). Hand this file to an orchestrator agent to fan out sub-agents.
- **Relationship to existing assets:** The repo already ships a canonical,
  exhaustive 626-line `docs/ISSUE_GENERATION_PROMPT.md`. This new prompt is a
  **sharper, post-roadmap-tuned, execution-first** companion: it embeds the
  current verified facts, an explicit parallel dispatch spec, the exact label
  set, the issue-body template (mirroring issue #900), and the guardrails — so
  an orchestrator can act immediately without re-deriving context. It does not
  delete or contradict the canonical doc.
- **Acceptance-criteria checklist:**
  - [x] Prompt produces a **parallel** sub-agent fan-out (scout wave + validator wave).
  - [x] Each sub-agent's scope is **bounded** to one `area-*` slot with ADR mapping.
  - [x] Every proposal is forced to map to **exactly one PRD KPI** and use only
        existing labels; governance gaps routed to "Human Decisions," not invented.
  - [x] Guardrails encoded: no harness hand-edits, no Critic bypass, no invented
        vocabulary/labels/phases, evidence-over-opinion, "zero is valid."
  - [x] Tuned to current state (shipped phases + 0 open issues): explicit list of
        *valid* post-roadmap issue sources so scouts don't fabricate feature demand.
  - [x] Returns a deterministic YAML schema + Conventional-Commits titles + the
        issue-body template that matches the repo's closed issues.
  - [x] Includes hard rejection gates, a 0–100 scoring rubric, and a final
        re-dedup synthesis step to catch races.

---

## The prompt (mirrored here for review)

The canonical copy lives at **`.agents/prompts/parallel-issue-discovery.md`**.
Paste its contents into an orchestrator agent (e.g. `squad-orchestrator` /
`general`) and set the runtime inputs at the top. See that file for the full
text.
