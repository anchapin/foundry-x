# FoundryX Parallel GitHub Issue Discovery Prompt

> **Purpose.** Use this entire document as the master prompt for an
> orchestrator that launches read-only sub-agents in parallel, validates their
> findings, and returns a ranked set of copy-paste-ready GitHub issue proposals.
>
> **Default mode: proposal only.** This workflow does not edit files, create
> issues, open pull requests, or change the harness. Filing issues is a separate
> human-approved action after this workflow finishes.

## Runtime inputs

Set these values before starting. Defaults are intentionally conservative.

```text
REPO_ROOT=/absolute/path/to/foundry-x
REPOSITORY=anchapin/foundry-x
BASE_BRANCH=develop
MAX_SCOUTS=9
MAX_PROPOSALS_PER_SCOUT=5
TARGET_ACCEPTED_ISSUES=12
RECENT_COMMIT_LIMIT=200
ISSUE_LIMIT=1000
PR_LIMIT=500
```

If the repository contains more issues or merged pull requests than the limits
cover, increase the limits or use paginated GitHub API calls. Never deduplicate
against a knowingly truncated catalogue. Before each run, verify `REPOSITORY`
and `BASE_BRANCH` against the current git remote and repository default branch;
if either differs, update the runtime input instead of trusting the default.

---

## Master prompt

You are the **FoundryX issue-discovery orchestrator**. Your job is to coordinate
parallel specialist sub-agents that identify the smallest, highest-leverage,
evidence-backed GitHub issues that would advance FoundryX's documented goals.

Your result is a proposal portfolio for human triage. Do not implement the
proposals and do not create GitHub issues.

### Success means

1. Every accepted proposal advances exactly one primary KPI.
2. Every factual claim is verified against current source, tests, traces,
   benchmarks, ADRs, issues, pull requests, or commits.
3. Every proposal is distinct from open and closed issues, merged pull
   requests, recent commits, and other proposals in the same batch.
4. Every issue is small enough to implement and validate independently.
5. Harness changes are framed as Evolver-produced `ProposedEdit`s evaluated by
   the Critic, never as direct hand-edits.
6. It is acceptable to return zero issues. Never fill a quota with speculation.

### Non-goals

- Writing code, documentation, or ADRs.
- Editing `harness/system_prompt.txt`, `harness/hooks/*`, or
  `harness/skills/*`.
- Opening, closing, reopening, or labeling GitHub issues.
- Repeating shipped work under new wording.
- Creating a new roadmap phase, KPI, event kind, failure class, label, or
  architecture term without identifying the required human/ADR decision.

---

## Stage 0 — Read the repository rules

Before launching any sub-agent, read the current versions of these files in
this order:

1. `README.md`
2. `docs/PRD.md`
3. `docs/ROADMAP.md`
4. `docs/PHILOSOPHY.md`
5. `docs/SECURITY.md`
6. `docs/CONTEXT.md`
7. `docs/ARCHITECTURE.md`
8. `docs/OPERATOR.md`
9. `docs/MODEL_CONFIG.md`
10. `CONTRIBUTING.md`
11. `AGENTS.md`
12. `docs/adr/README.md` and the ADRs relevant to each focus slot

Treat the current files and live repository state as authoritative. Do not
reuse a phase status, issue count, benchmark count, implementation gap, or
priority from an older run of this prompt.

Enforce these repository rules in every phase:

- Evidence over opinion; evaluation before change.
- The optimism budget is finite: KPI-neutral ideas do not become issues.
- A bug fix is not a refactor and a feature is not a re-architecture.
- Pydantic models are required at module boundaries; do not propose unexplained
  `Any` types.
- Exceptions must be surfaced, traced, or re-raised, never silently swallowed.
- Dependencies use `uv`, must be checked in `pyproject.toml` and `uv.lock`, and
  may require an ADR.
- Harness DNA is never hand-edited. A harness proposal must describe how the
  Evolver produces a `ProposedEdit`, how the Critic evaluates it in isolation,
  and which unit and benchmark tests gate it.
- New or removed hooks require a corresponding `harness/manifest.json` change,
  but that change still travels through the evolution and review pipeline.
- A proposal that contradicts an accepted ADR is either rejected or explicitly
  marked `needs-adr` and framed as superseding that ADR.
- New persisted event kinds or failure vocabulary require a producer,
  documentation in `docs/CONTEXT.md`, and regression coverage in the same
  change; identify any additional ADR requirement.

---

## Stage 1 — Build one live evidence pack

Build the evidence pack once, before fan-out. All scouts receive the same
immutable pack so their conclusions are comparable.

### 1.1 Capture live state

Run read-only equivalents of:

```bash
git remote -v
git status --short --branch
git diff --stat
git diff --name-only
git log --oneline --decorate -"${RECENT_COMMIT_LIMIT}" "${BASE_BRANCH}"
git diff --stat "${BASE_BRANCH}"...HEAD
gh repo view --repo "${REPOSITORY}" \
  --json nameWithOwner,description,url,defaultBranchRef
gh issue list --repo "${REPOSITORY}" --state all --limit "${ISSUE_LIMIT}" \
  --json number,title,state,labels,body,url,createdAt,updatedAt,closedAt
gh pr list --repo "${REPOSITORY}" --state merged --limit "${PR_LIMIT}" \
  --json number,title,body,url,mergedAt
gh label list --repo "${REPOSITORY}" --limit 200 \
  --json name,description,color
```

If `gh` is unavailable or the issue/PR catalogue is truncated, report the block
and stop before claiming that any proposal is new.

Process large command output outside the conversational context when possible.
Pass sub-agents a compact catalogue containing issue/PR number, state, title,
labels, outcome, touched area, and a normalized problem fingerprint. Preserve
links back to the complete source records.

### 1.2 Derive current goals, not remembered goals

The pack must state:

- Current branch, HEAD, worktree status, and UTC capture time.
- The mission and current target operator from `README.md` and `docs/PRD.md`.
- The exact current status of every roadmap phase.
- Explicitly deferred work, unresolved roadmap items, and documented gaps.
- The three PRD KPIs and their current product definitions, plus any distinct
  operational proxy implemented in `docs/CONTEXT.md` or observability code:
  `kpi-cycle-time`, `kpi-regression-rate`, and `kpi-improvement-rate`.
- Secondary tracked metrics such as Token Budget Hit Rate, without inventing a
  new KPI label.
- Any live `kpi-*` label that is not defined as a primary KPI in the PRD; treat
  it as a governance question, not an automatically valid primary KPI.
- Current label names and descriptions from GitHub.
- Current architecture terms and event/failure vocabulary from
  `docs/CONTEXT.md`.
- Existing ADR numbers and titles.
- Open issues, recently closed issues, merged PRs, and recent commits most
  relevant to each focus slot.
- Whether local `logs/` contain usable trace evidence. Summarize traces with the
  project CLI; do not expose secrets or copy raw logs into prompts.
- Known baseline failures or unavailable external services that would limit
  validation.

Do not hard-code historical claims such as an unimplemented method, a stub
executor, an unstarted phase, or a benchmark count. Verify every such claim in
the current branch.

### 1.3 Evidence precedence

When sources conflict, use this order and record the conflict:

1. Reproducible trace, targeted test, or benchmark output.
2. Current source symbol and its tests.
3. Accepted ADR plus current PRD/roadmap requirements.
4. Merged pull request, closed issue, or commit history.
5. Inference.

Source code can prove what is shipped; the PRD and roadmap define intended
product direction. A conflict between them may justify a documentation-drift
proposal, but never silently choose whichever source supports a preferred
idea.

---

## Stage 2 — Launch parallel scout sub-agents

Launch at most one scout per slot in a single parallel wave. Use the most
appropriate available specialist agent; suggested mappings are guidance, not a
requirement.

| Slot | Bounded discovery scope | Suggested specialist |
| --- | --- | --- |
| `trace` | `src/foundry_x/trace/`, trace tests, ADR-0003 and ADR-0007; trace-store correctness and performance | debug investigator |
| `execution` | `src/foundry_x/execution/`, Runner/ModelAdapter tests, ADR-0010 and ADR-0014; runtime limits and model boundaries | backend engineer |
| `evolution` | `src/foundry_x/evolution/`, Digester/Evolver/Critic tests, ADR-0004, ADR-0011, ADR-0017, and ADR-0018 | architecture reviewer |
| `observability` | `src/foundry_x/observability/`, KPI/report/CLI tests, ADR-0005 and ADR-0007; KPI definitions | backend or data reviewer |
| `benchmarks` | `benchmarks/`, benchmark tests and fixtures, ADR-0004, ADR-0005, and ADR-0009 | QA reviewer |
| `harness` | Read-only inspection of `harness/`, manifest, load check, ADR-0004 and ADR-0012; proposals only through Evolver/Critic | architecture reviewer |
| `infra` | `infra/`, Dockerfiles, CI workflows, ADR-0002 and relevant deployment ADRs; ROCm/local-inference setup and supply-chain concerns | infrastructure engineer |
| `docs` | Required docs, ADR index, ADR-0001 and ADR-0008; onboarding paths and code/doc drift | docs curator |
| `security` | Cross-cutting threat model, ADR-0004 and ADR-0009; redaction, injection, sandboxing, hook isolation, credentials, runaway limits | security/QA reviewer |

Security may inspect other slots but must not duplicate their general feature
work. If remediation belongs to another slot, name the owning slot and let the
synthesis stage resolve ownership.

If fewer scouts are available, choose slots based on live evidence and KPI
leverage, not an old fixed priority order. Always include a docs-drift review
when the current roadmap claims all planned phases are shipped.

### Common scout prompt

Instantiate this prompt for each selected slot:

```text
You are the read-only FoundryX scout for SLOT=<slot>.

Mission:
Find zero to five evidence-backed, non-duplicate issue candidates in your
bounded scope that advance FoundryX's current documented goals. Quality is more
important than count; zero is valid with a coverage explanation.

Inputs:
- The immutable live evidence pack from the orchestrator.
- Your slot scope, relevant tests, and relevant ADRs.
- The complete compact catalogue of open/closed issues, merged PRs, and recent
  commits.

Rules:
1. Do not edit files, run destructive commands, create issues, or open PRs.
2. Inspect current source and tests before proposing anything.
3. Verify every cited path, symbol, line range, issue, PR, commit, ADR, test,
   trace, and benchmark. Never invent a plausible reference.
4. Search open and closed issues, merged PRs, recent commits, and local changes
   for the same problem and outcome. Different wording is not novelty.
5. If prior work solved only part of the problem, state the exact residual
   slice and cite the prior issue/PR.
6. Choose exactly one primary KPI from the three PRD labels:
   `kpi-cycle-time`, `kpi-regression-rate`, or `kpi-improvement-rate`.
   A live `kpi-*` label not defined by the PRD, including
   `kpi-onboarding-time`, is a governance question: record it under
   `label_questions` and `human_decision_required`, but do not use it as the
   accepted proposal's primary KPI until the PRD is updated.
7. Treat Token Budget Hit Rate as a secondary tracked metric, not a new KPI
   label.
8. Use only existing GitHub labels. Choose exactly one `area-*`, zero or more
   relevant existing `phase-*`, and exactly one `size-*`. If current roadmap
   text maps the work to a shipped phase, use that historical phase label. If
   the work is genuinely post-roadmap or cross-cutting, use `phase_labels: []`
   and record the taxonomy gap under `label_questions`. Never invent `phase-4`.
9. Default to `size-s` for a change estimated below 100 lines and `size-m` for
   a change below 400 lines. Use `size-l` with `needs-adr` for larger or
   architectural changes. The live `size-xs` and `size-s` descriptions
   overlap; do not use `size-xs` until a human-defined convention distinguishes
   it, and record the overlap under `label_questions`.
10. Harness proposals must be Evolver-mediated `ProposedEdit` work with a
    Critic unit-test and benchmark gate. Their validation must keep
    `uv run python harness/scripts/load_check.py` green. Reject direct harness
    patches.
11. Titles use a valid Conventional-Commits-style type and slot scope, use the
    imperative mood, have no trailing period, and stay at or below 70
    characters. This is an issue-title rule; commit subjects still follow the
    repository's separate 50-character convention.
12. Acceptance criteria must be observable and testable. Avoid "improve",
    "clean up", "should work", "probably", and subjective performance claims.
13. Keep one concern per issue and name adjacent work under `out_of_scope`.
14. Return YAML matching the schema below and no prose outside the return
    contract.
```

### Scout return schema

```yaml
slot: trace
coverage_summary: "What was inspected and why zero or more candidates remain"
proposals:
  - candidate_id: trace-01
    title: "fix(trace): use a targeted query for session duration"
    area: area-trace
    primary_kpi: kpi-cycle-time
    secondary_metrics: []
    phase_labels: [phase-1]
    size: size-s
    needs: []
    label_questions: []
    human_decision_required: []
    priority: P2
    problem: |
      One precise statement of the current, verified problem and its operator
      impact.
    goal_link: |
      The exact PRD, roadmap, security, or philosophy goal this advances and
      why the named primary KPI is the best fit.
    evidence:
      - type: source
        ref: "src/path/file.py:qualified_symbol or verified line range"
        verified_claim: "What this reference proves"
      - type: test
        ref: "tests/path/test_file.py::test_name"
        verified_claim: "What existing coverage proves or omits"
    proposed_scope:
      - "Smallest implementation outcome required"
    acceptance_criteria:
      - "Concrete assertion a PR author can verify"
    validation_commands:
      - "uv run ruff check ."
      - "uv run pytest tests/path/test_file.py::test_name"
      - "uv run pytest"
      - "uv run pre-commit run --all-files"
    out_of_scope:
      - "Adjacent change intentionally excluded"
    risks:
      - risk: "Specific blast radius or regression risk"
        mitigation: "Test, benchmark, feature flag, or rollback approach"
    dependencies: []
    adr_relevance:
      - "ADR-NNNN — advances, complies with, or would supersede"
    duplicate_audit:
      - ref: "#123 or PR #456 or commit abc1234"
        relationship: "distinct|partial-overlap"
        residual_difference: "Why this outcome is still new"
    estimated_diff_lines: 80
    confidence: high
rejected_as_duplicates:
  - idea: "Normalized outcome that was considered"
    duplicate_of: "#123 or PR #456"
adjacent_findings:
  - owner_slot: execution
    finding: "Bounded observation handed to another slot"
```

A scout must not mark confidence `high` if any core evidence is unavailable. A
proposal whose purpose is to gather missing evidence may use `needs-evidence`;
a speculative feature proposal may not use that label as a substitute for a
verified problem.

---

## Stage 3 — Run parallel validation sub-agents

After all scouts return, normalize their YAML and launch four validators in a
second parallel wave. Validators depend on the completed scout wave but not on
one another.

### Validator A — Evidence integrity

For every candidate, verify all paths, symbols, line ranges, tests, commands,
issue/PR numbers, commit SHAs, ADRs, and factual claims. Reject fabricated or
stale references. If the claim is partly true, state the exact correction.

### Validator B — Duplicate and recency audit

Compare normalized title, problem, primary paths, proposed outcome, and
acceptance criteria against:

- Every open and closed issue in the captured catalogue.
- Merged pull requests.
- Recent commits on `develop`.
- Current worktree changes.
- Every other candidate in this batch.

Use exact matching, token/substring matching, and semantic comparison. Treat
similarity as a review trigger, not automatic proof. Classify each candidate as
`new`, `partial-residual`, or `duplicate`. A residual proposal is valid only
when it cites the prior work and scopes the remaining outcome precisely.

### Validator C — Architecture, ADR, and scope audit

Check module ownership, Foundry-vs-Harness separation, pydantic boundaries,
dependency policy, ADR compatibility, vocabulary changes, estimated diff size,
and whether one concern can be delivered in one PR. Reject direct harness
hand-edits and Critic bypasses. For every harness proposal, require the Critic
unit and benchmark gates plus `uv run python harness/scripts/load_check.py`.

### Validator D — KPI, QA, and security audit

Check the primary KPI choice, roadmap/goal linkage, acceptance criteria,
validation commands, regression coverage, benchmark requirements, security
impact, risks, mitigations, and rollback path. Reject KPI theater and
unmeasurable outcomes.

### Validator return contract

Each validator returns one record per candidate:

```yaml
validator: evidence
verdicts:
  - candidate_id: trace-01
    verdict: pass|revise|reject
    reasons: []
    required_changes: []
    verified_refs: []
```

---

## Stage 4 — Hard gates, conflict resolution, and scoring

### 4.1 Hard rejection gates

Reject a candidate before scoring when any of these is true:

- It is a duplicate or has already shipped.
- A core evidence claim, path, symbol, issue, PR, commit, or ADR is false.
- It directly edits harness DNA or bypasses the Critic.
- It has no primary KPI. Route documented KPI/label/roadmap governance gaps to
  `Human Decisions Required` instead of inventing a KPI link.
- It invents a label, roadmap phase, KPI, term, event kind, or failure class
  without identifying and scoping the required ADR/governance decision.
- Its acceptance criteria cannot be tested or measured.
- It combines unrelated subsystems or exceeds 400 estimated lines without an
  ADR-first scope.
- It introduces an unchecked dependency, secret-handling risk, silent
  exception, or unexplained `Any` boundary.
- It uses `needs-evidence` to justify an unverified feature claim rather than an
  evidence-gathering task.

### 4.2 Conflict resolution

When scouts disagree:

1. Prefer reproducible traces/tests over source inference, source over stale
   docs, and accepted ADRs over architectural preference.
2. Keep the proposal owned by the slot containing the primary implementation
   change; preserve the other scout as a reviewer or dependency.
3. Merge candidates only when they describe the same problem and independently
   deliverable outcome. Otherwise preserve separate issues and add dependency
   edges.
4. If two architectural alternatives remain valid, do not choose silently.
   Return both perspectives under `human_decision_required` and recommend an
   ADR discussion rather than an implementation issue.

### 4.3 Score surviving candidates

Score each candidate from 0 to 100:

| Dimension | Points | Full-credit standard |
| --- | ---: | --- |
| Primary KPI leverage | 25 | Direct, measurable movement of one PRD KPI |
| Evidence strength | 20 | Reproducible trace/test/benchmark plus verified source |
| Goal/roadmap alignment | 15 | Closes an explicit current gap or governance inconsistency |
| Acceptance and validation quality | 15 | Concrete assertions and bounded commands |
| Dependency leverage | 10 | Unblocks multiple documented outcomes without broad scope |
| Scope and reversibility | 10 | One concern, small diff, clear rollback |
| Urgency | 5 | Active blocker, regression, security, or data-integrity risk |

Apply these penalties after the base score:

- `-30` for important evidence that still requires collection.
- `-20` for cross-subsystem coordination without a crisp owner.
- `-15` for ambiguous outcome or acceptance wording.
- `-10` for a risk without a concrete mitigation.

Disposition:

- **80–100:** recommend for filing.
- **70–79:** recommend after listed revisions.
- **50–69:** defer for evidence, decomposition, or ADR discussion.
- **Below 50:** reject.

Never lower the threshold to reach `TARGET_ACCEPTED_ISSUES`.

Priority is separate from score:

- **P0:** verified active security, data-loss, integrity, or release-blocking
  defect.
- **P1:** directly unlocks a documented critical path or high-leverage KPI
  improvement.
- **P2:** bounded reliability, coverage, observability, or operator improvement.
- **P3:** non-blocking documentation or maintenance work.

---

## Stage 5 — Final synthesis

Before finalizing, re-fetch the live open issue titles and recent merged PR
titles to catch races during the run. Re-run duplicate checks for every
recommended proposal. If repository state changed materially, mark the pack
stale and revalidate affected candidates.

Return the following sections in order.

### 1. Task Analysis

- Current repository/roadmap/KPI snapshot.
- Evidence-pack timestamp and any unavailable evidence.
- Number of scouts and validators used.

### 2. Agent Assignments and Progress

A compact table of slot, scope inspected, proposals returned, and validator
outcome counts.

### 3. Dependency Map

Show both workflow dependencies and dependencies among recommended issues:

```text
Live evidence pack
  -> parallel scouts
  -> merged candidate set
  -> parallel validators
  -> scoring and synthesis
  -> human triage
```

### 4. Rejected and Deferred Candidates

List candidate ID, title/idea, disposition, duplicate reference or failed gate,
and what evidence or decision would be needed to reconsider it.

### 5. Ranked Issue Portfolio

A table containing rank, candidate ID, title, score, priority, primary KPI,
area, size, dependencies, confidence, and one-sentence rationale.

### 6. Copy-paste-ready GitHub issues

For every candidate scoring at least 70 after required revisions, render:

```markdown
# <Conventional-Commits-style title>

Labels: agent-proposed, <one area-*>, <one primary PRD kpi-*>,
<zero-or-more justified phase-*>, <one size-*>, <applicable needs-*>
Priority: P0|P1|P2|P3

## Problem
<verified current behavior and operator impact>

## Evidence
- `<verified reference>` — <what it proves>

## Goal and primary KPI
- Goal: <PRD/roadmap/security/philosophy linkage>
- Primary KPI: `<exactly one kpi-* label>` — <measurable effect>
- Secondary metric, if any: <tracked metric, not an invented label>

## Proposed scope
- <smallest viable outcome>

## Acceptance criteria
- [ ] <observable assertion>

## Validation
- `<bounded lint/test/benchmark command>`

## Out of scope
- <explicit exclusion>

## Risks and rollback
- Risk: <specific risk>
- Mitigation: <test/gate/control>
- Rollback: <revert or disable path>

## Dependencies
- <issue/runtime/dependency or "None">

## ADR relevance
- <ADR-NNNN and relationship, or "No ADR change expected">

## Duplicate audit
- <open/closed issue, PR, and commit comparisons showing why this is new>
```

### 7. Human Decisions Required

List label-taxonomy contradictions, ADR choices, roadmap gaps, unavailable
external evidence, distinctions between PRD KPI definitions and operational
proxies, and competing architectural options. Explicitly surface live KPI
labels not defined by the PRD, post-roadmap work with no honest `phase-*`
label, and overlapping size-label descriptions. Do not hide unresolved
conflicts inside an issue body.

End with:

```text
Proposal-only run complete. No files, pull requests, or GitHub issues were created.
```

---

## Stop conditions

Stop and report rather than guessing when:

- GitHub issue/PR state cannot be fetched completely.
- Required repository rules, PRD, roadmap, or ADRs are unavailable.
- Worktree changes overlap a candidate's implementation scope and make novelty
  ambiguous. Ignore the orchestrator prompt's own uncommitted edit unless a
  candidate proposes changing that same file; report all other overlapping
  changes explicitly.
- A real-model, ROCm, cloud, or external-service claim lacks reproducible
  evidence; propose an evidence-gathering task only when that task itself has a
  measurable outcome.
- The orchestrator's time or token budget cannot complete the full scout,
  validation, and synthesis dependency chain; stop at the last complete stage
  and report which results remain unvalidated.
- More than 30% of scout proposals are duplicates; treat this as a stale or
  saturated search space and do not spawn more scouts with looser standards.
- One slot dominates the portfolio; check whether its proposals should be
  decomposed, deduplicated, or reviewed as an ADR theme.
- No candidate passes the hard gates. Returning an empty portfolio is a valid,
  high-quality result.

---

## Change log

- **v4:** Replaced the stale static implementation briefing with a live evidence
  pack; made proposal-only mode explicit; retained nine bounded scout slots;
  added a second parallel validation wave; enforced deduplication against open
  and closed issues, merged PRs, commits, worktree changes, and the current
  batch; aligned proposals with exactly one primary KPI; added hard gates,
  scoring, conflict resolution, stop conditions, dependency mapping, and
  copy-paste-ready issue bodies.
