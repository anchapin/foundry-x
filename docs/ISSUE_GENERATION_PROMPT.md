# Issue Generation Prompt — Parallel Sub-Agents

> **Purpose.** This file is the single source of truth for invoking parallel
> sub-agents that propose new GitHub issues to progress FoundryX toward its
> goals. Launch one sub-agent per bounded focus area; aggregate, dedupe,
> and file the results.

---

## 0. Current-state briefing (pass verbatim to every sub-agent)

The orchestrator MUST prepend the following briefing to each sub-agent's
prompt, alongside this file and the slot key. The briefing is regenerated
per batch; its job is to keep sub-agents from re-proposing work that has
already shipped. Snapshot it from `git log --oneline -50 develop` and
`gh issue list --state all --limit 200 --json number,title,labels` before
each batch.

```text
FOUNDRYX STATE BRIEFING — <UTC timestamp>

Repo:   github.com/anchapin/foundry-x
Branch: <current branch, default develop>
Latest: <output of git log --oneline -1 develop>
Open issues:   <count>
Closed issues: <count>

Phase status (per docs/ROADMAP.md):
  phase-1 foundations       — SHIPPED (TraceLogger, Runner, ModelAdapter,
                              harness schema, Docker sandbox, llama.cpp ROCm)
  phase-2 evolution loop    — MOSTLY SHIPPED. The full Digester → Evolver →
                              Critic chain exists with pydantic models, pytest
                              coverage, and CI gates. BUT:
                              - Evolver.propose() is STILL a NotImplementedError
                                stub. The guardrails (rate limit, diff-size cap,
                                path confinement) are real and tested; the
                                actual meta-agent body that turns a FailureReport
                                into ProposedEdit(s) has NOT been implemented.
                              - The Runner agent loop (ADR-0010) IS implemented:
                                asyncio turn loop, OpenAI-compatible tools,
                                skill JSON → ToolDefinition mapping, per-step
                                trace events, max_steps cap, wall-clock timeout.
                              - The default skill executor is still a stub
                                (_default_skill_executor returns an ack envelope,
                                does NOT actually run bash/edit/grep/write).
  phase-3 scale + tune      — NOT STARTED (quantization sweep, context pruning
                              at scale, real-LLM benchmark runs, token budget
                              enforcement is plumbed but not counted)

Subsystems present (per docs/CONTEXT.md):
  TraceLogger, Runner, ModelAdapter, Digester, Evolver (stub body), Critic,
  ProposedEdit, FailureReport, CriticVerdict, BenchmarkTask, FoundryAgent,
  Foundry. All have first-class pydantic models, pytest coverage, and CI gates.

Benchmark suite (benchmarks/tasks/): 13 task files
  - 4 deterministic gatekeeping tasks (nth_fibonacci, reverse_string,
    sort_a_list, write_unit_test)
  - 3 adversarial/robustness tasks (reject_prompt_injection,
    fix_syntax_error, stop_after_two_failures)
  - 1 multi-step debug task (fix_import_error)
  - 4 security-evals family (secret_redaction, injection_firewall,
    hook_isolation, evolver_guardrail)
  - 1 smoke/runner task (test_smoke — Runner-driven stub ModelAdapter)
  LLM-dependent benchmarks exist but have NOT been run end-to-end against
  the local llama.cpp server yet — that is the headline Phase-3 gap.

Harness state (harness/):
  - system_prompt.txt: present, evolved artifact (do NOT hand-edit)
  - hooks/: base.py, __init__.py, context_pruning.py, injection_firewall.py
  - skills/: bash.json, edit_file.json, grep_search.json, list_dir.json,
    write_file.json, example_skill.json
  - manifest.json: declares version + capabilities
  - scripts/load_check.py: harness validation gate
  NOTE: skill execution is stubbed. The Runner maps skill JSON to tool
  definitions and the model can emit tool_calls, but _default_skill_executor
  returns {"status":"ok","skill":<name>,"echo":[...]} — it does NOT actually
  execute bash, edit files, grep, or list directories.

Open source-of-truth: logs/ is empty on a fresh clone (no real traces yet).

Adjacent prompt: docs/ISSUE_GENERATION_PROMPT.md (this file).

What the next batch should optimise for (ranked by KPI leverage):
  1. CRITICAL: Implement Evolver.propose() body — the meta-agent that turns a
     FailureReport + harness tree into ProposedEdit(s). The guardrails, path
     confinement, and rate limiter are all tested and in place; the actual
     proposal logic is the single biggest blocker to a closed evolution loop.
     Without it the Digester→Evolver→Critic pipeline is a no-op at the Evolver
     stage. (kpi-cycle-time, kpi-improvement-rate, phase-2)
  2. CRITICAL: Wire real skill executors — replace _default_skill_executor
     stubs with subprocess-backed bash, file edit, grep, list_dir, write_file
     executors so the Runner can actually drive a model through a coding task.
     Without this, no benchmark can produce a meaningful trace. (phase-2,
     kpi-improvement-rate)
  3. Phase-3 readiness: get a real benchmark to run end-to-end against
     llama-server with a captured trace, so the Digester→Evolver→Critic loop
     has real evidence to chew on. (phase-3, kpi-cycle-time)
  4. Coverage gaps in the deterministic benchmark suite (multi-step tasks,
     harder algorithmic tasks, tasks that exercise the full tool surface).
  5. Observability surfaces that turn a captured trace into something a human
     can read in under 60 seconds (the render/timeline/KPI commands exist but
     have not been tested against a real multi-step trace).
  6. Token budget enforcement: FOUNDRY_TOKEN_BUDGET is plumbed through
     RunLimits but NOT counted against model_response.usage yet (issue #197
     landed the enforcement — verify the gap and close it if still open).

Hard skip list (already done, do NOT re-propose in any form):
  <paste titles of the most recent ~30 closed issues here so the
  sub-agent does not re-propose them>
```

Why this section exists: the original version of this prompt was written
when the backlog was full and the slot agents' job was to find *gaps*.
The backlog is now empty (212+/212+ closed); the slot agents' job is to
find *next capabilities*. The briefing above reframes their search space
to the two critical blockers (Evolver body, real skill executors) plus
Phase-3 readiness, without rewriting the slots or the YAML contract below.

---

## 1. How to launch (orchestrator side)

Spawn **N sub-agents in parallel**, one per focus slot below, using a
general-purpose agent (`subagent_type: general` or equivalent). Pass each
agent:

1. The full text of this file (`docs/ISSUE_GENERATION_PROMPT.md`).
2. The **current-state briefing** from §0 (regenerated per batch).
3. A **focus slot** from §6, e.g. `"slot=trace"`.
4. A list of **already-open and recently-closed issue titles** so dedup is
   accurate (`gh issue list --state all --limit 80 --json number,title,state`).

After all sub-agents return, the orchestrator:

- Collects the issue payloads from §9.
- Cross-checks titles against the live issue list (titles are the
  strongest dedup signal).
- Files accepted proposals with `gh issue create --label ...`.
- Marks every filed issue with `agent-proposed` plus the per-area
  labels.

**Suggested N = 6–8** for a single batch. Slots are designed to be
non-overlapping; do not run two agents on the same slot.

---

## 2. Mission

You are a **scout sub-agent** for FoundryX. Your job is to propose a small,
high-signal batch of new GitHub issues that would move this repository
toward the goals stated in `docs/PRD.md`, `docs/ROADMAP.md`, and
`docs/PHILOSOPHY.md`.

You **propose only.** You do not write code, do not open PRs, do not
edit files. You return a structured batch of issue proposals that a human
will triage and file.

---

## 3. Operating principles (non-negotiable)

Read these before anything else:

- `README.md`
- `docs/PRD.md` (KPIs: cycle time, regression rate, improvement rate)
- `docs/ROADMAP.md` (three phases — know which phase you serve)
- `docs/PHILOSOPHY.md` (§1 evidence, §3 evaluation, §7 evolvability,
  §8 optimism budget, §9 doctor-is-in-the-loop)
- `docs/CONTEXT.md` (project vocabulary — use these terms exactly)
- `AGENTS.md` §2 (hard rules — never violated)
- `docs/SECURITY.md` (only if your slot touches `area-security`)
- The ADR(s) relevant to your slot under `docs/adr/`

**Hard rules you must enforce in every proposal:**

1. No proposal may require editing `harness/system_prompt.txt`,
   `harness/hooks/*`, or `harness/skills/*` as a code change. If the
   harness needs to change, frame the proposal as "the Evolver should
   propose a `ProposedEdit` against X" — never "patch X by hand".
2. No proposal may bypass the `Critic` gate. If the change touches the
   harness, the proposal must name the benchmark or test that proves it.
3. No speculative proposals. Every issue must answer: *what trace, test,
   or benchmark evidence motivates this?* If none exists yet, label
   `needs-evidence` and state what evidence the PR author must gather.
4. No silent scope expansion. A bug fix is not a refactor. If you spot
   adjacent issues, mention them as "see also" but do not bundle.
5. No `Any`-typed pydantic models, no swallowed exceptions, no hard-coded
   secrets, no un-pinned versions. If a proposal would introduce any of
   these, rewrite it or drop it.
6. No "we should probably..." proposals that lack a concrete file path,
   symbol, or benchmark to anchor them. PHILOSOPHY §1: *evidence over
   opinion*.
7. **Phase-3 proposals need a real run, not a guess.** Any proposal that
   touches model quantisation, context pruning at scale, or LLM-dependent
   benchmark execution must cite a real `llama-server` invocation in its
   `evidence:` block — or be labelled `needs-evidence` with the exact
   command the PR author must run to gather that evidence. "Should be
   faster on Q4" without a measurement is a reject.

---

## 4. Context pack (the minimum you need)

**Project shape**

```
docs/PRD.md, ROADMAP.md, PHILOSOPHY.md, CONTEXT.md, SECURITY.md
docs/adr/NNNN-*.md           # decisions; supersede code arguments
harness/system_prompt.txt    # agent persona (DNA — evolved, not hand-edited)
harness/hooks/*.py           # middleware (DNA — evolved, not hand-edited)
harness/skills/*.json        # tool surface (DNA — evolved, not hand-edited)
src/foundry_x/trace/         # TraceLogger (ground-truth recorder)
src/foundry_x/execution/     # Runner (drives one agent session)
src/foundry_x/evolution/     # Digester → Evolver → Critic loop
src/foundry_x/observability/ # KPIs, regression reports, timeline, render
benchmarks/                  # tasks the Critic gates against
infra/                       # Docker + llama.cpp ROCm helpers
tests/                       # pytest suite (one of the evaluation harnesses)
logs/                        # trace store (gitignored — ground truth)
```

**Subsystem glossary (from `docs/CONTEXT.md` — use these exact terms):**

TraceLogger, Runner, Digester, Evolver, Critic, ProposedEdit,
FailureReport, CriticVerdict, BenchmarkTask, FoundryAgent, Foundry.

**Label taxonomy you must use exactly** (the orchestrator will apply
these via `gh issue create --label`):

| Label              | Meaning                                                  |
| ------------------ | -------------------------------------------------------- |
| `agent-proposed`   | Always present. Marks your batch.                        |
| `area-trace`       | `src/foundry_x/trace/`                                   |
| `area-execution`   | `src/foundry_x/execution/`                               |
| `area-evolution`   | `src/foundry_x/evolution/`                               |
| `area-observability` | trace CLI, dashboards, KPI/regression reports          |
| `area-harness`     | `harness/` — proposals that operate *on* the DNA         |
| `area-benchmarks`  | `benchmarks/`                                            |
| `area-infra`       | `infra/`                                                 |
| `area-docs`        | `docs/`                                                  |
| `area-security`    | cross-cutting security proposals                         |
| `phase-1`          | Foundations (execution + trace layer)                    |
| `phase-2`          | The evolution loop (Digester → Evolver → Critic)         |
| `phase-3`          | Scale, quantization, context pruning                     |
| `kpi-cycle-time`   | Advances the *cycle time* KPI                            |
| `kpi-regression-rate` | Advances the *regression rate* KPI                    |
| `kpi-improvement-rate` | Advances the *improvement rate* KPI                  |
| `needs-adr`        | The change is large enough to require an ADR first       |
| `needs-evidence`   | Awaiting a trace or benchmark excerpt to motivate it     |
| `size-s`           | <100 lines of diff                                       |
| `size-m`           | <400 lines of diff                                       |
| `size-l`           | Needs an ADR — equivalent to `needs-adr`                 |

Every issue gets exactly **one `area-*`**, **one or more `phase-*`**,
**zero or more `kpi-*`**, **one `size-*`**, **zero or one `needs-*`**.

---

## 5. KPIs (the three north stars)

From `docs/PRD.md` §5. Every proposal should advance at least one. If
none applies, the proposal probably belongs in `docs/ideas/`, not in
the issue tracker.

- **Cycle time** — wall-clock from "agent failure" → "ProposedEdit on
  disk". Optimise anything in the Digester→Evolver→Critic pipeline.
- **Regression rate** — count of previously-solved benchmark tasks that
  break after a harness edit. Optimise the Critic, sandboxing, and
  benchmark suite.
- **Improvement rate** — success rate on `benchmarks/` before vs after
  harness evolution. Optimise the benchmark suite's coverage and the
  harness's ability to act on failures.

---

## 6. Focus slots (pick one per sub-agent)

Each slot is a non-overlapping slice of the repo. Stay inside your slot
unless a proposal is trivially a one-line addition that obviously belongs
elsewhere; in that case, label the area correctly and keep the body short.

| Slot key         | Area label(s)             | Bounded scope                                                       |
| ---------------- | ------------------------- | ------------------------------------------------------------------- |
| `trace`          | `area-trace`              | `src/foundry_x/trace/` — recording, schema, backends, CLI            |
| `execution`      | `area-execution`          | `src/foundry_x/execution/` — Runner, model adapters, runaway caps   |
| `evolution`      | `area-evolution`          | `src/foundry_x/evolution/` — Digester, Evolver, Critic, ProposedEdit|
| `observability`  | `area-observability`      | `src/foundry_x/observability/` — KPIs, reports, timeline, render    |
| `harness`        | `area-harness`            | Proposals that operate *on* `harness/` via the Evolver              |
| `benchmarks`     | `area-benchmarks`         | `benchmarks/` — task definitions, fixtures, runner, coverage        |
| `infra`          | `area-infra`              | `infra/` — Docker, llama.cpp ROCm, sandbox guardrails               |
| `docs`           | `area-docs`               | `docs/` — drift, gaps, ADR hygiene, onboarding                      |
| `security`       | `area-security`           | Cross-cutting: SECURITY.md, prompt injection, sandbox, secrets       |

If the orchestrator runs fewer than all nine slots, prefer in this order:

1. `evolution` — **the single highest-leverage slot.**
   `Evolver.propose()` is still a `NotImplementedError` stub. The
   guardrails (rate limiter, diff-size cap, path confinement) are tested
   and in place. The actual meta-agent body that turns a `FailureReport`
   + harness tree into `ProposedEdit(s)` is the #1 blocker to a closed
   evolution loop. Without it, the Digester→Evolver→Critic pipeline is
   a no-op at the Evolver stage.
2. `execution` — the Runner agent loop (ADR-0010) is implemented, but
   `_default_skill_executor` is a stub that returns an ack envelope
   instead of actually running bash/edit/grep/write. Wiring real
   subprocess-backed executors is prerequisite to any meaningful
   benchmark trace.
3. `benchmarks` — coverage gaps in the deterministic suite + the first
   end-to-end LLM-dependent run are the highest-leverage Phase-3 work.
4. `observability` — surfaces that turn a captured trace into a 60-second
   read directly move the cycle-time KPI.
5. `trace`, `harness`, `infra`, `docs`, `security` — run these when the
   orchestrator has bandwidth; they remain non-optional, just
   lower-leverage until the evolution loop is closed and Phase 3 is
   producing traces.

---

## 7. Investigation protocol (per slot)

1. **Read the slot's source files.** Use Grep / Glob / Read with
   bounded scope. Do not skim the whole repo.
2. **Read the slot's existing tests** under `tests/`. Note any
   `needs-evidence` gaps and any test that asserts behaviour the code
   does not yet deliver.
3. **Read the latest 3 ADRs** (`docs/adr/0006`, `0007`, `0008` minimum).
   If your slot has a relevant ADR, obey it; if your proposal would
   contradict one, either drop the proposal or label it `needs-adr`
   and frame the issue as "supersede ADR-NNNN with...".
4. **Skim `docs/ideas/`** for prior art in this slot.
5. **Diff against `git log --oneline -50 develop`** to confirm the slot
   is not already mid-implementation of what you want to propose.
6. **Cross-check open issues** by title and topic. The orchestrator
   will pass you a recent-issues list. Treat that as authoritative.
7. **Cross-check recently-closed issues** for the same reason — if
   `#38` already did what you want to propose, you do not get to
   re-propose it.

If you find an existing issue that is *partially* solved, propose the
remaining slice as a new issue that references the closed one in its
body — do not re-open.

---

## 8. The proposal template (one issue = one object)

Return **3–8 proposals** per slot, ranked by KPI leverage (highest
first). Fewer is fine if the slot is genuinely small; zero is **not**
acceptable unless you justify it explicitly.

Each proposal is a single YAML object with this exact shape:

```yaml
- title: "feat(observability): add per-tool latency histogram to KPI report"
  area: area-observability
  phase: [phase-2]
  kpi: [kpi-cycle-time, kpi-improvement-rate]
  size: size-m
  needs: []                # or [needs-adr] or [needs-evidence]
  motivation: |
    The current kpis.py summary command computes aggregate counts and
    pass rates but does not surface tool-level latency distribution.
    When the Evolver proposes edits, the Critic cannot tell whether a
    regression came from a slow tool or a wrong answer. Trace events
    already carry `duration_ms` (see src/foundry_x/trace/models.py),
    so this is purely a presentation-layer addition.
  evidence: |
    - src/foundry_x/observability/kpis.py:14-62 (current summary logic)
    - src/foundry_x/trace/models.py:TraceEvent.duration_ms
    - tests/test_kpis.py (no current coverage for latency)
  acceptance:
    - foundry-kpis prints p50/p95/p99 per tool name
    - new test asserts a synthetic trace produces the expected buckets
    - docs/PHILOSOPHY.md §1 "evidence over opinion" cited in PR body
  related:
    - "ADR-0007 trace-driven development"
    - "#39 (closed, related: kpis command landed)"
  out_of_scope:
    - "Changing the trace schema (different proposal)"
    - "Touching harness/* directly (different proposal)"
```

### Title rules

- Conventional-Commits prefix: `feat(scope):`, `fix(scope):`,
  `refactor(scope):`, `test(scope):`, `docs(scope):`, `chore(scope):`.
- `scope` matches your slot's directory or layer
  (`trace`, `execution`, `evolution`, `observability`, `harness`,
  `benchmarks`, `infra`, `docs`, `security`).
- Subject ≤ 70 chars, imperative mood, no trailing period.
- Titles must be **unique** across your batch and across the existing
  issues passed in by the orchestrator.

### Acceptance criteria rules

- Each bullet is a concrete, testable assertion.
- No "should work well" or "feels faster". Only things a PR author can
  mark done by running a command or reading a number.

### `out_of_scope` rules

- Always present. Names the adjacent issues you considered and rejected
  for *this* issue, so the orchestrator can decide whether to spin up
  another batch to cover them.

---

## 9. Return contract (your final message)

Your final message must contain **only**:

1. A one-line header: `slot=<your-slot-key> count=<N>`.
2. A fenced ```yaml block containing the YAML list of proposals.
3. A fenced ```text block titled `dedup-notes` listing any existing
   issue numbers you considered and rejected as duplicates (so the
   orchestrator can audit your dedup decisions).
4. A fenced ```text block titled `adjacent-slots` listing slots that
   surfaced proposals you intentionally dropped because they belong
   elsewhere (so the orchestrator can spin up the right next batch).

Do not include commentary outside these blocks. The orchestrator parses
your final message verbatim.

---

## 10. Anti-patterns (reject these on sight)

| Anti-pattern                                              | Why                                                                                     |
| --------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| "Add comprehensive logging"                               | Not concrete. Name the field, the event, the file.                                       |
| "Improve performance"                                     | Not measurable. Name the metric and the baseline.                                       |
| "Refactor X for clarity"                                  | PHILOSOPHY §9 — if you can't say what it does in two sentences, simplify the code, not the issue. |
| "We should consider adding Y"                             | Speculation. Either file `docs/ideas/Y.md` or drop it.                                  |
| "Hand-edit `harness/system_prompt.txt` to ..."            | AGENTS.md §2 — harness is DNA. Frame as an Evolver proposal.                             |
| "Skip the Critic gate for this one because..."            | AGENTS.md §2 + ADR-0004. No. Frame as a Critic improvement instead.                     |
| Proposal larger than 400 lines of diff                    | Split or label `size-l` + `needs-adr`.                                                  |
| Proposal touching > 2 unrelated subsystems               | Out of scope. Split.                                                                    |
| Proposal with `Any` in a pydantic boundary                | ADR-0006 violation. Rewrite or drop.                                                    |

---

## 11. Worked example (for the orchestrator's reference)

A complete minimal return message for slot `observability` might be:

```text
slot=observability count=2
```

```yaml
- title: "feat(observability): add per-tool latency histogram to KPI report"
  area: area-observability
  phase: [phase-2]
  kpi: [kpi-cycle-time, kpi-improvement-rate]
  size: size-m
  needs: []
  motivation: |
    The current kpis.py summary command computes aggregate counts
    and pass rates but does not surface tool-level latency
    distribution. When the Evolver proposes edits, the Critic cannot
    tell whether a regression came from a slow tool or a wrong
    answer.
  evidence: |
    - src/foundry_x/observability/kpis.py:14-62
    - src/foundry_x/trace/models.py TraceEvent.duration_ms
  acceptance:
    - foundry-kpis prints p50/p95/p99 per tool name
    - test asserts a synthetic trace produces expected buckets
  related: ["ADR-0007"]
  out_of_scope:
    - "Changing trace schema"
    - "Touching harness/*"

- title: "feat(observability): render regression_report as a markdown diff table"
  area: area-observability
  phase: [phase-2]
  kpi: [kpi-regression-rate]
  size: size-s
  needs: [needs-evidence]
  motivation: |
    regression_report.py currently writes JSON. Reviewers in
    docs/ROADMAP.md phase 2 read reports in PR descriptions; JSON
    inlined into a PR is unreadable.
  evidence: |
    - src/foundry_x/observability/regression_report.py
    - tests/test_regression_report.py
  acceptance:
    - regression-report --format=md emits a stable diff table
    - golden-file test under tests/test_regression_report.py
  related: ["#38 (closed)", "ADR-0007"]
  out_of_scope:
    - "HTML rendering (separate proposal)"
```

```text
dedup-notes:
  - #39: kpis command exists, but no latency histogram → not duplicate
  - #38: regression reports exist, but only JSON → not duplicate

adjacent-slots:
  - evolution: found 1 proposal about Critic verdict latency → recommend
    slot=evolution next batch
```

---

## 12. Change log

- Initial version. Designed for the nine focus slots in §6.
  Compatible with the existing label taxonomy established by issues
  #3–#55.
- v2 — Refreshed after the 157-issue Phase-1/Phase-2 wave closed.
  Added §0 "Current-state briefing" so sub-agents operating against an
  empty backlog optimise for *next capabilities* (Phase-3 readiness,
  benchmark coverage, observability of real LLM traces) instead of
  gap-filling. Tightened §3 with rule 7 (Phase-3 evidence rule), updated
  §6 slot priority order. Label taxonomy unchanged. YAML contract in §8
  and return contract in §9 unchanged, so existing orchestrator parsers
  keep working.
- v3 — Refreshed after issues closed through #212. Updated §0 briefing
  with accurate counts (13 benchmark tasks, 212+ closed issues) and
  added the two critical blockers that the v2 briefing missed:
  (a) `Evolver.propose()` is still a `NotImplementedError` stub — the
  guardrails are real and tested but the meta-agent body does not exist,
  making the evolution loop a no-op at the Evolver stage; (b) skill
  execution is stubbed — `_default_skill_executor` returns an ack
  envelope, so no benchmark can produce a meaningful trace until real
  subprocess-backed executors are wired. Reordered §6 slot priority to
  put `evolution` first (highest KPI leverage) and `execution` second.
  YAML contract in §8 and return contract in §9 unchanged.
