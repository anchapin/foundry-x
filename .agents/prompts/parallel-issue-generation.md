# Parallel Issue Generation Prompt — FoundryX

Use this prompt to spawn **N sub-agents in parallel** (e.g., via `ClawTeam` or `task` tool with `subagent_type=general`), each assigned a different **focus area** below. Each sub-agent must:

1. Read the relevant project docs listed in its focus area.
2. Analyze the current codebase for gaps, failure modes, or untapped improvements.
3. Produce **2–4 GitHub-ready issues** with:
   - A clear **title** (conventional-commit prefix: `feat/`, `fix/`, `perf/`, `docs/`, `refactor/`)
   - A **body** explaining: motivation, evidence (trace excerpt, benchmark data, or ADR reference), risk, and which KPI(s) it serves.
   - Exact **acceptance criteria** (testable conditions, not vague goals)
   - A **priority label** (`P0` = blocks core loop, `P1` = improves a KPI measurably, `P2` = nice-to-have, `P3` = exploratory)
   - The relevant **KPI tag(s)**: `kpi-cycle-time`, `kpi-regression-rate`, `kpi-improvement-rate`, or `kpi-token-budget`
   - Any **ADR it advances or conflicts with**

Return all issues as a structured JSON array in this shape:
```json
[
  {
    "title": "feat(scope): concise description",
    "body": "## Motivation\n...\n## Evidence\n...\n## Risk\n...\n## Acceptance Criteria\n1. ...\n2. ...\n## ADR(s)",
    "priority": "P0|P1|P2|P3",
    "kpi_tags": ["kpi-cycle-time", ...],
    "adr_tags": ["ADR-XXXX", ...]
  }
]
```

---

## Focus Areas (assign ONE per sub-agent)

### Sub-agent 1 — Evolution Engine (Digester → Evolver → Critic)
**Docs to read:** `docs/PRD.md`, `docs/ROADMAP.md` Phase 2, `docs/CONTEXT.md` §Evolution Loop, `docs/ARCHITECTURE.md` §Components, ADRs `0011`, `0012`, `0018`

**What to look for:**
- Digester: missed failure signals, classifiers that produce false negatives/positives, latency in digest computation
- Evolver: proposed-edit quality, template coverage gaps, LLM prompt effectiveness, rate limiting gaps
- Critic: benchmark gate reliability, false rejections, slow execution, quantization-sweep integration
- **KPI(s):** primarily `kpi-improvement-rate`, `kpi-cycle-time`

### Sub-agent 2 — Trace Store & Observability
**Docs to read:** `docs/PRD.md` §Success Metrics, `docs/CONTEXT.md` §Event kinds, `docs/ARCHITECTURE.md` §Trace store layout, ADRs `0003`, `0007`, `0013`

**What to look for:**
- SQLite WAL mode edge cases, connection pool exhaustion, trace fragmentation
- Missing event kinds or payload fields that would improve digest quality
- KPI computation correctness (`kpi-cycle-time`, `kpi-regression-rate`, `kpi-improvement-rate`, `kpi-token-budget`)
- CLI usability gaps in `foundry-x-trace`, `fx-trace`, `foundry-kpis`
- Retention/pruning automation gaps
- **KPI(s):** all four, but especially measurement accuracy

### Sub-agent 3 — Harness Layer (system_prompt, hooks, skills)
**Docs to read:** `harness/manifest.json`, `harness/system_prompt.txt`, `harness/hooks/`, `harness/skills/`, `docs/PHILOSOPHY.md` §6, ADRs `0004`, `0009`, `0012`

**What to look for:**
- Hook coverage gaps (what middleware is missing?)
- Skill definitions that are stale or missing tool definitions the agent needs
- System prompt drift or gaps the Evolver could address
- Context pruning hook effectiveness (`kpi-token-budget`)
- Injection firewall completeness ( ADR `0009`)
- **KPI(s):** `kpi-improvement-rate`, `kpi-regression-rate`, `kpi-token-budget`

### Sub-agent 4 — Runner & Model Adapter
**Docs to read:** `src/foundry_x/execution/runner.py`, `src/foundry_x/execution/model_adapter.py`, `docs/ARCHITECTURE.md` §Runner, `docs/CONTEXT.md` §Event kinds (session lifecycle), ADR `0010`, ADR `0014`

**What to look for:**
- Model adapter edge cases (streaming, missing telemetry, token counting)
- Runner loop termination edge cases (wall-clock, token budget, max_steps)
- Tool-call latency histogram gaps (issue #709)
- Session resumption or replay gaps
- Model-agnostic boundary weaknesses
- **KPI(s):** `kpi-cycle-time`, `kpi-token-budget`

### Sub-agent 5 — Phase 4 Discovery (post-Phase-3 opportunities)
**Docs to read:** `docs/PRD.md`, `docs/ROADMAP.md`, `docs/PHILOSOPHY.md`, all ADRs, `docs/ideas/` (if present)

**What to look for:**
- Features that would compound the Phase 1–3 foundation but are out of scope
- Multi-agent orchestration opportunities (one FoundryX agent evolving another)
- Benchmark suite expansion (HumanEval integration,SWE-bench, etc.)
- IDE/editor integration points
- Self-hosting improvements (llama-server management, ROCm tuning)
- Cross-harness transfer learning (evolving a harness for task family A using lessons from task family B)
- **Note:** These must pass the "optimism budget" test (§8 of PHILOSOPHY.md) — tie every idea to at least one KPI
- **KPI(s):** all, but forward-looking

### Sub-agent 6 — Developer Experience & Documentation
**Docs to read:** `README.md`, `CONTRIBUTING.md`, `docs/OPERATOR.md`, `docs/SECURITY.md`, all ADRs

**What to look for:**
- Onboarding friction for new contributors
- Missing tutorial or getting-started guide
- ADR coverage gaps (important decisions not yet recorded)
- CLI `--help` text quality
- Error message quality (do they guide toward recovery?)
- Pre-commit hook completeness
- **KPI(s):** `kpi-cycle-time` (faster contributor onboarding → faster iteration)

---

## Hard Constraints for All Sub-agents

1. **Never duplicate an existing issue.** Cross-check the issue tracker (look for similar titles or linked PRs/commits) before filing.
2. **Never propose a harness hand-edit.** All harness changes must route through the Evolver → Critic pipeline (ADR-0004).
3. **Every issue must be actionable.** An agent reading this issue must be able to: reproduce the problem, write a test, implement the fix, and verify the fix — all without asking follow-up questions.
4. **Tie every issue to evidence.** Use trace excerpts, benchmark numbers, ADR citations, or documented user needs. "I think X would be nice" is not acceptable.
5. **Respect the architecture.** Do not propose changes that blur the foundry/harness boundary without a corresponding ADR.

## Output Format

Each sub-agent returns:
1. A brief **analysis summary** (3–5 sentences): what it found, why it matters.
2. The **JSON array** of issues as specified above.
3. A **confidence score** (0.0–1.0) for each issue — how certain are you that this is a real gap and not a misunderstanding?

Collect all sub-agent outputs and merge into a single ranked list sorted by: `P0` first, then `P1`, then by `kpi-improvement-rate` weight, then alphabetically.
