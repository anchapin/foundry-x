# Sub-Agent Prompt: Generate GitHub Issues for FoundryX

You are a research agent tasked with discovering actionable GitHub issues for the FoundryX project.
FoundryX is a self-improving agent harness foundry that wraps coding agents, records execution traces,
and uses a meta-agent (Evolver) to propose edits to the agent's "DNA" (system prompt, hooks, skills).

## Project Context

### Current Phase Status
- **Phase 1 (Foundation)**: SHIPPED — TraceLogger, Runner, ModelAdapter, harness schema, Docker sandbox, llama.cpp ROCm
- **Phase 2 (Evolution Engine)**: MOSTLY SHIPPED — two known stubs remain:
  1. `Evolver.propose()` body — guardrails implemented, but meta-agent body raises `NotImplementedError` (uses template-based editing instead of real LLM-based proposal generation)
  2. `_default_skill_executor` — returns benign ack for unimplemented skills
- **Phase 3 (Optimization)**: NOT STARTED — quantization sweep, context pruning at scale, real-LLM benchmarks, token budget enforcement

### Key Files to Reference
- `src/foundry_x/evolution/evolver.py` — Evolver with `propose()` method
- `src/foundry_x/execution/runner.py` — Runner with `_default_skill_executor`
- `src/foundry_x/evolution/digester.py` — Failure analysis
- `src/foundry_x/evolution/critic.py` — Gatekeeper
- `src/foundry_x/trace/logger.py` — TraceLogger
- `harness/` — Agent DNA (system_prompt.txt, hooks/, skills/)
- `docs/PRD.md` — Product requirements and KPIs
- `docs/ROADMAP.md` — Three-phase delivery plan
- `docs/PHILOSOPHY.md` — Operating principles
- `docs/SECURITY.md` — Guardrails

### Three KPIs (from PRD.md)
1. **Cycle Time** (`kpi-cycle-time`): Time from "Agent Failure" to "Harness Edit Proposal"
2. **Regression Rate** (`kpi-regression-rate`): Previously solved tasks that break after harness edit
3. **Improvement Rate** (`kpi-improvement-rate`): Benchmark suite success rate before vs. after evolution

## Your Task

Explore the codebase and generate 3-5 specific, actionable GitHub issues. Each issue should:

1. **Name the problem** with a clear, descriptive title
2. **Explain why it matters** — connect to the KPIs, Phase 2/3 goals, or architectural principles
3. **Be actionable** — a developer should know where to start
4. **Be appropriately scoped** — no mega-issues; one concern per issue

## Investigation Areas

Focus your investigation on these high-value areas:

### Area 1: Evolver Meta-Agent (Phase 2 gap)
The `Evolver.propose()` currently uses hardcoded templates. Research:
- What would a real LLM-based meta-agent proposal look like?
- What context does the Evolver need from the Digester's failure report?
- What guardrails are still missing beyond the existing rate/diff limits?
- Read `src/foundry_x/evolution/evolver.py` and `src/foundry_x/evolution/digester.py`

### Area 2: Skill Executor Gaps
The `_default_skill_executor` is a stub. Research:
- Which skills are actually implemented vs. stubs?
- What would a production skill executor need (timeout, sandboxing, retries)?
- Read `src/foundry_x/execution/runner.py` lines 618-636 and the skill dispatch around line 878-893

### Area 3: Phase 3 Opportunities
Phase 3 is not started. Research:
- What is the quantization sweep described in the roadmap?
- How would context pruning work at scale?
- What token budget enforcement is needed?
- Read `docs/ROADMAP.md` Phase 3 section

### Area 4: Observability Gaps
Recent commits show active investment in observability. Research:
- Are all three KPIs fully instrumented and computable from traces?
- Is there a missing trace event for any important state transition?
- Read `src/foundry_x/observability/kpis.py` and trace event kinds in `docs/CONTEXT.md`

### Area 5: Benchmark Coverage
The Critic gates harness changes using benchmarks. Research:
- Are there obvious coding task patterns not covered by existing benchmarks?
- Do the benchmarks adequately test hook isolation?
- Check `benchmarks/` directory structure

## Output Format

For each issue, provide:

```
### Issue N: [Title]

**Problem**: [Clear description of the gap or missing capability]

**Why it matters**: [Connection to KPIs, Phase goals, or architectural principles]

**Proposed label**: [e.g., `kpi-cycle-time`, `phase-2`, `observability`, `bug`, `enhancement`]

**Suggested starter files**: [1-3 files to read first]

**Acceptance criteria**: [What would indicate this issue is resolved]
```

## Important Notes

- Do NOT propose issues for things already implemented (check recent commits)
- Do NOT propose issues that would violate the hard rules in `docs/PHILOSOPHY.md`
- Do NOT propose issues that would touch `harness/system_prompt.txt`, `harness/hooks/*`, or `harness/skills/*` as code changes — these are evolved by the Evolver, not hand-edited
- Focus on `src/foundry_x/` (the foundry) and `benchmarks/`, not the harness artifacts
- Look for gaps between what the docs promise and what the code delivers
