# Roadmap: Project "FoundryX"

This project follows a three-phase development cycle.

## Phase 1: Foundation -- the execution and trace layer

**Status: SHIPPED** (as of 929b327, 2026-07-11) — TraceLogger, Runner,
ModelAdapter, harness schema, Docker sandbox, llama.cpp ROCm all
implemented and tested.

**Goal:** Establish the bridge between the agent and the LLM.

**Milestones:**

- Set up `llama-server` (llama.cpp) with ROCm for the RX 6600 XT.
- Define the Harness schema: create a directory structure to version-control prompts, tools, and hooks as JSON or YAML.
- Build the `TraceLogger`: a wrapper for OpenCode that records the full context (prompt, tool calls, output, outcome) to a structured SQLite database or JSONL file.

## Phase 2: The Evolution Engine -- the meta-agent layer

**Status: MOSTLY SHIPPED** (as of 929b327, 2026-07-11) — The full
Digester → Evolver → Critic chain exists with pydantic models, pytest
coverage, and CI gates. Two known stubs remain:

1. **`Evolver.propose()` body** (`src/foundry_x/evolution/evolver.py`) —
   guardrails are implemented and tested, but the meta-agent body that
   turns a `FailureReport` into `ProposedEdit`(s) raises
   `NotImplementedError`.
2. **`_default_skill_executor`** (`src/foundry_x/execution/runner.py`) —
   returns a benign ack envelope; does *not* actually run bash/edit/grep/write.

**Goal:** Create the automated feedback loop.

**Milestones:**

- Build the `Digester`: a script that parses logs and generates "failure reports" for failed coding tasks.
- Build the `Evolver`: a meta-agent script that takes the failure report and modifies the harness schema.
- Implement the `Critic`: a gatekeeper that runs unit tests (or simple syntax checks) on new harness configurations before allowing a "production" deployment.

## Phase 3: Optimization and Scaling

**Status: SHIPPED** (as of 929b327, 2026-07-11) — Quantization
sweep, token-aware context pruning, full LLM Evolver with rate limiting
and retry logic, real-model smoke tests, and token budget observability
all implemented. Remaining work captured in phase-3b issues.

**Goal:** Hardware and model performance tuning.

**What was delivered:**

- `foundry-sweep` CLI + `Critic.quantization_sweep()` (issues #464, PRs #473, #527, #526, #528)
- `TokenAwarePruningHook` + `FOUNDRY_CONTEXT_TOKENS` config (issues #465, #418, PR #519)
- Token budget observability: KPIs, session-summary, regression-report (issue #466)
- Full LLM Evolver with rate limiting and retry logic (issues #476–#481, PRs #516, #518, #523, #533, #536)
- Real-model full-loop smoke test (issues #483, #484, PRs #525, #529)
- Token usage in trace events + `RuntimeWarning` on missing telemetry (issues #191, #486, PRs #489, #514, #521)

**Phase 3b remaining work:**

- #537 — `BenchmarkTask.token_budget` needs wiring to `RunLimits` enforcement
- #539 — aggregate and publish Phase 3 performance tuning findings
- #540 — verify #481 is fully closed (validated in PR #536); close if duplicate
