# Roadmap: Project "FoundryX"

This project follows a three-phase development cycle.

> **Live status.** For the most current phase progress — including shipped
> components, open stubs, and open issue counts — see the "Current-state
> briefing" in [`docs/ISSUE_GENERATION_PROMPT.md` §0](ISSUE_GENERATION_PROMPT.md).

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

**Status: NOT STARTED** (as of 929b327, 2026-07-11) — Quantization
sweep, context pruning at scale, real-LLM benchmark runs, and token
budget enforcement (plumbed but not counted) remain to be implemented.

**Goal:** Hardware and model performance tuning.

**Milestones:**

- Automate model swapping: test the same harness against different quantizations (for example Q4 vs. Q5) to find the "intelligence floor."
- Optimize context management: refine the hooks to prune historical logs efficiently and keep inference latency low on the 5600G / 6600 XT setup.
