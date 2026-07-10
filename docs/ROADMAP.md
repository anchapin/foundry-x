# Roadmap: Project "FoundryX"

This project follows a three-phase development cycle.

## Phase 1: Foundation -- the execution and trace layer

**Goal:** Establish the bridge between the agent and the LLM.

**Milestones:**

- Set up `llama-server` (llama.cpp) with ROCm for the RX 6600 XT.
- Define the Harness schema: create a directory structure to version-control prompts, tools, and hooks as JSON or YAML.
- Build the `TraceLogger`: a wrapper for OpenCode that records the full context (prompt, tool calls, output, outcome) to a structured SQLite database or JSONL file.

## Phase 2: The Evolution Engine -- the meta-agent layer

**Goal:** Create the automated feedback loop.

**Milestones:**

- Build the `Digester`: a script that parses logs and generates "failure reports" for failed coding tasks.
- Build the `Evolver`: a meta-agent script that takes the failure report and modifies the harness schema.
- Implement the `Critic`: a gatekeeper that runs unit tests (or simple syntax checks) on new harness configurations before allowing a "production" deployment.

## Phase 3: Optimization and Scaling

**Goal:** Hardware and model performance tuning.

**Milestones:**

- Automate model swapping: test the same harness against different quantizations (for example Q4 vs. Q5) to find the "intelligence floor."
- Optimize context management: refine the hooks to prune historical logs efficiently and keep inference latency low on the 5600G / 6600 XT setup.
