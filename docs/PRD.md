# Product Requirements Document: Foundational Agent Harness Foundry

## 1. Project Overview

FoundryX is a framework for developing and evolving agentic coding harnesses. It transitions the development of AI agents from "manual prompt engineering" to "automated harness evolution."

## 2. User Requirements

- **User:** An AI Engineer (You).
- **Requirement 1:** Ability to store agent "DNA" (prompts, hooks, skill sets) in a version-controlled, machine-readable format.
- **Requirement 2:** Automated execution of agent tasks with mandatory logging of inputs and outputs.
- **Requirement 3:** A feedback-loop mechanism that analyzes failure traces and suggests prompt or hook modifications.

## 3. Functional Specifications

### Harness Schema

- `system_prompt.txt`: The core agent personality.
- `hooks/`: Python modules that act as middleware for agent tool calls.
- `skills/`: JSON definitions for agent-accessible functions.

### The Evolution Loop

- **Input:** Task failure log.
- **Process:** Meta-agent evaluates diffs between desired outcome and actual trace.
- **Output:** A `ProposedEdit` object containing specific line changes for prompts or logic.

### Execution Environment

Must support containerization (Docker) to keep the Linux Mint host safe during agent experimentation.

## 4. Technical Constraints

- **Hardware:** Must optimize for local AMD ROCm inference.
- **Environment:** Python 3.11+ using `uv` for dependency management.
- **Modularity:** The harness must be model-agnostic; it should work whether you are using a 7B coder model or a 70B generalist model.

## 5. Success Metrics (KPIs)

- **Cycle Time:** Time taken from "Agent Failure" to "Harness Edit Proposal."
- **Regression Rate:** Number of previously solved tasks that break after a harness edit.
- **Improvement Rate:** Success rate on a standardized coding benchmark (for example a subset of HumanEval) before vs. after harness evolution.
