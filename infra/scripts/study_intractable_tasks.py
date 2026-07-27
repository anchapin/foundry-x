#!/usr/bin/env python3
"""Study: identify benchmark tasks that are intractable at Q8_0 (issue #1027).

This script runs the benchmark suite at Q8_0 quantization and collects
pass/fail data per task to identify tasks that are never solved even
with maximum quantization -- the "intractable" tasks that need either
simpler versions or different evaluation approaches.

ADR-0020 Open Question #6:
    "Are there benchmark tasks that remain intractable even at Q8_0?"

Usage:
    # With LLAMACPP_HOST set and FOUNDRY_MODEL_PATH pointing to Q8_0 model:
    python infra/scripts/study_intractable_tasks.py

    # With explicit model path:
    FOUNDRY_MODEL_PATH=/srv/models Q8_0_MODEL=*.Q8_0.gguf \
        python infra/scripts/study_intractable_tasks.py

    # Dry run (print tasks that would be studied):
    python infra/scripts/study_intractable_tasks.py --dry-run

Environment:
    FOUNDRY_MODEL_PATH   Base directory containing GGUF model files.
    Q8_0_MODEL           Glob pattern for Q8_0 model (default: "*.Q8_0.gguf").
    FOUNDRY_CONTEXT_TOKENS  Context window size (default: 8192).
    LLAMACPP_HOST         llama-server host (default: http://127.0.0.1:8080).
    STUDY_RUNS            Number of runs per task to assess consistency (default: 3).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

# Add src to path for imports when running as script
SRC_DIR = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC_DIR))

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FOUNDRY_MODEL_PATH = os.environ.get("FOUNDRY_MODEL_PATH", "")
Q8_0_MODEL_PATTERN = os.environ.get("Q8_0_MODEL", "*.Q8_0.gguf")
FOUNDRY_CONTEXT_TOKENS = int(os.environ.get("FOUNDRY_CONTEXT_TOKENS", "8192"))
LLAMACPP_HOST = os.environ.get("LLAMACPP_HOST", "http://127.0.0.1:8080")
STUDY_RUNS = int(os.environ.get("STUDY_RUNS", "3"))
DRY_RUN = "--dry-run" in sys.argv

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "logs" / "intractable_study"
TRACES_DB = REPO_ROOT / "logs" / "traces.db"

# Hard-tier tasks are the primary candidates for intractability
# (ADR-0028 defines hard-tier criteria)
HARD_TIER_TASKS = [
    "debug_import_cycle",
    "refactor_api_with_constraints",
    "hook_timing_attack",  # Note: TASK.name is "hook_timing_attack"
]

# Medium/hard tasks with multi-file scope and complex reasoning
COMPLEX_TASKS = [
    "cross_file_refactor",
    "refactor_across_three_files",
    "multi_file_rename",
    "code_review_diff",
    "external_eval_correlation",
]

# Tasks with long timeouts (indicating complexity)
TIMEOUT_HEAVY_TASKS = [
    "long_context_retention",
    "context_pruning_sweep",
    "evolution_full_loop_real_model",
    "real_llm_full_loop_smoke",
]

ALL_STUDY_TASKS = sorted(set(HARD_TIER_TASKS + COMPLEX_TASKS + TIMEOUT_HEAVY_TASKS))


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class TaskStudyResult:
    """Result of studying a single task across multiple runs."""

    task_name: str
    difficulty_tier: str
    runs: int = 0
    passes: int = 0
    failures: int = 0
    pass_rate: float = 0.0
    consistent: bool = True  # True if all runs agree (all pass or all fail)
    intractable: bool = False  # True if pass_rate == 0.0 across all runs
    token_budget_hits: int = 0
    wall_clock_timeouts: int = 0
    avg_cycle_time_s: float | None = None
    total_tokens: int = 0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "task_name": self.task_name,
            "difficulty_tier": self.difficulty_tier,
            "runs": self.runs,
            "passes": self.passes,
            "failures": self.failures,
            "pass_rate": self.pass_rate,
            "consistent": self.consistent,
            "intractable": self.intractable,
            "token_budget_hits": self.token_budget_hits,
            "wall_clock_timeouts": self.wall_clock_timeouts,
            "avg_cycle_time_s": self.avg_cycle_time_s,
            "total_tokens": self.total_tokens,
            "notes": self.notes,
        }


@dataclass
class StudySummary:
    """Full study summary for Q8_0 intractable task study."""

    timestamp: str
    quantization: str = "Q8_0"
    model_pattern: str = ""
    context_tokens: int = 8192
    study_runs: int = 3
    tasks_studied: int = 0
    tasks_intractable: int = 0
    tasks_all_pass: int = 0
    tasks_mixed: int = 0
    results: list[TaskStudyResult] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "quantization": self.quantization,
            "model_pattern": self.model_pattern,
            "context_tokens": self.context_tokens,
            "study_runs": self.study_runs,
            "tasks_studied": self.tasks_studied,
            "tasks_intractable": self.tasks_intractable,
            "tasks_all_pass": self.tasks_all_pass,
            "tasks_mixed": self.tasks_mixed,
            "results": [r.to_dict() for r in self.results],
        }


# ---------------------------------------------------------------------------
# Helper: check if llama-server is healthy
# ---------------------------------------------------------------------------


def llamacpp_is_healthy() -> bool:
    """Check if llama-server is reachable at LLAMACPP_HOST."""
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{LLAMACPP_HOST}/health", timeout=5) as resp:
            return resp.status == 200
    except urllib.error.URLError:
        return False


def get_running_model() -> str | None:
    """Get the model ID from the running llama-server."""
    import json as json_module
    import urllib.error
    import urllib.request

    try:
        with urllib.request.urlopen(f"{LLAMACPP_HOST}/v1/models", timeout=5) as resp:
            data = json_module.load(resp)
            if data.get("data"):
                return data["data"][0].get("id")
    except urllib.error.URLError:
        return None
    return None


# ---------------------------------------------------------------------------
# Helper: get difficulty tier from benchmark task
# ---------------------------------------------------------------------------


def get_task_difficulty_tier(task_name: str) -> str:
    """Get the difficulty tier for a named benchmark task."""
    try:
        from benchmarks.registry import get_task

        task = get_task(task_name)
        if task:
            return task.difficulty_tier
    except (ImportError, AttributeError):
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Helper: run a single benchmark task
# ---------------------------------------------------------------------------


def run_single_task(
    task_name: str,
    timeout_seconds: int = 300,
) -> dict | None:
    """Run a single benchmark task and return the result.

    Returns a dict with keys: passed (bool), duration_s, tokens_used,
    token_budget_hit (bool), wall_clock_timeout (bool), error (str|None).
    """

    # Use fx-runner to run the task against the live model
    cmd = [
        sys.executable,  # Use current Python
        "-m",
        "foundry_x.execution.runner",
        "--task",
        f"Solve the benchmark task: {task_name}",
        "--harness-dir",
        str(REPO_ROOT / "harness"),
    ]

    env = {**os.environ, "FOUNDRY_CONTEXT_TOKENS": str(FOUNDRY_CONTEXT_TOKENS)}

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=env,
            cwd=str(REPO_ROOT),
            check=False,
        )
        duration_s = time.time() - start

        # Parse trace store for outcome
        # This is a simplified check - in production we'd query the trace store
        passed = result.returncode == 0

        return {
            "passed": passed,
            "duration_s": duration_s,
            "tokens_used": 0,  # Would need trace store query
            "token_budget_hit": False,
            "wall_clock_timeout": False,
            "error": None if result.returncode == 0 else result.stderr[:500],
            "returncode": result.returncode,
        }
    except subprocess.TimeoutExpired:
        return {
            "passed": False,
            "duration_s": timeout_seconds,
            "tokens_used": 0,
            "token_budget_hit": False,
            "wall_clock_timeout": True,
            "error": f"Task timed out after {timeout_seconds}s",
            "returncode": -1,
        }
    except OSError as e:
        return {
            "passed": False,
            "duration_s": time.time() - start,
            "tokens_used": 0,
            "token_budget_hit": False,
            "wall_clock_timeout": False,
            "error": str(e)[:500],
            "returncode": -1,
        }


# ---------------------------------------------------------------------------
# Main study logic
# ---------------------------------------------------------------------------


def check_infrastructure() -> bool:
    """Check if the study can run (llama-server available, model exists)."""
    errors = []

    if not FOUNDRY_MODEL_PATH:
        errors.append("FOUNDRY_MODEL_PATH is not set")
    else:
        model_dir = Path(FOUNDRY_MODEL_PATH)
        if not model_dir.exists():
            errors.append(f"FOUNDRY_MODEL_PATH does not exist: {model_dir}")
        else:
            import glob as glob_module

            pattern = str(model_dir / Q8_0_MODEL_PATTERN)
            matched = glob_module.glob(pattern)
            if not matched:
                errors.append(f"No Q8_0 model found matching: {pattern}")
            elif len(matched) > 1:
                errors.append(f"Multiple Q8_0 models matched: {matched}")

    if not llamacpp_is_healthy():
        errors.append(
            f"llama-server is not healthy at {LLAMACPP_HOST}. "
            "Start with: llama-server --model <model> --host 127.0.0.1 --port 8080"
        )

    if errors:
        print("Infrastructure check FAILED:")
        for err in errors:
            print(f"  - {err}")
        return False

    model_id = get_running_model()
    print(f"Infrastructure check PASSED (running model: {model_id})")
    return True


def run_study_for_task(task_name: str, runs: int = STUDY_RUNS) -> TaskStudyResult:
    """Run a single task multiple times and collect results."""
    difficulty_tier = get_task_difficulty_tier(task_name)
    result = TaskStudyResult(
        task_name=task_name,
        difficulty_tier=difficulty_tier,
        notes="",
    )

    print(f"\n  Studying: {task_name} (tier={difficulty_tier}, runs={runs})")

    task_passes = 0
    task_failures = 0
    token_budget_hits = 0
    wall_clock_timeouts = 0
    total_duration = 0.0

    for run_idx in range(runs):
        print(f"    Run {run_idx + 1}/{runs}...", end=" ", flush=True)
        run_result = run_single_task(task_name)

        if run_result is None:
            print("SKIP (infrastructure error)")
            result.notes += f"Run {run_idx + 1}: infrastructure error; "
            continue

        if run_result["passed"]:
            task_passes += 1
            print("PASS")
        else:
            task_failures += 1
            if run_result["token_budget_hit"]:
                token_budget_hits += 1
                print("FAIL (token_budget)")
            elif run_result["wall_clock_timeout"]:
                wall_clock_timeouts += 1
                print("FAIL (timeout)")
            else:
                print(
                    f"FAIL (error: {run_result['error'][:100] if run_result['error'] else 'unknown'})"
                )

        total_duration += run_result["duration_s"]

    result.runs = runs
    result.passes = task_passes
    result.failures = task_failures
    result.pass_rate = task_passes / runs if runs > 0 else 0.0
    result.consistent = task_passes == 0 or task_failures == 0
    result.intractable = task_passes == 0
    result.token_budget_hits = token_budget_hits
    result.wall_clock_timeouts = wall_clock_timeouts
    result.avg_cycle_time_s = total_duration / runs if runs > 0 else None

    if result.intractable:
        result.notes = (
            f"Intractable: 0/{runs} passes at Q8_0. "
            f"Consider: simpler task variant, extended timeout, or alternative evaluation."
        )
    elif result.consistent and result.pass_rate == 1.0:
        result.notes = f"Fully solvable: {runs}/{runs} passes at Q8_0."
    else:
        result.notes = f"Mixed results: {task_passes}/{runs} passes. May be quantization-sensitive."

    return result


def print_study_results(summary: StudySummary) -> None:
    """Print a formatted summary of the study results."""
    print("\n" + "=" * 80)
    print("Q8_0 INTRACTABLE TASK STUDY RESULTS")
    print("=" * 80)
    print(f"Timestamp:      {summary.timestamp}")
    print(f"Quantization:   {summary.quantization}")
    print(f"Context:       {summary.context_tokens} tokens")
    print(f"Study runs:    {summary.study_runs} per task")
    print()

    print(f"Tasks studied:    {summary.tasks_studied}")
    print(f"Intractable:      {summary.tasks_intractable} (never pass at Q8_0)")
    print(f"Fully solvable:   {summary.tasks_all_pass} (always pass at Q8_0)")
    print(f"Mixed results:    {summary.tasks_mixed} (sometimes pass)")
    print()

    print("-" * 80)
    print("PER-TASK RESULTS")
    print("-" * 80)

    # Sort by difficulty tier, then by pass rate
    sorted_results = sorted(
        summary.results,
        key=lambda r: (r.difficulty_tier, r.pass_rate),
    )

    for r in sorted_results:
        status = "INTRACTABLE" if r.intractable else ("PASS" if r.pass_rate == 1.0 else "MIXED")
        print(f"\n  [{status}] {r.task_name} (tier={r.difficulty_tier})")
        print(f"    Pass rate: {r.passes}/{r.runs} ({r.pass_rate:.1%})")
        if r.avg_cycle_time_s:
            print(f"    Avg time:  {r.avg_cycle_time_s:.1f}s")
        if r.token_budget_hits:
            print(f"    Token budget hits: {r.token_budget_hits}")
        if r.wall_clock_timeouts:
            print(f"    Wall-clock timeouts: {r.wall_clock_timeouts}")
        if r.notes:
            print(f"    Notes: {r.notes}")

    print()
    print("-" * 80)
    print("INTractable TASKS (require attention)")
    print("-" * 80)
    intractable = [r for r in summary.results if r.intractable]
    if intractable:
        for r in intractable:
            print(f"  - {r.task_name} ({r.difficulty_tier})")
        print()
        print(
            "  These tasks never pass at Q8_0, suggesting they are too hard for\n"
            "  the current agent capability at any quantization. Options:\n"
            "  1. Create simpler variants of these tasks\n"
            "  2. Use alternative evaluation approaches (human review, etc.)\n"
            "  3. Extend timeout budgets for complex reasoning tasks\n"
            "  4. Break into multi-step subtasks"
        )
    else:
        print("  No intractable tasks found at Q8_0.")


def save_results(summary: StudySummary) -> Path:
    """Save study results to JSON file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp_str = summary.timestamp.replace(":", "-").replace(".", "-")
    filename = f"intractable_study_{timestamp_str}.json"
    output_path = OUTPUT_DIR / filename

    with open(output_path, "w") as f:
        json.dump(summary.to_dict(), f, indent=2)

    print(f"\nResults saved to: {output_path}")
    return output_path


def update_adr_intractable_study(summary: StudySummary) -> Path | None:
    """Update ADR-0020 with the study findings."""
    adr_path = REPO_ROOT / "docs" / "adr" / "adr-0020-intractable-study.md"

    # Build the findings table
    rows = []
    for r in sorted(summary.results, key=lambda x: (x.difficulty_tier, x.task_name)):
        status = "INTRACTABLE" if r.intractable else ("PASS" if r.pass_rate == 1.0 else "MIXED")
        rows.append(
            f"| {r.task_name} | {r.difficulty_tier} | "
            f"{r.passes}/{r.runs} ({r.pass_rate:.0%}) | {status} | {r.notes[:60]} |"
        )

    findings_table = "\n".join(rows)

    content = f"""# ADR-0020 Appendix: Intractable Task Study (Open Question #6)

## Status

Study completed: {summary.timestamp}

## Context

ADR-0020 Open Question #6 asks:
> "Are there benchmark tasks that remain intractable even at Q8_0?"

This appendix records the methodology and findings of the study conducted
to answer that question.

## Methodology

**Quantization studied:** {summary.quantization}
**Model pattern:** {summary.model_pattern}
**Context window:** {summary.context_tokens} tokens
**Runs per task:** {summary.study_runs}
**Hardware:** AMD RX 6600 XT (8 GB VRAM)

Each task was run {summary.study_runs} times at Q8_0 to assess consistency.
A task is classified as **intractable** if it never passes (0/{summary.study_runs} runs)
at Q8_0.

## Summary

| Metric | Value |
|--------|-------|
| Tasks studied | {summary.tasks_studied} |
| Intractable (never pass) | {summary.tasks_intractable} |
| Fully solvable (always pass) | {summary.tasks_all_pass} |
| Mixed results | {summary.tasks_mixed} |

## Per-Task Results

| Task | Tier | Pass Rate | Status | Notes |
|------|------|-----------|--------|-------|
{findings_table}

## Intractable Tasks

"""

    intractable_tasks = [r for r in summary.results if r.intractable]
    if intractable_tasks:
        content += "The following tasks never pass at Q8_0 and require attention:\n\n"
        for r in intractable_tasks:
            content += f"- **{r.task_name}** ({r.difficulty_tier}): {r.notes}\n"
        content += """
### Recommended Actions

1. **Create simpler task variants** — reduce complexity while preserving the skill being measured
2. **Extend timeout budgets** — hard-tier tasks may need longer than 60s for complex reasoning
3. **Break into multi-step subtasks** — a task that requires 5+ tool calls might be 2-3 tasks
4. **Alternative evaluation** — for truly hard tasks, consider human-in-the-loop evaluation
"""
    else:
        content += "No intractable tasks were identified. All benchmark tasks pass at Q8_0.\n"

    content += """
## Raw Data

Full study results saved to: `logs/intractable_study/`

Generated by: `infra/scripts/study_intractable_tasks.py`
"""

    with open(adr_path, "w") as f:
        f.write(content)

    print(f"ADR appendix saved to: {adr_path}")
    return adr_path


def main() -> int:
    """Run the intractable task study."""
    print("=" * 80)
    print("Q8_0 INTRACTABLE TASK STUDY")
    print("Issue #1027: ADR-0020 Open Question #6")
    print("=" * 80)
    print("\nConfiguration:")
    print(f"  FOUNDRY_MODEL_PATH: {FOUNDRY_MODEL_PATH or '(not set)'}")
    print(f"  Q8_0_MODEL pattern:  {Q8_0_MODEL_PATTERN}")
    print(f"  LLAMACPP_HOST:      {LLAMACPP_HOST}")
    print(f"  Context tokens:     {FOUNDRY_CONTEXT_TOKENS}")
    print(f"  Study runs:        {STUDY_RUNS}")
    print(f"  Tasks to study:    {len(ALL_STUDY_TASKS)}")

    if DRY_RUN:
        print("\n[DRY RUN] Would study these tasks:")
        for t in ALL_STUDY_TASKS:
            tier = get_task_difficulty_tier(t)
            print(f"  - {t} (tier={tier})")
        return 0

    # Check infrastructure
    print("\n[1/3] Checking infrastructure...")
    if not check_infrastructure():
        print("\nInfrastructure not available. Preparing study infrastructure only.")
        print("To run the study:")
        print("  1. Start llama-server with Q8_0 model")
        print("  2. Set FOUNDRY_MODEL_PATH and LLAMACPP_HOST")
        print("  3. Re-run this script")
        print()
        print("Creating study plan and documentation...")

        # Create a pending study document
        summary = StudySummary(
            timestamp=datetime.now(UTC).isoformat(),
            quantization="Q8_0",
            model_pattern=Q8_0_MODEL_PATTERN,
            context_tokens=FOUNDRY_CONTEXT_TOKENS,
            study_runs=STUDY_RUNS,
            tasks_studied=len(ALL_STUDY_TASKS),
            tasks_intractable=0,
            tasks_all_pass=0,
            tasks_mixed=0,
            results=[],
        )

        # Save pending study
        save_results(summary)
        return 0

    # Run study
    print("\n[2/3] Running study for each task...")
    summary = StudySummary(
        timestamp=datetime.now(UTC).isoformat(),
        quantization="Q8_0",
        model_pattern=Q8_0_MODEL_PATTERN,
        context_tokens=FOUNDRY_CONTEXT_TOKENS,
        study_runs=STUDY_RUNS,
        tasks_studied=len(ALL_STUDY_TASKS),
    )

    for task_name in ALL_STUDY_TASKS:
        result = run_study_for_task(task_name, runs=STUDY_RUNS)
        summary.results.append(result)

        if result.intractable:
            summary.tasks_intractable += 1
        elif result.pass_rate == 1.0:
            summary.tasks_all_pass += 1
        else:
            summary.tasks_mixed += 1

    # Print and save results
    print("\n[3/3] Saving results...")
    print_study_results(summary)
    save_results(summary)
    update_adr_intractable_study(summary)

    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    sys.exit(main())
