from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from foundry_x.trace.logger import TraceLogger


async def run_task(task: str, harness_dir: Path, log: TraceLogger, session_id: str) -> None:
    raise NotImplementedError(
        "Phase 1 wiring: instantiate your OpenCode client here, "
        "fan tool calls through harness.hooks.get_registry(), "
        "and stream events into the TraceLogger."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="FoundryX execution runner")
    parser.add_argument("--task", required=True, help="Task prompt for the agent")
    parser.add_argument(
        "--harness-dir",
        default=os.environ.get("FOUNDRY_HARNESS_DIR", "./harness"),
    )
    parser.add_argument(
        "--trace-path",
        default=os.environ.get("FOUNDRY_TRACE_PATH", "./logs/traces.db"),
    )
    args = parser.parse_args()

    harness_dir = Path(args.harness_dir).resolve()
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    logger = TraceLogger(args.trace_path)
    harness_version = "0.1.0"

    with logger.session(harness_version=harness_version) as session_id:
        logger.record(session_id, kind="task_received", payload={"prompt": args.task})
        asyncio.run(run_task(args.task, harness_dir, logger, session_id))


if __name__ == "__main__":
    main()
