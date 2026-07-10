from __future__ import annotations

import argparse
import asyncio
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from foundry_x.trace.logger import TraceLogger

# Default wall-clock cap (seconds) for a single task. Generous enough for a
# real model turn, small enough to abort a runaway loop before it exhausts
# the GPU/wallet. Mandated by docs/SECURITY.md "Runaway detection".
DEFAULT_TASK_TIMEOUT_S: float = 600.0

# Last-resort literal when neither ``harness/VERSION`` nor a git tag of the
# harness directory can be read (issue #11). The harness is the evolved
# artifact (PHILOSOPHY.md §6, ADR-0004); the *resolved* version is what gets
# stamped into trace sessions so each trace is attributable to the harness
# revision that produced it.
_FALLBACK_HARNESS_VERSION: str = "0.1.0"


def resolve_harness_version(harness_dir: Path) -> str:
    """Return the version of the harness rooted at ``harness_dir``.

    Resolution order (issue #11):

    1. ``harness_dir / "VERSION"`` — a single-line text file owned by the
       harness itself. The foundry reads a value the harness owns; it does
       not hand-edit harness DNA (AGENTS.md §2, ADR-0004).
    2. ``git describe --tags --always`` run in ``harness_dir``. Lets an
       evolved checkout self-describe even before a ``VERSION`` bump.
    3. The :data:`_FALLBACK_HARNESS_VERSION` literal.

    Whitespace (including a trailing newline) is stripped from the file
    contents and from the git output so the stamped value is canonical.

    Failures (missing file, git not installed, git error, non-repo
    directory) fall through silently to the next source rather than
    aborting the run; a missing version stamp is preferable to a run that
    cannot start.
    """
    version_file = harness_dir / "VERSION"
    try:
        text = version_file.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError, UnicodeDecodeError):
        text = ""
    if text.strip():
        return text.strip()

    try:
        completed = subprocess.run(  # noqa: S603 — args are a literal list
            ["git", "describe", "--tags", "--always"],
            cwd=str(harness_dir),
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return _FALLBACK_HARNESS_VERSION
    candidate = completed.stdout.strip()
    return candidate or _FALLBACK_HARNESS_VERSION


class RunLimits(BaseModel):
    """Configurable caps guarding against resource exhaustion (SECURITY.md).

    ``task_timeout_s`` is enforced today via :func:`run_with_limits`.
    ``token_budget`` is a hook point that ``run_task`` will consult once the
    model client is wired (a prerequisite for ADR-0004); it is **not** counted
    yet, only plumbed through so the budget is observable in abort events.
    """

    task_timeout_s: float | None = Field(
        default=DEFAULT_TASK_TIMEOUT_S,
        description="Wall-clock seconds allowed for a single task before abort. "
        "None disables the wall-clock cap.",
    )
    token_budget: int | None = Field(
        default=None,
        description="Total tokens permitted per evolution cycle (hook point; "
        "enforced when the model client lands).",
    )


def run_limits_from_env(env: dict[str, str] | None = None) -> RunLimits:
    """Build :class:`RunLimits` from environment variables.

    Recognized keys (consistent with ``.env.example``):

    - ``FOUNDRY_TASK_TIMEOUT``: seconds; ``<= 0`` disables the wall-clock cap.
    - ``FOUNDRY_TOKEN_BUDGET``: integer token cap; empty/absent leaves it
      unset (hook point only).
    """
    source = env if env is not None else os.environ

    task_timeout_s: float | None
    timeout_raw = source.get("FOUNDRY_TASK_TIMEOUT", "")
    if timeout_raw == "":
        task_timeout_s = DEFAULT_TASK_TIMEOUT_S
    else:
        value = float(timeout_raw)
        task_timeout_s = None if value <= 0 else value

    token_budget: int | None
    budget_raw = source.get("FOUNDRY_TOKEN_BUDGET", "")
    token_budget = None if budget_raw == "" else int(budget_raw)

    return RunLimits(task_timeout_s=task_timeout_s, token_budget=token_budget)


async def run_with_limits(
    awaitable: Awaitable[Any],
    log: TraceLogger,
    session_id: str,
    limits: RunLimits,
) -> Any:
    """Await ``awaitable`` under the wall-clock cap in ``limits``.

    On timeout, record a ``task_aborted`` trace event (reason ``wall_clock``)
    carrying the exceeded cap, then re-raise :class:`asyncio.TimeoutError` so
    the caller observes the abort. This is the SECURITY.md "Runaway
    detection" guardrail: a degenerate harness edit that loops unbounded is
    aborted before it can exhaust resources.
    """
    if limits.task_timeout_s is None:
        return await awaitable
    try:
        return await asyncio.wait_for(awaitable, timeout=limits.task_timeout_s)
    except asyncio.TimeoutError:
        log.record(
            session_id,
            kind="task_aborted",
            payload={
                "reason": "wall_clock",
                "timeout_s": limits.task_timeout_s,
                "token_budget": limits.token_budget,
            },
        )
        raise


async def run_task(task: str, harness_dir: Path, log: TraceLogger, session_id: str) -> None:
    raise NotImplementedError(
        "Phase 1 wiring: instantiate your OpenCode client here, "
        "fan tool calls through harness.hooks.get_registry(), "
        "and stream events into the TraceLogger."
    )


def main(run_task_fn: Callable[..., Awaitable[None]] | None = None) -> None:
    """Entry point for the FoundryX execution runner.

    A single task session is opened, a ``task_received`` event is recorded,
    the task is awaited under :class:`RunLimits`, and a *terminal* event is
    recorded before the session closes:

    - ``task_completed`` with ``duration_ms`` on success.
    - ``task_failed`` with ``error_type``, ``message``, and ``duration_ms``
      on any :class:`Exception` (including the ``TimeoutError`` re-raised by
      :func:`run_with_limits`); the exception is then re-raised so the
      caller still observes it. Recording the outcome satisfies ADR-0007
      (traces carry observable behavior) and gives the Phase 2 Digester a
      terminal status to reason about. Stack frames are deliberately omitted
      to keep the trace compact.

    ``run_task_fn`` defaults to the module-level :func:`run_task`; tests may
    inject a stub to drive ``main`` without the real model client.
    """
    task = run_task_fn if run_task_fn is not None else run_task

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
    harness_version = resolve_harness_version(harness_dir)
    limits = run_limits_from_env()

    with logger.session(harness_version=harness_version) as session_id:
        logger.record(session_id, kind="task_received", payload={"prompt": args.task})
        start = time.monotonic()
        try:
            asyncio.run(
                run_with_limits(
                    task(args.task, harness_dir, logger, session_id),
                    logger,
                    session_id,
                    limits,
                )
            )
        except Exception as exc:
            duration_ms = int((time.monotonic() - start) * 1000)
            logger.record(
                session_id,
                kind="task_failed",
                payload={
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                    "duration_ms": duration_ms,
                },
            )
            raise
        duration_ms = int((time.monotonic() - start) * 1000)
        logger.record(
            session_id,
            kind="task_completed",
            payload={"duration_ms": duration_ms},
        )


if __name__ == "__main__":
    main()
