from __future__ import annotations

import argparse
import asyncio
import fnmatch
import hashlib
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from pydantic import BaseModel, Field

from foundry_x.execution.harness_layout import (
    HarnessValidationError,
    validate as validate_harness_layout,
)
from foundry_x.execution.model_adapter import (
    ModelAdapter,
    ModelMessage,
    ModelResponse,
    ModelRetryEvent,
    ModelToolCall,
    ModelUsage,
    OpenAICompatibleAdapter,
    ToolCallFunction,
    ToolDefinition,
    ToolFunctionSchema,
    resolve_adapter_max_retries,
)
from foundry_x.trace.logger import TraceLogger

# Default wall-clock cap (seconds) for a single task. Generous enough for a
# real model turn, small enough to abort a runaway loop before it exhausts
# the GPU/wallet. Mandated by docs/SECURITY.md "Runaway detection".
DEFAULT_TASK_TIMEOUT_S: float = 600.0


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

# Env-var names consulted for model identity (issue #12). ``FOUNDRY_MODEL_ID``
# is the explicit, foundry-owned override; the other two already exist in
# ``.env.example`` and let a local-first deployment self-describe without an
# extra setting (PHILOSOPHY.md §5).
_MODEL_ID_ENV = "FOUNDRY_MODEL_ID"
_LLAMACPP_MODEL_PATH_ENV = "LLAMACPP_MODEL_PATH"
_OPENCODE_SERVER_URL_ENV = "OPENCODE_SERVER_URL"
_LLAMACPP_HOST_ENV = "LLAMACPP_HOST"
_FOUNDRY_MODEL_API_KEY_ENV = "FOUNDRY_MODEL_API_KEY"
_OPENAI_API_KEY_ENV = "OPENAI_API_KEY"
_FALLBACK_REQUEST_MODEL = "foundry-local"

# Trace-backend selection (issue #13). ``.env.example`` documents
# ``FOUNDRY_TRACE_BACKEND`` as the way to switch the trace store between the
# default SQLite database and the JSONL export format (ADR-0003). Keeping the
# supported set in one place lets the runner validate the value up front
# rather than silently falling through to a no-op backend, honoring
# AGENTS.md §2 ("Never silently swallow an exception").
_TRACE_BACKEND_ENV = "FOUNDRY_TRACE_BACKEND"
_SUPPORTED_TRACE_BACKENDS: frozenset[str] = frozenset({"sqlite", "jsonl"})
_DEFAULT_TRACE_BACKEND: str = "sqlite"

# Agent-loop step cap (issue #89, ADR-0010). ``FOUNDRY_MAX_AGENT_STEPS`` is the
# foundry-owned override; an empty value or a non-positive integer falls back
# to the literal below. The cap exists for two reasons: (a) PRD §5 /
# SECURITY.md "Runaway detection" — a degenerate harness edit that loops
# unbounded must not exhaust the GPU/wallet, and (b) the Phase 2 Digester
# needs a finite ``steps`` field on the outcome event so a session is
# attributable rather than open-ended.
_MAX_AGENT_STEPS_ENV = "FOUNDRY_MAX_AGENT_STEPS"
_DEFAULT_MAX_AGENT_STEPS: int = 16

# Per-request httpx timeout (issue #201). ``FOUNDRY_REQUEST_TIMEOUT_S`` caps a
# single chat-completion HTTP round-trip so a stuck model endpoint cannot hang
# the agent loop indefinitely. The wall-clock ``RunLimits`` cap guards the
# whole session; this guards each step (SECURITY.md "Runaway detection"). A
# non-positive value falls back to the default; a non-numeric or non-finite
# value raises at process start (AGENTS.md §2 — fail fast, do not silently
# swallow).
_REQUEST_TIMEOUT_ENV = "FOUNDRY_REQUEST_TIMEOUT_S"
_DEFAULT_REQUEST_TIMEOUT_S: float = 30.0

_WORKSPACE_ROOT_ENV = "FOUNDRY_WORKSPACE_ROOT"


def _resolve_workspace_root(env: dict[str, str] | None = None) -> Path:
    """Resolve the agent workspace root for file-operation skill executors.

    ``FOUNDRY_WORKSPACE_ROOT`` is the explicit override. When absent or
    empty the current working directory is used as the workspace root.
    The returned path is always absolute.
    """
    source = env if env is not None else os.environ
    raw = source.get(_WORKSPACE_ROOT_ENV, "").strip()
    if raw:
        return Path(raw).resolve()
    return Path.cwd()


# Skill-executor protocol (issue #89, ADR-0010). A callable that maps a skill
# name + already-decoded arguments dict to the tool result the model will see.
# ``Awaitable[Any]`` lets callers return arbitrary JSON-shaped data; the runner
# serializes the result with ``json.dumps`` before injecting it back into the
# model's tool channel. The default executor is a stub that acknowledges the
# call (issue #104 declared the JSON contract; the actual subprocess.run-
# backed hook lands in a follow-up PR so the Critic can gate it independently).
SkillExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]


class ModelResponseChunkEvent(BaseModel):
    """Trace event payload emitted per SSE delta during streaming (#199).

    Emitted for each chunk from ``adapter.stream()`` so a KPI consumer can
    split "model latency" (time-to-first-token) from "network latency"
    (inter-chunk gaps) without waiting for the terminal ``model_response``.
    """

    step: int = Field(description="Agent-loop step index (0-based).")
    delta_index: int = Field(description="Zero-based chunk ordinal within this step.")
    content_so_far: str = Field(
        description="Concatenated assistant content up to and including this delta.",
    )
    chunk_duration_ms: int = Field(
        description="Wall-clock milliseconds since the previous chunk "
        "(or stream start for delta_index 0).",
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
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get(_WORKSPACE_ROOT_ENV),
        help="Root directory for file-operation skill executors. "
        "Defaults to the current working directory.",
    )
    parser.add_argument(
        "--model-id",
        default=None,
        help="Model identifier (e.g. Q5_K_M). Overrides FOUNDRY_MODEL_ID env var. "
        "Stored in the trace session for quantization-level KPI attribution (issue #361).",
    )
    args = parser.parse_args()

    harness_dir = Path(args.harness_dir).resolve()
    try:
        validate_harness_layout(harness_dir)
    except HarnessValidationError as exc:
        joined = ", ".join(exc.missing)
        print(
            f"error: harness directory {exc.harness_dir} is missing required entries: {joined}",
            file=sys.stderr,
        )
        sys.exit(2)
    if str(harness_dir) not in sys.path:
        sys.path.insert(0, str(harness_dir))

    logger = TraceLogger(args.trace_path)
    harness_version = "0.1.0"
    limits = run_limits_from_env()

    logger = TraceLogger(args.trace_path, backend=resolve_trace_backend())
    harness_version = resolve_harness_version(harness_dir)
    model_id = args.model_id if args.model_id is not None else resolve_model_id()
    limits = run_limits_from_env()

    with logger.session(harness_version=harness_version, model_id=model_id) as session_id:
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
