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
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

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

if TYPE_CHECKING:
    from harness.hooks.base import HookRegistry, ToolCall, ToolResult
else:
    # Imported lazily inside ``_resolve_hook_registry`` and ``run_task``
    # so that importing this module does not require the harness package
    # to be on ``sys.path`` (the AGENTS.md §7 self-reference rule the
    # foundry must respect in import direction).
    ToolCall = ToolResult = HookRegistry = object  # type: ignore[assignment]

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


def _resolve_workspace_root(env: Mapping[str, str] | None = None) -> Path:
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


@dataclass
class _StreamingToolCallAccumulator:
    """Accumulates partial tool-call deltas indexed by stream position."""

    id: str | None = None
    type: str | None = None
    name: str | None = None
    arguments: str = ""


@dataclass
class _ParsedToolArguments:
    """Decoded tool-call arguments plus any parse failure (issue #261).

    ``arguments`` is the decoded dict on success or ``{}`` when ``error``
    is set, preserving ADR-0010's resilience contract (the loop proceeds
    with empty arguments rather than aborting on a malformed wire value).
    ``error`` carries the human-readable failure reason so the runner can
    stamp a ``tool_argument_parse_error`` trace event for Digester/operator
    visibility; it is ``None`` on a clean parse so no spurious event fires
    for a legitimate no-argument call.
    """

    arguments: dict[str, Any]
    error: str | None


def resolve_trace_backend(env: Mapping[str, str] | None = None) -> str:
    """Resolve the trace backend from ``FOUNDRY_TRACE_BACKEND`` (issue #13).

    The value is lower-cased and whitespace-stripped before comparison so a
    value such as ``"JSONL"`` or ``" sqlite "`` from a hand-edited ``.env``
    is accepted. An empty/absent value yields the :data:`_DEFAULT_TRACE_BACKEND`
    (``sqlite``), matching ``.env.example``. An unrecognized value raises
    :class:`ValueError` naming the valid options — failing fast at startup is
    preferable to a silent fall-through that writes nothing and leaves the
    engineer debugging why no trace appeared (AGENTS.md §2).
    """
    source = env if env is not None else os.environ
    raw = source.get(_TRACE_BACKEND_ENV, "").strip().lower()
    backend = raw or _DEFAULT_TRACE_BACKEND
    if backend not in _SUPPORTED_TRACE_BACKENDS:
        valid = ", ".join(sorted(_SUPPORTED_TRACE_BACKENDS))
        raise ValueError(
            f"Unsupported FOUNDRY_TRACE_BACKEND={backend!r}; valid options are: {valid}"
        )
    return backend


# Trace-backend selection (issue #13). ``.env.example`` documents
# ``FOUNDRY_TRACE_BACKEND`` as the way to switch the trace store between the
# default SQLite database and the JSONL export format (ADR-0003). Keeping the
# supported set in one place lets the runner validate the value up front
# rather than silently falling through to a no-op backend, honoring
# AGENTS.md §2 ("Never silently swallow an exception").
_TRACE_BACKEND_ENV = "FOUNDRY_TRACE_BACKEND"
_SUPPORTED_TRACE_BACKENDS: frozenset[str] = frozenset({"sqlite", "jsonl"})
_DEFAULT_TRACE_BACKEND: str = "sqlite"


def resolve_model_id(env: Mapping[str, str] | None = None) -> str | None:
    """Resolve the model identity to stamp into the trace session (issue #12).

    Resolution order, highest precedence first:

    1. ``FOUNDRY_MODEL_ID`` — an explicit, foundry-owned override. Whitespace
       is trimmed; an empty value falls through.
    2. ``LLAMACPP_MODEL_PATH`` basename — for a local-first llama.cpp run the
       model file name (e.g. ``codellama-7b.Q5_K_M.gguf``) is a stable,
       human-readable identity with no extra configuration.
    3. ``OPENCODE_SERVER_URL`` host — the network address of an OpenAI-
       compatible endpoint (e.g. ``127.0.0.1``), stripped of scheme/port/path.
    4. ``None`` — provenance is unknown rather than fabricated.

    The value is purely informational provenance supporting ADR-0007 and the
    improvement-rate KPI's before/after comparability (PRD §4): so that a
    change in *success rate* can be separated from a change in *model*. No
    value is ever invented; absent evidence the field stays ``NULL``.
    """
    source = env if env is not None else os.environ

    explicit = source.get(_MODEL_ID_ENV, "").strip()
    if explicit:
        return explicit

    model_path = source.get(_LLAMACPP_MODEL_PATH_ENV, "").strip()
    if model_path:
        basename = os.path.basename(model_path)
        if basename:
            return basename

    server_url = source.get(_OPENCODE_SERVER_URL_ENV, "").strip()
    if server_url:
        host = urlsplit(server_url).hostname
        if host:
            return host

    return None


def _resolve_model_request_name(env: Mapping[str, str] | None = None) -> str:
    """Resolve the model name sent to an OpenAI-compatible endpoint.

    ``resolve_model_id`` may fall back to the endpoint host for provenance,
    but the request body's ``model`` field should be a model-like value. If a
    caller has not configured one, local OpenAI-compatible servers generally
    accept an arbitrary placeholder, so the fallback stays explicit and stable.
    """
    source = env if env is not None else os.environ
    explicit = source.get(_MODEL_ID_ENV, "").strip()
    if explicit:
        return explicit
    model_path = source.get(_LLAMACPP_MODEL_PATH_ENV, "").strip()
    if model_path:
        basename = os.path.basename(model_path)
        if basename:
            return basename
    return _FALLBACK_REQUEST_MODEL


def build_model_adapter(env: Mapping[str, str] | None = None) -> OpenAICompatibleAdapter:
    """Create the default OpenAI-compatible ModelAdapter from environment.

    ``OPENCODE_SERVER_URL`` remains the primary endpoint knob from
    ``.env.example``; ``LLAMACPP_HOST`` is accepted as a local-first fallback.
    API keys are read only from environment and never persisted here.
    ``FOUNDRY_REQUEST_TIMEOUT_S`` is plumbed into the owned
    :class:`httpx.AsyncClient` as the per-request cap (issue #201).
    """
    source = env if env is not None else os.environ
    base_url = (
        source.get(_OPENCODE_SERVER_URL_ENV, "").strip()
        or source.get(_LLAMACPP_HOST_ENV, "").strip()
    )
    if not base_url:
        raise ValueError(
            "Set OPENCODE_SERVER_URL or LLAMACPP_HOST to an OpenAI-compatible endpoint"
        )
    api_key = (
        source.get(_FOUNDRY_MODEL_API_KEY_ENV, "").strip()
        or source.get(_OPENAI_API_KEY_ENV, "").strip()
    )
    return OpenAICompatibleAdapter(
        base_url=base_url,
        model=_resolve_model_request_name(source),
        api_key=api_key or None,
        timeout=_resolve_request_timeout(source),
        max_retries=resolve_adapter_max_retries(source),
    )


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


def run_limits_from_env(env: Mapping[str, str] | None = None) -> RunLimits:
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


def _resolve_max_steps(env: Mapping[str, str] | None = None) -> int:
    """Resolve the per-task step cap (issue #89, ADR-0010).

    ``FOUNDRY_MAX_AGENT_STEPS`` is the foundry-owned override. An empty /
    missing value or a non-positive integer falls back to
    :data:`_DEFAULT_MAX_AGENT_STEPS`. Errors (non-integer values) propagate so
    a typo in ``.env`` surfaces immediately rather than silently disabling the
    cap (AGENTS.md §2).
    """
    source = env if env is not None else os.environ
    raw = source.get(_MAX_AGENT_STEPS_ENV, "").strip()
    if not raw:
        return _DEFAULT_MAX_AGENT_STEPS
    value = int(raw)
    return value if value > 0 else _DEFAULT_MAX_AGENT_STEPS


def _resolve_request_timeout(env: Mapping[str, str] | None = None) -> float:
    """Resolve the per-request httpx timeout in seconds (issue #201).

    ``FOUNDRY_REQUEST_TIMEOUT_S`` caps a single chat-completion HTTP
    round-trip so a model endpoint that accepts the connection but never
    completes the response cannot hang the agent loop. The wall-clock
    :class:`RunLimits` cap guards the whole session; this guards each step.

    Resolution rules:

    - Empty / absent → :data:`_DEFAULT_REQUEST_TIMEOUT_S`.
    - Non-positive (``<= 0``) → :data:`_DEFAULT_REQUEST_TIMEOUT_S` (the cap
      cannot be disabled per-request; a stuck step would otherwise hang the
      session until the wall-clock cap fires).
    - Non-numeric or non-finite (``inf`` / ``nan``) → :class:`ValueError` so a
      typo in ``.env`` surfaces at process start rather than silently
      disabling the guard (AGENTS.md §2).
    """
    source = env if env is not None else os.environ
    raw = source.get(_REQUEST_TIMEOUT_ENV, "").strip()
    if not raw:
        return _DEFAULT_REQUEST_TIMEOUT_S
    value = float(raw)
    if not math.isfinite(value):
        raise ValueError(f"FOUNDRY_REQUEST_TIMEOUT_S={raw!r} is not a finite number")
    return value if value > 0 else _DEFAULT_REQUEST_TIMEOUT_S


def _load_tool_definitions(skills_dir: Path) -> list[ToolDefinition]:
    """Build the ``ToolDefinition`` surface from ``harness/skills/*.json``.

    Maps the harness's per-skill JSON Schema (issue #104, #105) to the
    OpenAI-compatible ``tools=`` array the :class:`ModelAdapter` protocol
    expects. The skill's ``name`` / ``description`` / ``input_schema`` keys
    flow directly into :class:`ToolFunctionSchema`; the output schema is
    intentionally not transmitted (the wire format has no slot for it).

    A missing ``skills/`` directory is treated as an empty surface: a freshly
    bootstrapped harness may legitimately have no skills yet, and the loop
    still works against a model that emits only final-answer messages. Any
    JSON parse error is re-raised — a malformed skill file is a Critic
    sandbox failure (ADR-0004, ``harness/scripts/load_check.py``) and the
    runner should not paper over it.
    """
    if not skills_dir.is_dir():
        return []
    definitions: list[ToolDefinition] = []
    for path in sorted(skills_dir.glob("*.json")):
        doc = json.loads(path.read_text(encoding="utf-8"))
        name = doc.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path}: skill 'name' must be a non-empty string")
        description = doc.get("description") or None
        parameters = doc.get("input_schema") or {}
        if not isinstance(parameters, dict):
            raise ValueError(f"{path}: skill 'input_schema' must be a JSON object")
        definitions.append(
            ToolDefinition(
                type="function",
                function=ToolFunctionSchema(
                    name=name,
                    description=description,
                    parameters=parameters,
                ),
            )
        )
    return definitions


def _resolve_hook_registry(log: TraceLogger, session_id: str) -> Any | None:
    """Return the harness hook registry, or ``None`` if no harness is wired.

    The foundry's AGENTS.md §7 self-reference rule forbids depending on the
    harness package from this module's import side; instead the registry is
    looked up lazily so a test that imports ``runner`` without a harness
    checkout (or before ``main()`` has inserted ``harness_dir`` into
    ``sys.path``) sees ``None`` and silently skips hook fan-out. When the
    harness IS importable the prompt-input firewall
    (``harness/hooks/injection_firewall.py``) is already registered there
    via ``harness/hooks/__init__``, so SECURITY.md "Prompt-input firewall"
    runs by default.

    A *missing* harness (``ImportError``) is a legitimate degraded state
    and returns ``None`` without comment. A harness that IS importable but
    whose :func:`get_registry` raises is a different matter: every hook —
    including the security-critical ``InjectionFirewallHook`` — would be
    silently disabled for the entire session with no trace signal
    (AGENTS.md §2 — never silently swallow an exception). Issue #260
    requires that case to record a ``hook_registry_error`` trace event
    carrying ``error_type`` and ``message`` so the Digester and the
    operator can observe that the firewall layer is off, while still
    completing the session in degraded mode (``registry is None``).
    """
    try:
        from harness.hooks import get_registry
    except ImportError:
        return None
    try:
        return get_registry()
    except Exception as exc:
        log.record(
            session_id,
            kind="hook_registry_error",
            payload={
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )
        return None


def _import_hook_types() -> tuple[type, type]:
    """Import :class:`ToolCall` and :class:`ToolResult` from the harness.

    Resolved lazily (issue #89, ADR-0010) so the runner can be imported
    without the harness package on ``sys.path``. Calling code should treat
    the returned classes as a contract — they implement the
    ``Hook`` protocol declared in ``harness/hooks/base.py``.
    """
    from harness.hooks.base import ToolCall, ToolResult

    return ToolCall, ToolResult


def _truncate_at_newline(data: bytes, max_bytes: int) -> tuple[bytes, bool]:
    """Truncate bytes to max_bytes at the largest '\\n' boundary <= max_bytes.

    Returns (truncated_data, was_truncated). If data fits in max_bytes,
    returns (data, False).
    """
    if len(data) <= max_bytes:
        return data, False
    candidate = data[:max_bytes]
    last_newline = candidate.rfind(b"\n")
    if last_newline >= 0:
        return candidate[: last_newline + 1], True
    return candidate, True


async def _bash_skill_executor(
    name: str, arguments: dict[str, Any], *, workspace_dir: Path | None = None
) -> dict[str, Any]:
    """Subprocess-backed bash skill executor (issue #258).

    Executes a single shell command via ``subprocess.run`` with ``shell=False``.
    Per ``harness/skills/bash.json``:
    - ``command`` is split with ``shlex.split`` and passed as argv (no shell)
    - ``cwd`` defaults to ``workspace_dir`` if not provided
    - ``timeout_seconds`` defaults to 30; on timeout exit_code=-1 and truncated=True
    - ``max_output_bytes`` defaults to 32768; output is truncated at newline boundary
    """
    command: str = arguments.get("command", "")
    cwd_arg: str | None = arguments.get("cwd")
    timeout_seconds: int = arguments.get("timeout_seconds", 30)
    max_output_bytes: int = arguments.get("max_output_bytes", 32768)

    cwd: Path | None = None
    if cwd_arg:
        cwd = Path(cwd_arg)
    elif workspace_dir:
        cwd = workspace_dir

    argv = shlex.split(command)

    try:
        proc = await asyncio.to_thread(
            subprocess.run,
            argv,
            cwd=cwd,
            capture_output=True,
            timeout=timeout_seconds,
            shell=False,
        )
        stdout_raw = proc.stdout or b""
        stderr_raw = proc.stderr or b""

        stdout_truncated, stdout_was_truncated = _truncate_at_newline(stdout_raw, max_output_bytes)
        stderr_truncated, stderr_was_truncated = _truncate_at_newline(stderr_raw, max_output_bytes)

        return {
            "stdout": stdout_truncated.decode("utf-8", errors="replace"),
            "stderr": stderr_truncated.decode("utf-8", errors="replace"),
            "exit_code": proc.returncode,
            "truncated": stdout_was_truncated or stderr_was_truncated,
        }
    except subprocess.TimeoutExpired as exc:
        stdout_raw = exc.stdout or b""
        stderr_raw = exc.stderr or b""

        stdout_truncated, _ = _truncate_at_newline(stdout_raw, max_output_bytes)
        stderr_truncated, _ = _truncate_at_newline(stderr_raw, max_output_bytes)

        return {
            "stdout": stdout_truncated.decode("utf-8", errors="replace"),
            "stderr": stderr_truncated.decode("utf-8", errors="replace"),
            "exit_code": -1,
            "truncated": True,
        }


async def _exec_read_file(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Execute the ``read_file`` skill (issue #416).

    Reads a file with path-escape prevention (``_resolve_path``) and
    returns bounded content with SHA-256, truncation flag, and byte counts
    per the ``harness/skills/read_file.json`` contract.
    """
    path_str = arguments.get("path", "")
    max_bytes = int(arguments.get("max_bytes", 32768))
    max_lines = arguments.get("max_lines")
    offset = int(arguments.get("offset", 0))

    if not path_str:
        return {
            "content": "",
            "sha256": "",
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": 0,
            "error": "path is required",
        }

    try:
        resolved = _resolve_path(path_str, workspace_root)
    except ValueError as exc:
        return {
            "content": "",
            "sha256": "",
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": 0,
            "error": str(exc),
        }

    if not resolved.exists():
        return {
            "content": "",
            "sha256": "",
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": 0,
            "error": f"file not found: {path_str}",
        }

    if not resolved.is_file():
        return {
            "content": "",
            "sha256": "",
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": 0,
            "error": f"not a file: {path_str}",
        }

    try:
        raw_bytes = resolved.read_bytes()
    except OSError as exc:
        return {
            "content": "",
            "sha256": "",
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": 0,
            "error": str(exc),
        }

    bytes_total = len(raw_bytes)

    if offset >= bytes_total:
        return {
            "content": "",
            "sha256": hashlib.sha256(b"").hexdigest(),
            "truncated": False,
            "bytes_returned": 0,
            "bytes_total": bytes_total,
            "error": None,
        }

    remaining = raw_bytes[offset:]
    content_bytes = remaining[:max_bytes]

    if max_lines is not None:
        text_so_far = content_bytes.decode("utf-8", errors="replace")
        lines = text_so_far.splitlines(keepends=True)
        if len(lines) > max_lines:
            truncated_at_line = "".join(lines[:max_lines])
            content_bytes = truncated_at_line.encode("utf-8")
            was_truncated = True
        else:
            was_truncated = False
    else:
        was_truncated = len(remaining) > max_bytes

    if was_truncated:
        content_bytes_truncated, _ = _truncate_at_newline(content_bytes, max_bytes)
        content_bytes = content_bytes_truncated

    content_str = content_bytes.decode("utf-8", errors="replace")
    sha = hashlib.sha256(content_bytes).hexdigest()

    return {
        "content": content_str,
        "sha256": sha,
        "truncated": was_truncated,
        "bytes_returned": len(content_bytes),
        "bytes_total": bytes_total,
        "error": None,
    }


async def _exec_read_multiple_files(
    arguments: dict[str, Any], workspace_root: Path
) -> dict[str, Any]:
    """Execute the ``read_multiple_files`` skill (issue #416).

    Reads multiple files in sequence, each with path-escape prevention.
    Returns a list of per-file results plus an overall truncated flag.
    """
    paths: list[str] = arguments.get("paths", [])
    max_bytes = int(arguments.get("max_bytes", 32768))
    max_lines = arguments.get("max_lines")
    offset = int(arguments.get("offset", 0))

    if not paths:
        return {"results": [], "truncated": False, "error": "paths is required"}

    results: list[dict[str, Any]] = []
    any_truncated = False

    for path_str in paths:
        result = await _exec_read_file(
            {"path": path_str, "max_bytes": max_bytes, "max_lines": max_lines, "offset": offset},
            workspace_root,
        )
        results.append(result)
        if result.get("truncated"):
            any_truncated = True

    return {"results": results, "truncated": any_truncated, "error": None}


async def _default_skill_executor(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Default skill executor for unimplemented skills (issue #416).

    Returns an explicit error envelope so the agent can distinguish a missing
    skill from a successful execution that returned no content.
    """
    return {
        "error": f"skill {name!r} is not implemented",
        "skill": name,
        "echo": sorted(arguments.keys()),
    }


def _resolve_path(path: str, workspace_root: Path) -> Path:
    """Resolve ``path`` against ``workspace_root`` and check it does not escape.

    Absolute paths are resolved as-is and must be inside ``workspace_root``.
    Relative paths are joined with ``workspace_root`` first. Path escapes
    (``..`` segments or absolute paths pointing outside ``workspace_root``)
    raise ``ValueError``.
    """
    if Path(path).is_absolute():
        resolved = Path(path).resolve()
    else:
        resolved = (workspace_root / path).resolve()
    try:
        resolved.relative_to(workspace_root.resolve())
    except ValueError:
        raise ValueError(f"path {path!r} escapes workspace root") from None
    return resolved


async def _exec_list_dir(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Execute the ``list_dir`` skill (issue #259)."""
    path_str = arguments.get("path", ".")
    glob_pattern = arguments.get("glob")
    include_hidden = bool(arguments.get("include_hidden", False))
    max_entries = int(arguments.get("max_entries", 1000))

    try:
        resolved = _resolve_path(path_str, workspace_root)
    except ValueError as exc:
        return {"entries": [], "truncated": False, "error": str(exc)}

    if not resolved.exists() or not resolved.is_dir():
        return {"entries": [], "truncated": False}

    raw_entries: list[tuple[str, str, int]] = []
    try:
        scan_iter = os.scandir(resolved)
    except OSError:
        return {"entries": [], "truncated": False}

    with scan_iter:
        for entry in scan_iter:
            if not include_hidden and entry.name.startswith("."):
                continue
            if glob_pattern is not None and not fnmatch.fnmatch(entry.name, glob_pattern):
                continue
            if entry.is_symlink():
                kind = "symlink"
            elif entry.is_dir(follow_symlinks=False):
                kind = "dir"
            elif entry.is_file(follow_symlinks=False):
                kind = "file"
            else:
                kind = "other"
            size = 0 if kind != "file" else entry.stat(follow_symlinks=False).st_size
            raw_entries.append((entry.name, kind, size))

    raw_entries.sort(key=lambda triple: triple[0])
    truncated = len(raw_entries) > max_entries
    visible = raw_entries[:max_entries]

    return {
        "entries": [{"name": name, "kind": kind, "size": size} for name, kind, size in visible],
        "truncated": truncated,
    }


async def _exec_grep_search(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Execute the ``grep_search`` skill (issue #259)."""
    pattern_str = arguments.get("pattern", "")
    path_str = arguments.get("path", ".")
    glob_pattern = arguments.get("glob")
    include_hidden = bool(arguments.get("include_hidden", False))
    context_lines = int(arguments.get("context_lines", 0))
    max_matches = int(arguments.get("max_matches", 1000))
    max_file_bytes = int(arguments.get("max_file_bytes", 1048576))
    max_match_text_bytes = int(arguments.get("max_match_text_bytes", 4096))

    if len(pattern_str) > 4096:
        return {"matches": [], "truncated": False, "error": "pattern exceeds 4096 character limit"}

    for feature in ("(?P<", "(?=", "(?!", "(?<=", "(?<!", "(?#", "(?<!"):
        if feature in pattern_str:
            return {
                "matches": [],
                "truncated": False,
                "error": "pattern uses disallowed regex feature",
            }
    if "**" in pattern_str:
        return {"matches": [], "truncated": False, "error": "pattern uses disallowed regex feature"}

    try:
        compiled = re.compile(pattern_str, flags=re.DOTALL)
    except re.error as exc:
        return {"matches": [], "truncated": False, "error": f"invalid regex: {exc}"}

    try:
        resolved = _resolve_path(path_str, workspace_root)
    except ValueError as exc:
        return {"matches": [], "truncated": False, "error": str(exc)}

    if not resolved.exists() or not resolved.is_dir():
        return {"matches": [], "truncated": False}

    matches: list[dict[str, Any]] = []
    truncated = False

    try:
        sorted_files = sorted(resolved.rglob("*"), key=lambda p: (str(p.parent), p.name))
    except OSError:
        return {"matches": [], "truncated": False}

    for file_path in sorted_files:
        if not include_hidden and any(part.startswith(".") for part in file_path.parts):
            continue
        if not file_path.is_file():
            continue
        if glob_pattern is not None and not fnmatch.fnmatch(file_path.name, glob_pattern):
            continue

        rel_path = file_path.relative_to(resolved)
        file_str = rel_path.as_posix()

        try:
            file_size = file_path.stat().st_size
        except OSError:
            continue

        if file_size > max_file_bytes:
            matches.append(
                {"file": file_str, "line": 0, "text": "<skipped: file exceeds max_file_bytes>"}
            )
            if len(matches) >= max_matches:
                truncated = True
                break
            continue

        try:
            raw_bytes = file_path.read_bytes()
        except OSError:
            continue

        text = raw_bytes.decode("utf-8", errors="replace")
        lines = text.splitlines(keepends=True)

        for line_no, line_text in enumerate(lines, start=1):
            if compiled.search(line_text):
                context_start = max(0, line_no - context_lines - 1)
                context_end = min(len(lines), line_no + context_lines)
                context_slice = lines[context_start:context_end]
                joined = "".join(context_slice)
                if len(joined.encode("utf-8")) > max_match_text_bytes:
                    cut = joined.encode("utf-8")[:max_match_text_bytes]
                    null_pos = cut.rfind(b"\n")
                    if null_pos > 0:
                        joined = cut[:null_pos].decode("utf-8", errors="replace")
                    else:
                        joined = cut.decode("utf-8", errors="replace")
                    truncated = True
                matches.append({"file": file_str, "line": line_no, "text": joined.rstrip("\n")})
                if len(matches) >= max_matches:
                    truncated = True
                    break

        if truncated:
            break

    return {"matches": matches, "truncated": truncated}


async def _exec_edit_file(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Execute the ``edit_file`` skill (issue #259)."""
    path_str = arguments.get("path", "")
    old_string = arguments.get("old_string", "")
    new_string = arguments.get("new_string", "")

    if not path_str or not old_string:
        return {
            "path": path_str,
            "sha256": "",
            "replacements_made": 0,
            "error": "path and old_string are required",
        }

    try:
        resolved = _resolve_path(path_str, workspace_root)
    except ValueError as exc:
        return {"path": path_str, "sha256": "", "replacements_made": 0, "error": str(exc)}

    try:
        original = resolved.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {
            "path": path_str,
            "sha256": "",
            "replacements_made": 0,
            "error": f"file not found: {path_str}",
        }
    except OSError as exc:
        return {"path": path_str, "sha256": "", "replacements_made": 0, "error": str(exc)}

    if old_string not in original:
        sha = hashlib.sha256(original.encode("utf-8")).hexdigest()
        return {"path": str(resolved), "sha256": sha, "replacements_made": 0}

    new_content = original.replace(old_string, new_string)
    try:
        resolved.write_text(new_content, encoding="utf-8")
    except OSError as exc:
        return {"path": str(resolved), "sha256": "", "replacements_made": 0, "error": str(exc)}

    sha = hashlib.sha256(new_content.encode("utf-8")).hexdigest()
    replacements = original.count(old_string)
    return {"path": str(resolved), "sha256": sha, "replacements_made": replacements}


async def _exec_write_file(arguments: dict[str, Any], workspace_root: Path) -> dict[str, Any]:
    """Execute the ``write_file`` skill (issue #259)."""
    path_str = arguments.get("path", "")
    content = arguments.get("content", "")

    if not path_str:
        return {"path": "", "sha256": "", "bytes_written": 0, "error": "path is required"}

    try:
        resolved = _resolve_path(path_str, workspace_root)
    except ValueError as exc:
        return {"path": path_str, "sha256": "", "bytes_written": 0, "error": str(exc)}

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except OSError as exc:
        return {"path": str(resolved), "sha256": "", "bytes_written": 0, "error": str(exc)}

    sha = hashlib.sha256(content.encode("utf-8")).hexdigest()
    bytes_written = len(content.encode("utf-8"))
    return {"path": str(resolved), "sha256": sha, "bytes_written": bytes_written}


async def _file_operation_skill_executor(
    name: str, arguments: dict[str, Any], workspace_root: Path
) -> dict[str, Any]:
    """Dispatch to the correct file-operation skill executor (issue #259).

    Skills that are not file operations fall through to the default stub.
    """
    if name == "list_dir":
        return await _exec_list_dir(arguments, workspace_root)
    if name == "grep_search":
        return await _exec_grep_search(arguments, workspace_root)
    if name == "edit_file":
        return await _exec_edit_file(arguments, workspace_root)
    if name == "write_file":
        return await _exec_write_file(arguments, workspace_root)
    if name == "read_file":
        return await _exec_read_file(arguments, workspace_root)
    if name == "read_multiple_files":
        return await _exec_read_multiple_files(arguments, workspace_root)
    return await _default_skill_executor(name, arguments)


def _parse_tool_arguments(raw: str) -> _ParsedToolArguments:
    """Parse an OpenAI-compatible tool-call ``arguments`` JSON string.

    Some models emit ``""`` (no arguments) and a few emit partially-formed
    JSON; both are coerced to an empty dict so the agent loop can stamp the
    call into the trace without a model-side parser failure (which would
    abort the loop and leave the session without an ``outcome`` event).
    Non-dict JSON values are also collapsed to ``{}`` for the same reason:
    the loop cares about the *executor contract*, not the wire shape.

    When coercion occurs (issue #261), ``error`` is populated so the caller
    can emit a ``tool_argument_parse_error`` trace event — the resilience
    contract (proceed with empty arguments) is unchanged, only observability
    is added so the Digester can distinguish a correct no-arg call from
    garbage JSON. An empty-string argument is a legitimate no-arg call and
    therefore returns ``error=None``.
    """
    if not raw:
        return _ParsedToolArguments(arguments={}, error=None)
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError as exc:
        return _ParsedToolArguments(arguments={}, error=f"JSONDecodeError: {exc}")
    if not isinstance(decoded, dict):
        return _ParsedToolArguments(
            arguments={},
            error=f"expected JSON object, got {type(decoded).__name__}",
        )
    return _ParsedToolArguments(arguments=dict(decoded), error=None)


def _serialize_tool_result(result: Any) -> str:
    """Render a :class:`ToolResult` to JSON for re-injection into the prompt.

    ``output`` is the canonical channel the model reads; ``error`` is the
    human-review flag (carrying previews the injection firewall writes, etc.)
    and is included as a sibling key so the model can branch on failure
    without us hiding the flag. ``str`` is the safe type fallback so a custom
    executor that returns a non-JSON-able object still produces a wire-ready
    payload instead of crashing the next ``complete()`` call.
    """
    payload: dict[str, Any] = {"output": result.output}
    if result.error is not None:
        payload["error"] = result.error
    return json.dumps(payload, default=str)


def _assemble_streamed_response(
    content_parts: list[str],
    tool_call_acc: dict[int, _StreamingToolCallAccumulator],
    finish_reason: str | None,
    usage: ModelUsage | None = None,
) -> ModelResponse:
    """Reconstruct a :class:`ModelResponse` from accumulated streaming deltas.

    OpenAI-compatible streaming sends tool-call arguments as incremental
    fragments keyed by a delta ``index``; this function reassembles them
    into complete :class:`ModelToolCall` objects the agent loop can execute.
    Content deltas are concatenated verbatim. A delta set with neither
    ``id`` nor ``name`` is treated as noise (some servers emit placeholder
    indexes) and dropped rather than producing a half-formed call that
    would crash the executor.
    """
    tool_calls: list[ModelToolCall] = []
    for idx in sorted(tool_call_acc):
        acc = tool_call_acc[idx]
        if acc.id and acc.name:
            tool_calls.append(
                ModelToolCall(
                    id=acc.id,
                    type="function",
                    function=ToolCallFunction(
                        name=acc.name,
                        arguments=acc.arguments,
                    ),
                )
            )
    content = "".join(content_parts) or None
    message = ModelMessage(
        role="assistant",
        content=content,
        tool_calls=tool_calls or None,
    )
    return ModelResponse(
        message=message,
        tool_calls=tool_calls,
        finish_reason=finish_reason,
        usage=usage,
    )


async def _consume_model_stream(
    adapter: ModelAdapter,
    messages: list[ModelMessage],
    tools: list[ToolDefinition],
    log: TraceLogger,
    session_id: str,
    step: int,
) -> tuple[ModelResponse, int | None, int]:
    """Consume ``adapter.stream()``, emit per-chunk trace events (issue #199).

    For every SSE delta a ``model_response_chunk`` trace event is recorded
    carrying ``delta_index``, ``content_so_far``, and ``chunk_duration_ms``
    so a KPI consumer can observe model latency as it happens rather than
    only at the terminal ``model_response``.

    Returns ``(assembled_response, time_to_first_token_ms, chunk_count)``.
    ``time_to_first_token_ms`` is measured from stream start to the first
    delta carrying content or a tool-call fragment; it is ``None`` when the
    stream produced no payload deltas.
    """
    stream_start = time.monotonic()
    content_parts: list[str] = []
    tool_call_acc: dict[int, _StreamingToolCallAccumulator] = {}
    finish_reason: str | None = None
    usage: ModelUsage | None = None
    delta_index = 0
    ttft_ms: int | None = None
    prev_time = stream_start

    async for chunk in adapter.stream(messages=messages, tools=tools):
        now = time.monotonic()
        chunk_duration_ms = int((now - prev_time) * 1000)
        prev_time = now

        if ttft_ms is None and (chunk.content or chunk.tool_calls):
            ttft_ms = int((now - stream_start) * 1000)

        if chunk.content:
            content_parts.append(chunk.content)

        for tc_chunk in chunk.tool_calls:
            idx = tc_chunk.index if tc_chunk.index is not None else 0
            acc = tool_call_acc.setdefault(idx, _StreamingToolCallAccumulator())
            if tc_chunk.id:
                acc.id = tc_chunk.id
            if tc_chunk.type:
                acc.type = tc_chunk.type
            if tc_chunk.function:
                if tc_chunk.function.name:
                    acc.name = tc_chunk.function.name
                if tc_chunk.function.arguments:
                    acc.arguments += tc_chunk.function.arguments

        if chunk.finish_reason:
            finish_reason = chunk.finish_reason

        if chunk.usage is not None:
            usage = chunk.usage

        log.record(
            session_id,
            kind="model_response_chunk",
            payload=ModelResponseChunkEvent(
                step=step,
                delta_index=delta_index,
                content_so_far="".join(content_parts),
                chunk_duration_ms=chunk_duration_ms,
            ).model_dump(),
        )
        delta_index += 1

    response = _assemble_streamed_response(content_parts, tool_call_acc, finish_reason, usage)
    return response, ttft_ms, delta_index


async def run_task(
    task: str,
    harness_dir: Path,
    log: TraceLogger,
    session_id: str,
    model_adapter: ModelAdapter | None = None,
    *,
    skill_executor: SkillExecutor | None = None,
    limits: RunLimits | None = None,
    workspace_root: Path | None = None,
) -> None:
    """Drive one task through the asyncio agent loop (issue #89, ADR-0010).

    Reads ``harness/system_prompt.txt`` and the OpenAI-compatible tool surface
    declared in ``harness/skills/*.json``, exchanges turns with the
    ``model_adapter`` until the model emits a final assistant message, the
    ``max_steps`` cap is reached, the running token total exceeds
    ``limits.token_budget`` (issue #197), or the wall-clock cap fires, and
    records every step into the trace store:

    1. ``user_prompt`` — the task enters the agent conversation
       (the lifecycle ``task_received`` marker from ``main()`` is preserved).
    2. ``model_request`` / ``model_response`` — every chat completion round-trip;
       the ``model_response`` payload carries the latest ``ModelResponse.usage``
       (when reported) so a token counter is observable per turn.
    3. ``tool_call`` / ``tool_result`` — one pair per ``ToolCall`` the model
       emits, bracketed by ``HookRegistry.run_pre`` and ``HookRegistry.run_post``
       fan-out so the prompt-input firewall (SECURITY.md) and future hooks
       observe every step.
    4. ``outcome`` — terminal event with ``status``, ``reason``, ``steps``, and
       a running ``tokens_total`` so the Phase 2 Digester can attribute success
       vs. truncation vs. failure vs. budget exhaustion, and the KPI consumer
       (PRD §5) can read the budget it actually used.

    The tool surface is data-driven: a skill lands as soon as its JSON file
    does (issue #104, #105). Skill execution is delegated to ``skill_executor``
    (default: ``_default_skill_executor`` — a stub that acknowledges the call);
    the ``bash`` skill is handled by ``_bash_skill_executor`` which uses
    ``subprocess.run`` with ``shell=False`` (issue #258).

    Token-budget enforcement (issue #197): the loop accumulates
    ``response.usage.total_tokens`` across steps. When the running total
    exceeds ``limits.token_budget`` after a ``model_response`` is recorded,
    the loop emits a ``task_aborted`` event with ``reason="token_budget"``
    and terminates with ``outcome.status="failed"``,
    ``outcome.reason="token_budget"``. The token-budget check lives in
    ``run_task`` (not :func:`run_with_limits`) because ``run_task`` owns the
    running counter; :func:`run_with_limits` continues to own the wall-clock
    cap (matches the existing wall_clock + ``task_aborted`` pairing in
    SECURITY.md "Runaway detection").

    Token-budget enforcement (issue #197): the loop accumulates
    ``response.usage.total_tokens`` across steps. When the running total
    exceeds ``limits.token_budget`` after a ``model_response`` is recorded,
    the loop emits a ``task_aborted`` event with ``reason="token_budget"``
    and terminates with ``outcome.status="failed"``,
    ``outcome.reason="token_budget"``. The token-budget check lives in
    ``run_task`` (not :func:`run_with_limits`) because ``run_task`` owns the
    running counter; :func:`run_with_limits` continues to own the wall-clock
    cap (matches the existing wall_clock + ``task_aborted`` pairing in
    SECURITY.md "Runaway detection").

    On any model error the loop records a ``model_error`` event, sets
    ``outcome.status="failed"`` and ``outcome.reason="model_error"``, and
    re-raises so ``main()`` can append the ``task_failed`` terminal marker.
    """
    system_prompt_path = harness_dir / "system_prompt.txt"
    system_prompt = system_prompt_path.read_text(encoding="utf-8")
    tool_definitions = _load_tool_definitions(harness_dir / "skills")

    messages: list[ModelMessage] = [
        ModelMessage(role="system", content=system_prompt),
        ModelMessage(role="user", content=task),
    ]
    created_adapter = model_adapter is None
    adapter = model_adapter or build_model_adapter()

    # Wire retry trace events (issue #200). Only `OpenAICompatibleAdapter`
    # has retry logic; injected fakes / stubs are left untouched.
    if isinstance(adapter, OpenAICompatibleAdapter):

        def _on_retry(event: ModelRetryEvent, _sid: str = session_id) -> None:
            log.record(
                _sid,
                kind="model_retry",
                payload=event.model_dump(),
            )

        adapter.on_retry = _on_retry

    registry = _resolve_hook_registry(log, session_id)
    hook_call_cls, hook_result_cls = _import_hook_types()

    resolved_workspace_root = (
        workspace_root if workspace_root is not None else _resolve_workspace_root()
    )

    async def _execute_skill(name: str, arguments: dict[str, Any]) -> Any:
        if skill_executor is not None:
            return await skill_executor(name, arguments)
        if name == "bash":
            return await _bash_skill_executor(name, arguments, workspace_dir=workspace_root)
        if name in ("list_dir", "grep_search", "edit_file", "write_file"):
            return await _file_operation_skill_executor(name, arguments, resolved_workspace_root)
        if name in ("read_file", "read_multiple_files"):
            return await _file_operation_skill_executor(name, arguments, resolved_workspace_root)
        return await _default_skill_executor(name, arguments)

    max_steps = _resolve_max_steps()

    async def _execute_skill(name: str, arguments: dict[str, Any]) -> Any:
        if skill_executor is not None:
            return await skill_executor(name, arguments)
        if name == "bash":
            return await _bash_skill_executor(name, arguments, workspace_dir=workspace_root)
        if name in ("list_dir", "grep_search", "edit_file", "write_file"):
            return await _file_operation_skill_executor(name, arguments, resolved_workspace_root)
        return await _default_skill_executor(name, arguments)

    max_steps = _resolve_max_steps()

    log.record(
        session_id,
        kind="user_prompt",
        payload={"content": task, "tool_count": len(tool_definitions)},
    )

    outcome_status = "success"
    outcome_reason = "final_answer"
    outcome_steps = 0
    turn_ttfts: list[int] = []
    tokens_used = 0
    token_budget = limits.token_budget if limits is not None else None

    try:
        for step in range(max_steps):
            outcome_steps = step + 1
            log.record(
                session_id,
                kind="model_request",
                payload={
                    "step": step,
                    "message_count": len(messages),
                    "tool_count": len(tool_definitions),
                },
            )
            try:
                response, ttft_ms, chunk_count = await _consume_model_stream(
                    adapter, messages, tool_definitions, log, session_id, step
                )
            except Exception as exc:
                outcome_status = "failed"
                outcome_reason = "model_error"
                log.record(
                    session_id,
                    kind="model_error",
                    payload={
                        "step": step,
                        "error_type": type(exc).__name__,
                        "message": str(exc),
                    },
                )
                raise

            if ttft_ms is not None:
                turn_ttfts.append(ttft_ms)
            step_tokens = response.usage.total_tokens if response.usage is not None else 0
            tokens_used += step_tokens

            log.record(
                session_id,
                kind="model_response",
                payload={
                    "step": step,
                    "finish_reason": response.finish_reason,
                    "message": response.message.model_dump(mode="json", exclude_none=True),
                    "tool_calls": [call.model_dump(mode="json") for call in response.tool_calls],
                    "time_to_first_token_ms": ttft_ms,
                    "chunk_count": chunk_count,
                    "token_usage": response.usage.model_dump(mode="json")
                    if response.usage is not None
                    else None,
                    "tokens_used": tokens_used,
                },
            )
            messages.append(response.message)

            if token_budget is not None and tokens_used > token_budget:
                outcome_status = "failed"
                outcome_reason = "token_budget"
                log.record(
                    session_id,
                    kind="task_aborted",
                    payload={
                        "reason": "token_budget",
                        "tokens_used": tokens_used,
                        "token_budget": token_budget,
                    },
                )
                break

            if not response.tool_calls:
                outcome_reason = "final_answer"
                break

            for tool_call in response.tool_calls:
                parsed = _parse_tool_arguments(tool_call.function.arguments)
                arguments = parsed.arguments

                if parsed.error is not None:
                    log.record(
                        session_id,
                        kind="tool_argument_parse_error",
                        payload={
                            "step": step,
                            "call_id": tool_call.id,
                            "name": tool_call.function.name,
                            "raw": tool_call.function.arguments,
                            "error": parsed.error,
                        },
                    )

                call = hook_call_cls(name=tool_call.function.name, arguments=arguments)
                if registry is not None:
                    call = await registry.run_pre(call)

                log.record(
                    session_id,
                    kind="tool_call",
                    payload={
                        "step": step,
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments,
                        "duration_ms": 0,
                    },
                )
                start = time.monotonic()
                output: Any
                error: str | None = None
                try:
                    output = await _execute_skill(call.name, dict(call.arguments))
                except Exception as exc:
                    output = None
                    error = f"{type(exc).__name__}: {exc}"
                duration_ms = int((time.monotonic() - start) * 1000)
                log.record(
                    session_id,
                    kind="tool_call",
                    payload={
                        "step": step,
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments,
                        "duration_ms": duration_ms,
                    },
                )
                result = hook_result_cls(name=call.name, output=output, error=error)

                if registry is not None:
                    result = await registry.run_post(call, result)

                log.record(
                    session_id,
                    kind="tool_result",
                    payload={
                        "step": step,
                        "call_id": tool_call.id,
                        "name": call.name,
                        "duration_ms": duration_ms,
                        "output": result.output,
                        "error": result.error,
                    },
                )

                messages.append(
                    ModelMessage(
                        role="tool",
                        name=call.name,
                        tool_call_id=tool_call.id,
                        content=_serialize_tool_result(result),
                    )
                )

            if step + 1 >= max_steps and response.tool_calls:
                outcome_status = "truncated"
                outcome_reason = "max_steps"
                break
    finally:
        if created_adapter and isinstance(adapter, OpenAICompatibleAdapter):
            await adapter.aclose()
        ttft_p50: int | None = int(statistics.median(turn_ttfts)) if turn_ttfts else None
        log.record(
            session_id,
            kind="outcome",
            payload={
                "status": outcome_status,
                "reason": outcome_reason,
                "steps": outcome_steps,
                "ttft_ms": ttft_p50,
                "tokens_total": tokens_used,
            },
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

    workspace_root = Path(args.workspace_root).resolve() if args.workspace_root else None

    logger = TraceLogger(args.trace_path, backend=resolve_trace_backend())
    harness_version = resolve_harness_version(harness_dir)
    model_id = args.model_id if args.model_id is not None else resolve_model_id()
    limits = run_limits_from_env()

    with logger.session(harness_version=harness_version, model_id=model_id) as session_id:
        logger.record(session_id, kind="task_received", payload={"prompt": args.task})
        start = time.monotonic()
        # ``run_task`` accepts the ``limits`` kwarg so it can enforce the
        # ``FOUNDRY_TOKEN_BUDGET`` cap (issue #197); injected ``run_task_fn``
        # stubs in the test suite keep the older four-positional-arg
        # signature for clarity, so the kwarg is only passed when we are
        # calling the module-level :func:`run_task`.
        if run_task_fn is not None:
            task_coro = task(args.task, harness_dir, logger, session_id)
        else:
            task_coro = task(
                args.task,
                harness_dir,
                logger,
                session_id,
                limits=limits,
                workspace_root=workspace_root,
            )
        try:
            asyncio.run(
                run_with_limits(
                    task_coro,
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
