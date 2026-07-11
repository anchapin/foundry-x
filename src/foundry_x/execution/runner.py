from __future__ import annotations

import argparse
import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

from pydantic import BaseModel, Field

from foundry_x.execution.model_adapter import (
    ModelAdapter,
    ModelMessage,
    OpenAICompatibleAdapter,
    ToolDefinition,
    ToolFunctionSchema,
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


# Skill-executor protocol (issue #89, ADR-0010). A callable that maps a skill
# name + already-decoded arguments dict to the tool result the model will see.
# ``Awaitable[Any]`` lets callers return arbitrary JSON-shaped data; the runner
# serializes the result with ``json.dumps`` before injecting it back into the
# model's tool channel. The default executor is a stub that acknowledges the
# call (issue #104 declared the JSON contract; the actual subprocess.run-
# backed hook lands in a follow-up PR so the Critic can gate it independently).
SkillExecutor = Callable[[str, dict[str, Any]], Awaitable[Any]]


def resolve_trace_backend(env: dict[str, str] | None = None) -> str:
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


def resolve_model_id(env: dict[str, str] | None = None) -> str | None:
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


def _resolve_model_request_name(env: dict[str, str] | None = None) -> str:
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


def build_model_adapter(env: dict[str, str] | None = None) -> OpenAICompatibleAdapter:
    """Create the default OpenAI-compatible ModelAdapter from environment.

    ``OPENCODE_SERVER_URL`` remains the primary endpoint knob from
    ``.env.example``; ``LLAMACPP_HOST`` is accepted as a local-first fallback.
    API keys are read only from environment and never persisted here.
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


def _resolve_max_steps(env: dict[str, str] | None = None) -> int:
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


def _resolve_hook_registry() -> Any | None:
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
    """
    try:
        from harness.hooks import get_registry
    except ImportError:
        return None
    try:
        return get_registry()
    except Exception:
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


async def _default_skill_executor(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Default skill executor that acknowledges the call (issue #89).

    Issue #104 declares the bash skill's JSON contract; the actual
    ``subprocess.run``-backed executor lands in a follow-up so the Critic
    gate (ADR-0004) can evaluate it independently. Until then the runner
    still needs *something* to close the loop and emit a ``tool_result``
    event, so this stub returns a benign envelope the model can act on.

    The shape is stable (``status`` + ``skill`` + ``echo`` of the argument
    keys) so tests asserting the wiring can pattern-match without coupling
    to the eventual real executor.
    """
    return {
        "status": "ok",
        "skill": name,
        "echo": sorted(arguments.keys()),
    }


def _parse_tool_arguments(raw: str) -> dict[str, Any]:
    """Parse an OpenAI-compatible tool-call ``arguments`` JSON string.

    Some models emit ``""`` (no arguments) and a few emit partially-formed
    JSON; both are coerced to an empty dict so the agent loop can stamp the
    call into the trace without a model-side parser failure (which would
    abort the loop and leave the session without an ``outcome`` event).
    Non-dict JSON values are also collapsed to ``{}`` for the same reason:
    the loop cares about the *executor contract*, not the wire shape.
    """
    if not raw:
        return {}
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    return dict(decoded) if isinstance(decoded, dict) else {}


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


async def run_task(
    task: str,
    harness_dir: Path,
    log: TraceLogger,
    session_id: str,
    model_adapter: ModelAdapter | None = None,
    *,
    skill_executor: SkillExecutor | None = None,
) -> None:
    """Drive one task through the asyncio agent loop (issue #89, ADR-0010).

    Reads ``harness/system_prompt.txt`` and the OpenAI-compatible tool surface
    declared in ``harness/skills/*.json``, exchanges turns with the
    ``model_adapter`` until the model emits a final assistant message, the
    ``max_steps`` cap is reached, or the wall-clock cap fires, and records
    every step into the trace store:

    1. ``user_prompt`` — the task enters the agent conversation
       (the lifecycle ``task_received`` marker from ``main()`` is preserved).
    2. ``model_request`` / ``model_response`` — every chat completion round-trip.
    3. ``tool_call`` / ``tool_result`` — one pair per ``ToolCall`` the model
       emits, bracketed by ``HookRegistry.run_pre`` and ``HookRegistry.run_post``
       fan-out so the prompt-input firewall (SECURITY.md) and future hooks
       observe every step.
    4. ``outcome`` — terminal event with ``status`` and ``reason`` so the Phase
       2 Digester can attribute success vs. truncation vs. failure.

    The tool surface is data-driven: a skill lands as soon as its JSON file
    does (issue #104, #105). Skill execution is delegated to ``skill_executor``
    (default: ``_default_skill_executor`` — a stub that acknowledges the call);
    real ``subprocess.run``-backed executors are wired in a follow-up so the
    Critic gate (ADR-0004) can evaluate them independently.

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

    registry = _resolve_hook_registry()
    hook_call_cls, hook_result_cls = _import_hook_types()

    executor = skill_executor or _default_skill_executor
    max_steps = _resolve_max_steps()

    log.record(
        session_id,
        kind="user_prompt",
        payload={"content": task, "tool_count": len(tool_definitions)},
    )

    outcome_status = "success"
    outcome_reason = "final_answer"
    outcome_steps = 0

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
                response = await adapter.complete(messages=messages, tools=tool_definitions)
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

            log.record(
                session_id,
                kind="model_response",
                payload={
                    "step": step,
                    "finish_reason": response.finish_reason,
                    "message": response.message.model_dump(mode="json", exclude_none=True),
                    "tool_calls": [call.model_dump(mode="json") for call in response.tool_calls],
                },
            )
            messages.append(response.message)

            if not response.tool_calls:
                outcome_reason = "final_answer"
                break

            for tool_call in response.tool_calls:
                arguments = _parse_tool_arguments(tool_call.function.arguments)
                log.record(
                    session_id,
                    kind="tool_call",
                    payload={
                        "step": step,
                        "call_id": tool_call.id,
                        "name": tool_call.function.name,
                        "arguments": arguments,
                    },
                )

                call = hook_call_cls(name=tool_call.function.name, arguments=arguments)
                if registry is not None:
                    call = await registry.run_pre(call)

                start = time.monotonic()
                output: Any
                error: str | None = None
                try:
                    output = await executor(call.name, dict(call.arguments))
                except Exception as exc:
                    output = None
                    error = f"{type(exc).__name__}: {exc}"
                duration_ms = int((time.monotonic() - start) * 1000)
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
        log.record(
            session_id,
            kind="outcome",
            payload={
                "status": outcome_status,
                "reason": outcome_reason,
                "steps": outcome_steps,
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

    logger = TraceLogger(args.trace_path, backend=resolve_trace_backend())
    harness_version = resolve_harness_version(harness_dir)
    model_id = resolve_model_id()
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
