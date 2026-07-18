"""Integration test for streaming against a real llama-server subprocess (issue #792).

Closes the "in-process ScriptedAdapter mocks only" gap called out in ADR-0020
§Open Questions (issue #552 "Does the real-LLM smoke job pass on CI with live
model?"). The runner's SSE streaming path is exercised end-to-end against an
actual ``llama-server`` subprocess bound to a local port, so the wire shape,
the retry boundary, and the ``model_response_chunk`` trace event emission are
all observed with a real GGUF model rather than a mocked adapter.

Acceptance criteria from issue #792:

- A new test file ``tests/test_execution_llamacpp_streaming.py`` spawns a real
  ``llama-server`` subprocess, runs ``run_task`` against it with
  ``FOUNDRY_TASK_TIMEOUT=120``, and asserts ``model_response_chunk`` events
  carry valid ``chunk_duration_ms`` and contiguous ``delta_index``.
- The test is skipped on CI when ``LLAMACPP_MODEL_PATH`` is not set.
- ``uv run pytest tests/test_execution_llamacpp_streaming.py -v`` passes
  locally when a GGUF model is available.

The skip rule is intentionally broader than "on CI": if no model is
available (developer workstation without a GGUF, fresh CI runner, the
binary missing from PATH), the test must skip rather than fail so the gate
stays green for everyone except the operator who explicitly opted in by
setting ``LLAMACPP_MODEL_PATH``.
"""

from __future__ import annotations

import asyncio
import os
import shutil
import socket
import subprocess
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from foundry_x.execution.model_adapter import OpenAICompatibleAdapter
from foundry_x.execution.runner import (
    RunLimits,
    run_task as real_run_task,
)
from foundry_x.trace.logger import TraceLogger

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS_DIR = REPO_ROOT / "harness"
LAUNCH_SCRIPT = REPO_ROOT / "infra" / "scripts" / "launch_llamacpp.sh"
WAIT_SCRIPT = REPO_ROOT / "infra" / "scripts" / "wait_for_llamacpp.sh"

# Task issued to the agent. Mirrors the example in the issue body
# ("write a hello world in python"). A short text task is enough to drive
# at least one SSE delta through the streaming path; the model is free
# to emit tool calls, which the stub skill executor absorbs.
_TASK = "write a hello world in python"

# Acceptance-criteria-mandated wall-clock cap (issue #792 acceptance:
# "FOUNDRY_TASK_TIMEOUT=120").
_TASK_TIMEOUT_S: float = 120.0

# Context window requested in the issue body.
_CTX_SIZE: int = 2048

# Maximum time to wait for the spawned llama-server to become healthy.
_HEALTH_TIMEOUT_S: int = 60


def _find_free_port() -> int:
    """Bind a kernel-allocated port, read it back, release the socket.

    Avoids collisions with anything else listening on the host; the
    returned port is free for the duration of the spawn window.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _resolve_skip_reason() -> str | None:
    """Return a human-readable reason to skip the integration test, or ``None``.

    The test is opt-in via ``LLAMACPP_MODEL_PATH``. The check order mirrors
    the dependency order: model file first, then the binary that loads it,
    then bash because ``launch_llamacpp.sh`` is a bash wrapper.
    """
    model_path = os.environ.get("LLAMACPP_MODEL_PATH", "").strip()
    if not model_path:
        return (
            "issue #792: LLAMACPP_MODEL_PATH is not set; provide a GGUF path "
            "to enable the real llama-server streaming integration test."
        )
    if not Path(model_path).is_file():
        return (
            f"issue #792: LLAMACPP_MODEL_PATH={model_path!r} does not point "
            "to an existing GGUF file."
        )
    if shutil.which("bash") is None:
        return "bash is not on PATH; cannot invoke launch_llamacpp.sh"
    if shutil.which("llama-server") is None and not os.environ.get("LLAMACPP_SERVER_BIN"):
        return (
            "llama-server binary is not on PATH and LLAMACPP_SERVER_BIN is "
            "unset; cannot spawn the real server."
        )
    return None


def _skip_unless_model_available() -> None:
    """Skip the integration test unless the model + binary are reachable.

    Issue #792 acceptance: "The test is skipped on CI if LLAMACPP_MODEL_PATH
    is not set." When CI is unset but the model still isn't available the
    test also skips — the absence of a GGUF on a developer workstation is
    not the test's failure to record.
    """
    reason = _resolve_skip_reason()
    if reason is not None:
        pytest.skip(reason)


def _spawn_llama_server(
    model_path: str,
    port: int,
    log_file: Path,
    pid_file: Path,
) -> subprocess.Popen[bytes]:
    """Spawn the real ``llama-server`` subprocess via the launch wrapper.

    The wrapper script is the single entry point documented in
    ``infra/llama-cpp/README.md``; reusing it (rather than invoking the
    binary directly) keeps the resolved argv identical between the test
    and the operator's Phase-3 sweeps.
    """
    return subprocess.Popen(  # noqa: S603 — args are a controlled literal list
        [
            "bash",
            str(LAUNCH_SCRIPT),
            "--model",
            model_path,
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--ctx-size",
            str(_CTX_SIZE),
            "--log-file",
            str(log_file),
            "--pid-file",
            str(pid_file),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _wait_for_ready(port: int, timeout_s: int) -> None:
    """Block until ``/health`` returns 200 or the deadline fires.

    Raises :class:`RuntimeError` on timeout so the caller can clean up the
    subprocess and surface the failure with the wrapper script's stderr.
    """
    result = subprocess.run(  # noqa: S603 — args are a controlled literal list
        [
            "bash",
            str(WAIT_SCRIPT),
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--timeout",
            str(timeout_s),
        ],
        capture_output=True,
        text=True,
        timeout=timeout_s + 10,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"llama-server failed /health probe within {timeout_s}s "
            f"(exit={result.returncode}):\n"
            f"stdout={result.stdout!r}\nstderr={result.stderr!r}"
        )


def _stop_llama_server(proc: subprocess.Popen[bytes], pid_file: Path) -> None:
    """Terminate the spawned server cleanly (SIGTERM, then SIGKILL).

    The wrapper writes its PID to ``pid_file`` before exec'ing, so the
    real llama-server PID is recoverable even though ``proc.pid`` is the
    PID of the short-lived bash wrapper itself.
    """
    pid: int | None = None
    if pid_file.is_file():
        try:
            pid = int(pid_file.read_text(encoding="utf-8").strip())
        except ValueError:
            pid = None
    if pid is not None:
        try:
            os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            pass
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


@pytest.fixture
def llama_server(tmp_path: Path) -> Iterator[tuple[str, int]]:
    """Spawn a real ``llama-server`` subprocess; yield ``(base_url, port)``.

    Cleanup is unconditional (try/finally) so a hung llama-server cannot
    leak across tests or poison subsequent invocations on the same port.
    Skips via :func:`_skip_unless_model_available` when prerequisites are
    missing so the suite stays green for developers without a GGUF.
    """
    _skip_unless_model_available()
    port = _find_free_port()
    log_file = tmp_path / "llama.log"
    pid_file = tmp_path / "llama.pid"
    proc = _spawn_llama_server(
        os.environ["LLAMACPP_MODEL_PATH"],
        port,
        log_file,
        pid_file,
    )
    try:
        _wait_for_ready(port, timeout_s=_HEALTH_TIMEOUT_S)
    except Exception:
        _stop_llama_server(proc, pid_file)
        raise
    base_url = f"http://127.0.0.1:{port}"
    try:
        yield base_url, port
    finally:
        _stop_llama_server(proc, pid_file)


def _reset_default_registry() -> None:
    """Clear the harness's default hook registry so test isolation holds.

    The foundry's import-side self-reference rule (AGENTS.md §7) means
    the runner only imports ``harness.hooks`` lazily, but the harness
    itself self-registers the prompt-input firewall on import. Calling
    :func:`reset_default_registry` ensures the in-process run starts
    with a clean registry even if a prior test loaded the harness. The
    firewall is then re-registered so subsequent test files
    (``tests/test_injection_firewall.py``, ``tests/test_hook_isolation.py``)
    still observe the default hook contract (issue #5 self-registration).
    """
    try:
        from harness.hooks.base import reset_default_registry
        from harness.hooks.injection_firewall import InjectionFirewallHook
        from harness.hooks import register_hook
    except ImportError:
        return
    reset_default_registry()
    register_hook(InjectionFirewallHook())


@pytest.fixture(autouse=True)
def _clean_registry() -> None:
    """Autouse isolation: clear the harness hook registry before every test."""
    _reset_default_registry()


async def _stub_skill_executor(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Stub executor that absorbs every tool call without side effects.

    The streaming path is what we are validating; tool execution is
    incidental. Without this stub the model could call ``bash`` /
    ``write_file`` and execute real commands against the workspace root,
    which is undesirable inside a test process.
    """
    return {"stub": True, "skill": name, "echo_keys": sorted(arguments.keys())}


def test_real_llama_server_streaming_emits_chunk_events(
    llama_server: tuple[str, int],
    tmp_path: Path,
) -> None:
    """End-to-end: a real GGUF model streamed over SSE produces
    ``model_response_chunk`` events with valid ``chunk_duration_ms`` and
    contiguous ``delta_index`` (issue #792 acceptance criteria).

    The chunk event stream is the integration point between the model
    adapter (issue #199) and the trace store; if ``delta_index`` skips or
    ``chunk_duration_ms`` is negative, the KPI consumer cannot reason
    about TTFT or inter-chunk latency.
    """
    base_url, _port = llama_server
    trace_db = tmp_path / "traces.db"

    model_id = Path(os.environ["LLAMACPP_MODEL_PATH"]).name
    adapter = OpenAICompatibleAdapter(
        base_url=base_url,
        model=model_id,
        timeout=60.0,
    )
    limits = RunLimits(task_timeout_s=_TASK_TIMEOUT_S, token_budget=None)

    async def drive() -> None:
        logger = TraceLogger(trace_db)
        with logger.session(harness_version="0.0.0-llamacpp-streaming-test") as session_id:
            try:
                await real_run_task(
                    _TASK,
                    HARNESS_DIR,
                    logger,
                    session_id,
                    model_adapter=adapter,
                    skill_executor=_stub_skill_executor,
                    limits=limits,
                )
            finally:
                await adapter.aclose()

    asyncio.run(drive())

    sessions = TraceLogger(trace_db).list_sessions()
    assert sessions, "run_task did not open a trace session"
    events = TraceLogger(trace_db).load_session(sessions[0].session_id)

    chunk_events = [event for event in events if event.kind == "model_response_chunk"]
    assert chunk_events, (
        "No model_response_chunk events emitted; the streaming path did not "
        f"fire. Events seen: {[e.kind for e in events]}"
    )

    durations = [event.payload["chunk_duration_ms"] for event in chunk_events]
    assert all(d >= 0 for d in durations), (
        f"chunk_duration_ms must be non-negative; saw {durations}"
    )

    deltas = [event.payload["delta_index"] for event in chunk_events]
    assert deltas == list(range(len(chunk_events))), (
        f"delta_index must be contiguous from 0; saw {deltas}"
    )

    # All chunks in this single-turn task share a single ``step`` ordinal.
    # Cross-step continuity is exercised by ``tests/execution/test_runner_stream.py``.
    steps = {event.payload["step"] for event in chunk_events}
    assert steps == {0}, f"all chunks should share step=0; got {steps}"


def test_skip_when_prerequisites_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sanity check: the integration test must skip — not error — when
    ``LLAMACPP_MODEL_PATH`` is unset (issue #792 acceptance).

    Without this guard an unattended CI run would hard-fail the suite
    even though the model is unavailable by design.
    """
    monkeypatch.delenv("LLAMACPP_MODEL_PATH", raising=False)
    monkeypatch.delenv("LLAMACPP_SERVER_BIN", raising=False)
    assert _resolve_skip_reason() is not None, (
        "expected a skip reason when LLAMACPP_MODEL_PATH is unset"
    )

    with pytest.raises(pytest.skip.Exception):
        _skip_unless_model_available()
