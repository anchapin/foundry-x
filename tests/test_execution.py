"""End-to-end Runner lifecycle tests for issue #87.

The focused unit tests under ``tests/execution/`` (harness_version,
limits, model_id, trace_backend, terminal_event) exercise the runner's
small helpers and a few terminal-event scenarios in isolation. This file
covers the seams those don't:

- ``argparse`` surface of ``main`` (required flag, all flags accepted)
- ``FOUNDRY_HARNESS_DIR`` / ``FOUNDRY_TRACE_PATH`` environment defaults
  when the matching CLI flag is absent (and the CLI override when it
  isn't)
- The harness_dir -> ``sys.path`` wiring that lets ``run_task`` import
  harness modules
- The full session lifecycle (one ``task_received`` on entry, one
  terminal ``task_completed``/``task_failed`` on exit)
- The documented ``NotImplementedError`` contract of the default
  ``run_task`` stub: ``main`` records ``task_failed`` with the original
  ``error_type`` and ``message`` preserved, then re-raises
- The ``run_task_fn`` injection point, with arguments flowing through
  unchanged

If a future refactor changes any of these, the tests in this file
should fail before the regression reaches production. Per
``docs/PHILOSOPHY.md`` (smallest viable change) and AGENTS.md §2 (never
widen scope) this file ONLY adds tests; ``runner.py`` is untouched.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

from foundry_x.execution.harness_layout import (
    HarnessValidationError,
    validate as validate_harness_layout,
)
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import DEFAULT_TASK_TIMEOUT_S, main, run_task
from foundry_x.trace.logger import TraceLogger


def _argv(task: str, trace_path: Path, harness_dir: Path | None = None) -> list[str]:
    """Build the ``sys.argv`` list ``main`` expects; tests monkeypatch this in.

    ``harness_dir`` is optional because two of the env-default tests want
    to assert what happens when the flag is omitted entirely; the rest
    pass it explicitly so the runner never resolves the relative
    ``./harness`` default against the test process's CWD.
    """
    argv = ["fx-runner", "--task", task, "--trace-path", str(trace_path)]
    if harness_dir is not None:
        argv += ["--harness-dir", str(harness_dir)]
    return argv


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``.

    The Runner validates the harness layout (system_prompt.txt, hooks/,
    skills/) on entry; without these stubs, main() aborts with a
    HarnessValidationError before reaching the run_task branch under test.
    """
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


# --- argparse surface ------------------------------------------------------


def test_main_requires_task_argument(tmp_path, monkeypatch):
    """``argparse`` exits non-zero when the mandatory ``--task`` is missing.

    Regression guard for the CLI contract: ``--task`` is documented as
    ``required=True`` and must remain so; a silent default would let an
    empty session slip through the trace store.
    """
    db = tmp_path / "traces.db"
    monkeypatch.setattr(sys, "argv", ["fx-runner", "--trace-path", str(db)])

    with pytest.raises(SystemExit):
        main()


def test_main_accepts_all_documented_cli_flags(tmp_path, monkeypatch):
    """Every documented CLI flag (``--task``, ``--harness-dir``,
    ``--trace-path``) parses cleanly.

    The runner's CLI surface is small on purpose; if a flag is renamed
    or removed this test fails first, before any downstream test that
    relies on ``_argv``.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fx-runner",
            "--task",
            "hello",
            "--trace-path",
            str(db),
            "--harness-dir",
            str(harness_dir),
        ],
    )
    # Clear env so we exercise the explicit CLI path, not the env default.
    monkeypatch.delenv("FOUNDRY_HARNESS_DIR", raising=False)
    monkeypatch.delenv("FOUNDRY_TRACE_PATH", raising=False)

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    # A session was opened and the SQLite database was written.
    assert db.exists()
    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1


# --- environment defaults --------------------------------------------------


def test_main_harness_dir_falls_back_to_foundry_harness_dir_env(tmp_path, monkeypatch):
    """When ``--harness-dir`` is omitted, ``FOUNDRY_HARNESS_DIR`` wins.

    The fallback chain is documented in ``.env.example``: env-var
    first, literal ``"./harness"`` last. The test exercises the first
    leg of that chain by capturing what ``run_task`` actually receives.
    """
    env_dir = tmp_path / "from_env"
    _stub_harness(env_dir)
    db = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_HARNESS_DIR", str(env_dir))
    monkeypatch.setattr(
        sys,
        "argv",
        ["fx-runner", "--task", "x", "--trace-path", str(db)],
    )

    captured: dict[str, Path] = {}

    async def capture_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        captured["harness_dir"] = Path(harness_dir)

    main(run_task_fn=capture_run_task)

    # ``main`` resolves the env-supplied path before handing it to run_task.
    assert captured["harness_dir"] == env_dir.resolve()


def test_main_trace_path_falls_back_to_foundry_trace_path_env(tmp_path, monkeypatch):
    """When ``--trace-path`` is omitted, ``FOUNDRY_TRACE_PATH`` is used.

    The trace store ends up at the env-supplied location, not at the
    argparse default of ``./logs/traces.db``.
    """
    env_trace = tmp_path / "from_env_traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    monkeypatch.setenv("FOUNDRY_TRACE_PATH", str(env_trace))
    monkeypatch.setattr(
        sys,
        "argv",
        ["fx-runner", "--task", "x", "--harness-dir", str(harness_dir)],
    )

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    assert env_trace.exists()
    # And the file is a valid SQLite database (default backend).
    assert env_trace.read_bytes()[:15] == b"SQLite format 3"


def test_main_cli_harness_dir_overrides_env(tmp_path, monkeypatch):
    """When both env and CLI are set, the CLI flag wins.

    Mirrors the ``default=os.environ.get(...)`` semantics: argparse
    evaluates the default ONCE at parse time. If the CLI value is
    provided it replaces the env-derived default wholesale.
    """
    env_dir = tmp_path / "from_env"
    _stub_harness(env_dir)
    cli_dir = tmp_path / "from_cli"
    _stub_harness(cli_dir)
    db = tmp_path / "traces.db"
    monkeypatch.setenv("FOUNDRY_HARNESS_DIR", str(env_dir))
    monkeypatch.setattr(
        sys,
        "argv",
        _argv("x", db, cli_dir),
    )

    captured: dict[str, Path] = {}

    async def capture_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        captured["harness_dir"] = Path(harness_dir)

    main(run_task_fn=capture_run_task)

    assert captured["harness_dir"] == cli_dir.resolve()


# --- harness_dir -> sys.path wiring ----------------------------------------


def test_main_inserts_resolved_harness_dir_into_sys_path(tmp_path, monkeypatch):
    """``main`` prepends the resolved ``--harness-dir`` to ``sys.path``.

    ``run_task`` imports harness modules (``harness.hooks``,
    ``harness.skills``); without this wiring the import would fail. The
    contract is: ``str(harness_dir.resolve())`` is added iff it is not
    already present.
    """
    harness_dir = tmp_path / "harness_subdir"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"
    monkeypatch.setattr(sys, "argv", _argv("x", db, harness_dir))

    expected = str(harness_dir.resolve())
    # Remove any prior occurrence so we observe a fresh insertion.
    sys.path[:] = [p for p in sys.path if p != expected]

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    assert expected in sys.path


# --- session lifecycle -----------------------------------------------------


def test_main_session_lifecycle_records_received_then_completed(tmp_path, monkeypatch):
    """On a successful run, exactly one ``task_received`` is recorded on
    entry and exactly one ``task_completed`` is recorded on exit; no
    ``task_failed`` is ever emitted.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    monkeypatch.setattr(sys, "argv", _argv("happy", db, harness_dir))

    async def noop_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        return None

    main(run_task_fn=noop_run_task)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)
    kinds = [e.kind for e in events]

    assert kinds.count("task_received") == 1
    assert kinds.count("task_completed") == 1
    assert "task_failed" not in kinds

    # The task prompt is preserved on the received event for traceability.
    received = [e for e in events if e.kind == "task_received"][0]
    assert received.payload["prompt"] == "happy"

    # The terminal event carries a non-negative duration.
    completed = [e for e in events if e.kind == "task_completed"][0]
    assert completed.payload["duration_ms"] >= 0


# --- run_task_fn exception contract -----------------------------------------


def test_main_records_task_failed_when_run_task_fn_raises(tmp_path, monkeypatch):
    """Issue #87 acceptance: when ``run_task_fn`` raises, ``main`` records
    ``task_failed`` with the original ``error_type`` and ``message``
    preserved, then re-raises so the CLI sees the real exception
    (never silently swallowed — AGENTS.md §2).

    Note: the original test asserted the module-level ``run_task`` stub
    raises ``NotImplementedError``. PR #148 (issue #88) wired
    ``run_task`` to a real ModelAdapter, so the stub contract moved
    behind the ``run_task_fn`` injection point. This test exercises
    the same contract via injection.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    monkeypatch.setattr(sys, "argv", _argv("injected-stub", db, harness_dir))

    async def raising_run_task(task, harness_dir, log, session_id):  # noqa: ANN001, ARG001
        raise NotImplementedError("Phase 1 wiring not yet connected")

    with pytest.raises(NotImplementedError) as exc_info:
        main(run_task_fn=raising_run_task)

    # The exception is propagated verbatim — the caller sees the same
    # object main caught, not a wrapped/replaced one.
    assert "Phase 1 wiring" in str(exc_info.value)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)
    failed = [e for e in events if e.kind == "task_failed"]
    assert len(failed) == 1
    assert failed[0].payload["error_type"] == "NotImplementedError"
    assert "Phase 1 wiring" in failed[0].payload["message"]

    logger = TraceLogger(db)
    sessions = logger.list_sessions()
    assert len(sessions) == 1
    events = logger.load_session(sessions[0].session_id)
    kinds = [e.kind for e in events]

    assert kinds.count("task_received") == 1
    assert kinds.count("task_failed") == 1
    assert "task_completed" not in kinds

    failed = [e for e in events if e.kind == "task_failed"][0]
    assert failed.payload["error_type"] == "NotImplementedError"
    # The full original message survives redaction-free into the trace.
    assert "Phase 1 wiring" in failed.payload["message"]
    assert failed.payload["duration_ms"] >= 0


# --- run_task_fn injection point -------------------------------------------


def test_run_task_fn_injection_replaces_default(tmp_path, monkeypatch):
    """``main(run_task_fn=stub)`` invokes ``stub`` instead of the
    module-level ``run_task``. The stub sees the task string and the
    resolved harness_dir.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    monkeypatch.setattr(sys, "argv", _argv("injected", db, harness_dir))

    captured: dict[str, object] = {}
    invocation_count = 0

    async def stub_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        nonlocal invocation_count
        invocation_count += 1
        captured["task"] = task
        captured["harness_dir"] = Path(harness_dir)
        captured["session_id"] = session_id

    main(run_task_fn=stub_run_task)

    # The stub ran exactly once — the module-level run_task would have
    # raised NotImplementedError, so reaching this assertion proves the
    # injection replaced it.
    assert invocation_count == 1
    assert captured["task"] == "injected"
    assert captured["harness_dir"] == harness_dir.resolve()
    assert isinstance(captured["session_id"], str)
    assert captured["session_id"]  # non-empty UUID


def test_run_task_fn_receives_logger_within_active_session(tmp_path, monkeypatch):
    """The logger passed to the injected ``run_task_fn`` is the same
    one that ends up writing the session, so events the stub records
    land in the trace alongside ``task_received``/``task_completed``.
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    monkeypatch.setattr(sys, "argv", _argv("logger-check", db, harness_dir))

    async def stub_run_task(task, harness_dir, log, session_id):  # noqa: ANN001
        log.record(session_id, kind="tool_call", payload={"name": "stub_event"})

    main(run_task_fn=stub_run_task)

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)
    kinds = [e.kind for e in events]

    # Stub event + lifecycle events all coexist in one session.
    assert "tool_call" in kinds
    assert kinds.count("task_received") == 1
    assert kinds.count("task_completed") == 1


def test_run_task_records_tool_call_duration_ms(tmp_path):
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    tool_call = ModelToolCall(
        id="call_latency",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "true"}),
        ),
    )

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    message=ModelMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[tool_call],
                    ),
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
            )

        async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
            return await self.complete(messages, tools, **kwargs)

        async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            response = await self.complete(messages, tools, **kwargs)
            if response.message.content:
                yield ModelResponseChunk(content=response.message.content)
            for i, tc in enumerate(response.tool_calls):
                yield ModelResponseChunk(
                    tool_calls=[
                        ModelToolCallChunk(
                            index=i,
                            id=tc.id,
                            type=tc.type,
                            function=ToolCallFunctionChunk(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    ]
                )
            if response.finish_reason:
                yield ModelResponseChunk(finish_reason=response.finish_reason)

    async def executor(name, arguments):  # noqa: ANN001, ARG001
        return {"status": "ok"}

    async def drive() -> None:
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as session_id:
            await run_task(
                "latency-check",
                harness_dir,
                logger,
                session_id,
                model_adapter=Adapter(),
                skill_executor=executor,
            )

    asyncio.run(drive())

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)
    tool_call_event = next(event for event in events if event.kind == "tool_call")
    assert "duration_ms" in tool_call_event.payload
    duration_ms = tool_call_event.payload["duration_ms"]
    assert duration_ms >= 0
    assert duration_ms < DEFAULT_TASK_TIMEOUT_S


def test_run_task_records_hook_overhead_ms_with_delayed_hook(tmp_path, monkeypatch):
    """hook_overhead_ms reflects run_pre delay when a hook is registered (issue #709).

    Verifies that when a hook with a known delay is registered via a mocked
    _resolve_hook_registry, the tool_call event carries the measured
    hook_overhead_ms. The test covers three scenarios: no hooks (null),
    one hook (positive integer), and multiple hooks (combined delay).
    """
    from foundry_x.execution import runner

    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    tool_call = ModelToolCall(
        id="call_hook",
        type="function",
        function=ToolCallFunction(
            name="bash",
            arguments=json.dumps({"command": "true"}),
        ),
    )

    class Adapter:
        def __init__(self) -> None:
            self.calls = 0

        async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            self.calls += 1
            if self.calls == 1:
                return ModelResponse(
                    message=ModelMessage(
                        role="assistant",
                        content=None,
                        tool_calls=[tool_call],
                    ),
                    tool_calls=[tool_call],
                    finish_reason="tool_calls",
                )
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
            )

        async def chat(self, messages, tools=None, **kwargs):  # noqa: ANN001
            return await self.complete(messages, tools, **kwargs)

        async def stream(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
            response = await self.complete(messages, tools, **kwargs)
            if response.message.content:
                yield ModelResponseChunk(content=response.message.content)
            for i, tc in enumerate(response.tool_calls):
                yield ModelResponseChunk(
                    tool_calls=[
                        ModelToolCallChunk(
                            index=i,
                            id=tc.id,
                            type=tc.type,
                            function=ToolCallFunctionChunk(
                                name=tc.function.name,
                                arguments=tc.function.arguments,
                            ),
                        )
                    ]
                )
            if response.finish_reason:
                yield ModelResponseChunk(finish_reason=response.finish_reason)

    async def executor(name, arguments):  # noqa: ANN001, ARG001
        return {"status": "ok"}

    # Case 1: No hooks - mock _resolve_hook_registry to return None
    def mock_resolve_none(log, session_id):  # noqa: ANN001
        return None

    monkeypatch.setattr(runner, "_resolve_hook_registry", mock_resolve_none)

    async def drive_no_hooks():
        logger = TraceLogger(db)
        with logger.session(harness_version="0.1.0") as session_id:
            await run_task(
                "hook-overhead-check",
                harness_dir,
                logger,
                session_id,
                model_adapter=Adapter(),
                skill_executor=executor,
            )

    asyncio.run(drive_no_hooks())

    logger = TraceLogger(db)
    events = logger.load_session(logger.list_sessions()[0].session_id)
    tool_call_events = [e for e in events if e.kind == "tool_call"]
    assert len(tool_call_events) == 2
    for event in tool_call_events:
        assert "hook_overhead_ms" in event.payload
        assert event.payload["hook_overhead_ms"] is None

    # Case 2: One hook with 50ms delay
    hook_delay_ms = 50

    class MockRegistryOneHook:
        _hooks: list = []

        async def run_pre(self, call):
            await asyncio.sleep(hook_delay_ms / 1000.0)
            return call

        async def run_post(self, call, result):
            return result

    def mock_resolve_one(log, session_id):  # noqa: ANN001
        return MockRegistryOneHook()

    monkeypatch.setattr(runner, "_resolve_hook_registry", mock_resolve_one)

    db2 = tmp_path / "traces2.db"

    async def drive_one_hook():
        logger = TraceLogger(db2)
        with logger.session(harness_version="0.1.0") as session_id:
            await run_task(
                "hook-overhead-check",
                harness_dir,
                logger,
                session_id,
                model_adapter=Adapter(),
                skill_executor=executor,
            )

    asyncio.run(drive_one_hook())

    logger = TraceLogger(db2)
    events = logger.load_session(logger.list_sessions()[0].session_id)
    tool_call_events = [e for e in events if e.kind == "tool_call"]
    assert len(tool_call_events) == 2
    for event in tool_call_events:
        assert "hook_overhead_ms" in event.payload
        assert event.payload["hook_overhead_ms"] is not None
        assert event.payload["hook_overhead_ms"] >= hook_delay_ms
        assert event.payload["hook_overhead_ms"] < hook_delay_ms + 100

    # Case 3: Two hooks with combined delay
    hook1_delay_ms = 30
    hook2_delay_ms = 40

    class MockRegistryTwoHooks:
        _hooks: list = []

        async def run_pre(self, call):
            await asyncio.sleep(hook1_delay_ms / 1000.0)
            await asyncio.sleep(hook2_delay_ms / 1000.0)
            return call

        async def run_post(self, call, result):
            return result

    def mock_resolve_two(log, session_id):  # noqa: ANN001
        return MockRegistryTwoHooks()

    monkeypatch.setattr(runner, "_resolve_hook_registry", mock_resolve_two)

    db3 = tmp_path / "traces3.db"

    async def drive_two_hooks():
        logger = TraceLogger(db3)
        with logger.session(harness_version="0.1.0") as session_id:
            await run_task(
                "hook-overhead-check",
                harness_dir,
                logger,
                session_id,
                model_adapter=Adapter(),
                skill_executor=executor,
            )

    asyncio.run(drive_two_hooks())

    logger = TraceLogger(db3)
    events = logger.load_session(logger.list_sessions()[0].session_id)
    tool_call_events = [e for e in events if e.kind == "tool_call"]
    assert len(tool_call_events) == 2
    min_expected = hook1_delay_ms + hook2_delay_ms
    for event in tool_call_events:
        assert "hook_overhead_ms" in event.payload
        assert event.payload["hook_overhead_ms"] is not None
        assert event.payload["hook_overhead_ms"] >= min_expected
        assert event.payload["hook_overhead_ms"] < min_expected + 100


# --- harness layout validation (issue #90) ---------------------------------


def test_validate_accepts_valid_harness_layout(tmp_path):
    """``validate()`` returns ``None`` on a layout that satisfies the runner.

    Acceptance criterion: a harness checkout exposing
    ``system_prompt.txt``, ``hooks/``, and ``skills/`` is treated as
    valid; ``validate()`` is a pure pre-flight (no side effects), so
    the absence of an exception is the contract.
    """
    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)

    assert validate_harness_layout(harness_dir) is None


def test_validate_reports_all_three_missing_entries_for_empty_dir(tmp_path):
    """An empty ``harness_dir`` reports every required entry in a single
    ``HarnessValidationError`` so the operator can fix all gaps in one
    pass instead of cycling one error at a time.
    """
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()

    with pytest.raises(HarnessValidationError) as exc_info:
        validate_harness_layout(harness_dir)

    assert sorted(exc_info.value.missing) == sorted(["system_prompt.txt", "hooks", "skills"])
    assert exc_info.value.harness_dir == harness_dir
    message = str(exc_info.value)
    assert str(harness_dir) in message


def test_validate_reports_missing_system_prompt(tmp_path):
    """With ``hooks/`` and ``skills/`` present but no ``system_prompt.txt``,
    only the prompt is reported.
    """
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "hooks").mkdir()
    (harness_dir / "skills").mkdir()

    with pytest.raises(HarnessValidationError) as exc_info:
        validate_harness_layout(harness_dir)

    assert exc_info.value.missing == ["system_prompt.txt"]


def test_validate_reports_missing_hooks_dir(tmp_path):
    """With the prompt and ``skills/`` present but no ``hooks/``, only
    the hooks directory is reported.
    """
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("stub\n")
    (harness_dir / "skills").mkdir()

    with pytest.raises(HarnessValidationError) as exc_info:
        validate_harness_layout(harness_dir)

    assert exc_info.value.missing == ["hooks"]


def test_validate_reports_missing_skills_dir(tmp_path):
    """With the prompt and ``hooks/`` present but no ``skills/``, only
    the skills directory is reported.
    """
    harness_dir = tmp_path / "harness"
    harness_dir.mkdir()
    (harness_dir / "system_prompt.txt").write_text("stub\n")
    (harness_dir / "hooks").mkdir()

    with pytest.raises(HarnessValidationError) as exc_info:
        validate_harness_layout(harness_dir)

    assert exc_info.value.missing == ["skills"]


# --- main() integration with harness validation ----------------------------


def test_main_exits_non_zero_and_prints_missing_entries_when_harness_invalid(
    tmp_path, monkeypatch, capsys
):
    """Issue #90 acceptance: ``main()`` invokes ``validate(args.harness_dir)``
    before ``sys.path`` injection; on failure it writes the harness_dir and
    each missing entry to ``stderr`` and ``sys.exit(2)``s -- the operator
    sees the gap, not a Python traceback.
    """
    bad = tmp_path / "bad_harness"
    bad.mkdir()
    (bad / "system_prompt.txt").write_text("stub\n")
    (bad / "skills").mkdir()
    # ``hooks/`` deliberately missing so a single-entry list is asserted below.

    db = tmp_path / "traces.db"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fx-runner",
            "--task",
            "x",
            "--harness-dir",
            str(bad),
            "--trace-path",
            str(db),
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2

    captured = capsys.readouterr()
    # The harness_dir is named explicitly so the error is actionable.
    assert str(bad) in captured.err
    # Every missing entry is named; order is stable (hooks is the only miss).
    assert "hooks" in captured.err
    # No traceback on the user-facing path (AGENTS.md §1 "evidence over opinion").
    assert "Traceback" not in captured.err
    # No trace session was opened because the runner aborts before TraceLogger.
    assert not db.exists() or not TraceLogger(db).list_sessions()


def test_main_does_not_pollute_sys_path_when_harness_invalid(tmp_path, monkeypatch):
    """``main()`` runs ``validate()`` strictly *before* ``sys.path.insert``;
    a malformed harness checkout therefore never lands on the import path
    and cannot be picked up by subsequent imports in the same process.
    """
    bad = tmp_path / "bad_harness"
    bad.mkdir()
    expected = str(bad.resolve())
    sys.path[:] = [p for p in sys.path if p != expected]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "fx-runner",
            "--task",
            "x",
            "--harness-dir",
            str(bad),
            "--trace-path",
            str(tmp_path / "traces.db"),
        ],
    )

    with pytest.raises(SystemExit):
        main()

    assert expected not in sys.path


# --- fx-runner console script registration (issue #198) ---------------------


def test_fx_runner_help_prints_documented_argparse_surface(tmp_path):
    """Issue #198 acceptance: ``fx-runner --help`` exits zero and prints the
    documented argparse surface (``--task``, ``--harness-dir``, ``--trace-path``).

    Guards the ``[project.scripts]`` entry in ``pyproject.toml``: if the
    registration is removed or the target moves, the subprocess fails to
    resolve the command and this test fails first.
    """
    import shutil
    import subprocess

    runner = shutil.which("fx-runner")
    assert runner is not None, (
        "fx-runner console script not on PATH; install with 'uv pip install -e .'"
    )

    result = subprocess.run(
        [runner, "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    for flag in ("--task", "--harness-dir", "--trace-path"):
        assert flag in result.stdout, f"{flag} missing from --help output"


def test_fx_runner_console_script_lands_session_in_trace_store(tmp_path):
    """Issue #198 acceptance: invoking the ``fx-runner`` console script via
    subprocess opens exactly one session and lands it in the SQLite trace
    store.

    The subprocess is expected to exit non-zero: without a model endpoint
    (``OPENCODE_SERVER_URL`` / ``LLAMACPP_HOST``) ``build_model_adapter``
    raises ``ValueError``, which ``main`` records as ``task_failed`` before
    re-raising. What matters for this acceptance criterion is that the
    session was opened and persisted — the operator can inspect it in the
    trace store regardless of the runner outcome.
    """
    import os
    import shutil
    import subprocess

    runner = shutil.which("fx-runner")
    assert runner is not None, (
        "fx-runner console script not on PATH; install with 'uv pip install -e .'"
    )

    harness_dir = tmp_path / "harness"
    _stub_harness(harness_dir)
    db = tmp_path / "traces.db"

    result = subprocess.run(
        [
            runner,
            "--task",
            "hello",
            "--harness-dir",
            str(harness_dir),
            "--trace-path",
            str(db),
        ],
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHONPATH": "."},
    )

    sessions = TraceLogger(db).list_sessions()
    assert len(sessions) == 1, (
        f"expected exactly 1 session in trace store, got {len(sessions)}; "
        f"returncode={result.returncode} "
        f"stdout={result.stdout!r} stderr={result.stderr!r}"
    )

    events = TraceLogger(db).load_session(sessions[0].session_id)
    kinds = [e.kind for e in events]
    assert "task_received" in kinds
