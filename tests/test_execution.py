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

import sys
from pathlib import Path

import pytest

from foundry_x.execution.runner import main
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
    cli_dir = tmp_path / "from_cli"
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


# --- NotImplementedError contract ------------------------------------------


def test_main_default_run_task_raises_not_implemented_error(tmp_path, monkeypatch):
    """Issue #87 acceptance: when no ``run_task_fn`` is injected, the
    default ``run_task`` stub raises ``NotImplementedError``; ``main``
    records ``task_failed`` with ``error_type='NotImplementedError'`` and
    the original message preserved, then re-raises so the CLI sees the
    real exception (never silently swallowed — AGENTS.md §2).
    """
    db = tmp_path / "traces.db"
    harness_dir = tmp_path / "harness"
    monkeypatch.setattr(sys, "argv", _argv("default-stub", db, harness_dir))

    with pytest.raises(NotImplementedError) as exc_info:
        main()  # No injection: uses the module-level run_task stub.

    # The exception is propagated verbatim — the caller sees the same
    # object main caught, not a wrapped/replaced one.
    assert "Phase 1 wiring" in str(exc_info.value)

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
