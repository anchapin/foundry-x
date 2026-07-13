"""Benchmark task: real-LLM trace-capture smoke against llama-server (issue #175).

Phase-3 plumbing canary for the Digester -> Evolver -> Critic loop. The test
drives :func:`foundry_x.execution.runner.run_task` against the live
``llama-server`` endpoint using the ``sort_a_list`` task prompt and asserts
the trace-store plumbing:

1. The trace database is created at the configured path (``logs/`` by default)
   with exactly one :class:`~foundry_x.trace.logger.TraceSession` whose
   ``harness_version`` is non-null.
2. The terminal ``outcome.status`` is one of ``{"success", "truncated",
   "failed"}`` -- model-quality is out of scope (PRD §5 places it on the
   observability KPIs).
3. Every recorded event has a non-null ``timestamp`` and the session carries
   both a non-null ``started_at`` and ``ended_at`` so the Digester's
   cycle-time KPI (ADR-0007) has attributable timestamps.
4. No secret-shaped substring survives in any ``tool_result`` payload
   (``docs/SECURITY.md`` §Secrets, ADR-0003).

The test is gated behind ``FOUNDRY_RUN_LIVE_LLM=1`` because the local
``llama.cpp`` stack is not always available in CI; without the gate the
test would always fail on hosts that lack ``llama-server``. Per the issue's
"Phase-3 evidence rule", the PR author MUST run the live test before merge
and paste the captured ``outcome`` trace event JSON into the PR body.

References:

* ADR-0010 §Consequences — Runner agent loop event vocabulary
* ``docs/SECURITY.md`` §Secrets — trace sanitization contract
* ``docs/ROADMAP.md`` Phase 3 — "real-LLM benchmark runs" headline gap
* ``infra/llama-cpp/README.md`` — how to bring ``llama-server`` up locally
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.harness_layout import validate as validate_harness_layout
from foundry_x.execution.runner import (
    DEFAULT_TASK_TIMEOUT_S,
    RunLimits,
    resolve_harness_version,
    run_task,
    run_with_limits,
)
from foundry_x.trace.logger import TraceLogger, _redact

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "harness"
DEFAULT_TRACE_PATH = REPO_ROOT / "logs" / "traces.db"

#: Literal values that enable the live LLM smoke. Mirrors how other
#: infrastructure gates handle booleans (``true`` / ``yes`` / ``on`` in
#: addition to the canonical ``1``); callers using ``FOUNDRY_RUN_LIVE_LLM=1``
#: get the canonical behaviour, and ``=true`` / ``=yes`` are accepted as
#: user-friendly aliases.
_LIVE_LLM_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})

#: Mirror of the ``sort_a_list`` benchmark task's prompt (issue #112). The
#: exact wording is not asserted here -- the live test is plumbing-focused --
#: but the prompt must reach the model through ``run_task`` so the trace
#: carries the corresponding ``user_prompt`` event.
SORT_A_LIST_PROMPT: str = (
    "Read space-separated integers from input.txt, sort them ascending, "
    "and write the result space-separated to output.txt."
)


def _live_llm_enabled(env: dict[str, str] | None = None) -> bool:
    """Return ``True`` iff ``FOUNDRY_RUN_LIVE_LLM`` opts into the live run."""
    source = env if env is not None else os.environ
    return source.get("FOUNDRY_RUN_LIVE_LLM", "").strip().lower() in _LIVE_LLM_TOKENS


TASK = BenchmarkTask(
    name="real_llm_smoke",
    description=(
        "Phase-3 plumbing canary: Runner.run_task drives the agent loop against "
        "the live llama-server endpoint using the sort_a_list prompt, and the "
        "trace store records one TraceSession with a non-null harness_version, "
        "every event has a non-null timestamp, the terminal outcome.status is "
        "valid, and no secret-shaped substring survives in any tool_result "
        "payload (SECURITY.md §Secrets)."
    ),
    prompt=(
        "Drive Runner.run_task against the sort_a_list prompt with the live "
        "llama-server endpoint (LLAMACPP_HOST) and verify the plumbing "
        "artifacts described in expected_outcome."
    ),
    expected_outcome=(
        "Runner.run_task completes with any of outcome.status in "
        "{success, truncated, failed} -- plumbing only, model quality is out "
        "of scope (PRD §5). The trace database at the configured path "
        "(FOUNDRY_TRACE_PATH or ./logs/traces.db) contains exactly one "
        "TraceSession with a non-null harness_version, every recorded event "
        "has a non-null timestamp, the session has both a non-null started_at "
        "and ended_at, the terminal outcome.status is one of "
        "{success, truncated, failed}, and no secret-shaped substring "
        "survives in any tool_result payload after TraceLogger redaction."
    ),
    difficulty_tier="medium",
    timeout_seconds=300,
    tags=["agent-loop", "phase-3"],
)


@pytest.mark.benchmark
@pytest.mark.skipif(
    not _live_llm_enabled(),
    reason=(
        "FOUNDRY_RUN_LIVE_LLM not set to 1 (or true/yes/on); real-LLM smoke is "
        "opt-in because the local llama-server stack is not always available "
        "in CI (issue #175). Set FOUNDRY_RUN_LIVE_LLM=1 to enable."
    ),
)
def test_real_llm_smoke(
    llamacpp_server: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Phase-3 plumbing canary for the live llama-server path (issue #175).

    Drives :func:`run_task` with the sort_a_list prompt and the
    OpenAI-compatible ``llama-server`` endpoint (per ``LLAMACPP_HOST`` /
    ``OPENCODE_SERVER_URL``), then asserts the trace-store plumbing:

    * the trace database is created under ``logs/`` (or
      ``FOUNDRY_TRACE_PATH``),
    * exactly one :class:`TraceSession` is recorded with a non-null
      ``harness_version``, ``started_at``, and ``ended_at``,
    * every recorded event has a non-null ``timestamp``,
    * the terminal ``outcome.status`` is one of
      ``{"success", "truncated", "failed"}``,
    * no secret-shaped substring survives in any ``tool_result`` payload
      (the trace logger redacts at write time; re-running ``_redact`` on
      the stored payload must therefore be a no-op).
    """
    if not HARNESS_DIR.is_dir():
        pytest.skip(f"harness directory {HARNESS_DIR} is not present")
    try:
        validate_harness_layout(HARNESS_DIR)
    except Exception as exc:  # HarnessValidationError or OSError
        pytest.skip(f"harness layout at {HARNESS_DIR} is invalid: {exc}")

    # Honour the runner's documented default so the test mirrors `main()`.
    trace_path = Path(os.environ.get("FOUNDRY_TRACE_PATH", str(DEFAULT_TRACE_PATH))).resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    # Fresh trace per run: list_sessions would otherwise return stale rows
    # from earlier opt-in runs and break the "exactly one TraceSession"
    # assertion below.
    if trace_path.exists():
        trace_path.unlink()

    # Mirror main(): prepend the resolved harness dir to sys.path so the
    # prompt-input firewall (harness/hooks/__init__.py) self-registers.
    # Cleanup runs in the finally block even if run_task raises.
    harness_str = str(HARNESS_DIR.resolve())
    inserted = False
    if harness_str not in sys.path:
        sys.path.insert(0, harness_str)
        inserted = True

    logger = TraceLogger(trace_path)
    limits = RunLimits(
        task_timeout_s=TASK.timeout_seconds
        if TASK.timeout_seconds is not None
        else DEFAULT_TASK_TIMEOUT_S
    )
    harness_version = resolve_harness_version(HARNESS_DIR) or "real_llm_smoke"

    try:
        with logger.session(
            harness_version=harness_version, model_id="real_llm_smoke"
        ) as session_id:
            asyncio.run(
                run_with_limits(
                    run_task(SORT_A_LIST_PROMPT, HARNESS_DIR, logger, session_id),
                    logger,
                    session_id,
                    limits,
                )
            )
    finally:
        if inserted and harness_str in sys.path:
            sys.path.remove(harness_str)

    # --- Plumbing assertions (issue #175 acceptance criteria) ------------

    assert trace_path.exists(), f"trace database was not created at {trace_path}"

    sessions = logger.list_sessions()
    assert (
        len(sessions) == 1
    ), f"expected exactly one TraceSession, got {len(sessions)}: {sessions!r}"
    session = sessions[0]
    assert session.harness_version, (
        "TraceSession.harness_version is null/empty -- the trace is not "
        "attributable to a harness revision (ADR-0007)"
    )
    assert session.started_at, (
        "TraceSession.started_at is null/empty -- the cycle-time KPI "
        "(PRD §5) needs an attributable start timestamp"
    )
    assert session.ended_at, (
        "TraceSession.ended_at is null/empty -- the runaway-detection "
        "guardrail (SECURITY.md) needs the session wall-clock close"
    )

    events = list(logger.iter_events(session.session_id))
    assert events, "no events were recorded for the session"

    for event in events:
        assert event.timestamp, (
            f"event {event.event_id} (kind={event.kind!r}) has a null/empty "
            "timestamp -- every event must be attributable on the timeline"
        )

    outcome_events = [event for event in events if event.kind == "outcome"]
    assert (
        len(outcome_events) == 1
    ), f"expected exactly one outcome event, got {len(outcome_events)}"
    status = outcome_events[0].payload.get("status")
    assert status in {"success", "truncated", "failed"}, (
        f"unexpected outcome.status={status!r}; the Digester (ADR-0007) "
        "only buckets these three values"
    )

    # Secret-shape invariant: the trace logger scrubs at write time
    # (SECURITY.md §Secrets), so re-running _redact on the stored payload
    # must be a no-op. A regression that drops a regex, weakens a key name,
    # or skips the recursive metadata walk surfaces here as a literal
    # secret-shaped substring surviving in the tool_result payload.
    tool_result_events = [event for event in events if event.kind == "tool_result"]
    for event in tool_result_events:
        scrubbed = _redact(event.payload)
        assert scrubbed == event.payload, (
            f"tool_result event {event.event_id} contains a secret-shaped "
            f"substring that should have been redacted at write time; "
            f"payload={event.payload!r}"
        )
