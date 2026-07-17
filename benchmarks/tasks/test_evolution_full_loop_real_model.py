"""Benchmark task: full evolution loop with real model via run_evolution_step (issue #484).

This test extends the Phase-3 plumbing canary (``test_real_llm_smoke.py``,
issue #175) to validate the complete evolution loop using
:func:`foundry_x.evolution.loop.run_evolution_step` -- the canonical entry
point that chains Digester → Evolver → Critic.

The test drives :func:`foundry_x.execution.runner.run_task` against the live
model endpoint (per ``LLAMACPP_HOST`` / ``OPENCODE_SERVER_URL``), captures
the trace, then runs the evolution loop through ``run_evolution_step`` and
asserts the complete pipeline:

1. The trace database is created with exactly one :class:`TraceSession` whose
   ``harness_version`` is non-null.
2. The terminal ``outcome.status`` is one of ``{"success", "truncated",
   "failed"}`` -- model quality is out of scope (PRD §5).
3. Every recorded event has a non-null ``timestamp``.
4. The session has both a non-null ``started_at`` and ``ended_at``.
5. No secret-shaped substring survives in any ``tool_result`` payload.
6. ``run_evolution_step`` returns an :class:`EvolutionResult` with a valid
   ``failure_report``.
7. The ``failure_report`` has a valid ``proposed_class``.
8. If edits are proposed, the Critic evaluates them and returns a verdict.

The test is gated behind ``FOUNDRY_RUN_LIVE_LLM=1`` because the local
llama.cpp stack is not always available in CI; without the gate the
test would always fail on hosts that lack ``llama-server``.

References:

* ADR-0004 §Consequences — Runner agent loop event vocabulary
* ``docs/SECURITY.md`` §Secrets — trace sanitization contract
* ``docs/ROADMAP.md`` Phase 3 — "real-LLM benchmark runs" headline gap
* ``infra/llama-cpp/README.md`` — how to bring ``llama-server`` up locally
* issue #484 — validate full loop with real model
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.evolution.loop import run_evolution_step
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

_LIVE_LLM_TOKENS: frozenset[str] = frozenset({"1", "true", "yes", "on"})

SORT_A_LIST_PROMPT: str = (
    "Read space-separated integers from input.txt, sort them ascending, "
    "and write the result space-separated to output.txt."
)


def _live_llm_enabled(env: dict[str, str] | None = None) -> bool:
    """Return ``True`` iff ``FOUNDRY_RUN_LIVE_LLM`` opts into the live run."""
    source = env if env is not None else os.environ
    return source.get("FOUNDRY_RUN_LIVE_LLM", "").strip().lower() in _LIVE_LLM_TOKENS


TASK = BenchmarkTask(
    name="evolution_full_loop_real_model",
    description=(
        "Full evolution loop via run_evolution_step: Runner.run_task drives the "
        "agent loop against the live model endpoint using the sort_a_list prompt, "
        "the trace store captures the session, then run_evolution_step processes "
        "the trace through Digester → Evolver → Critic and returns an EvolutionResult."
    ),
    prompt=(
        "Drive Runner.run_task against the sort_a_list prompt with the live "
        "model endpoint (LLAMACPP_HOST) and verify the full evolution loop "
        "using run_evolution_step: trace → Digester → Evolver → Critic."
    ),
    expected_outcome=(
        "Runner.run_task completes with any of outcome.status in "
        "{success, truncated, failed}. The trace database contains one "
        "TraceSession with a non-null harness_version. run_evolution_step "
        "returns an EvolutionResult with a valid failure_report. "
        "If edits are proposed, the Critic returns a verdict."
    ),
    difficulty_tier="medium",
    timeout_seconds=600,
    tags=["agent-loop", "phase-3", "evolution-loop", "full-loop"],
)


@pytest.mark.benchmark
@pytest.mark.skipif(
    not _live_llm_enabled(),
    reason=(
        "FOUNDRY_RUN_LIVE_LLM not set to 1 (or true/yes/on); real-model smoke "
        "is opt-in because the local llama-server stack is not always available "
        "in CI (issue #484). Set FOUNDRY_RUN_LIVE_LLM=1 to enable."
    ),
)
def test_evolution_full_loop_real_model() -> None:
    """Full evolution loop validation via run_evolution_step (issue #484).

    Drives :func:`run_task` with the sort_a_list prompt and the
    OpenAI-compatible ``llama-server`` endpoint (per ``LLAMACPP_HOST`` /
    ``OPENCODE_SERVER_URL``), captures the trace, then runs the evolution
    loop via :func:`run_evolution_step` and asserts:

    * the trace database is created under ``logs/`` (or
      ``FOUNDRY_TRACE_PATH``),
    * exactly one :class:`TraceSession` is recorded with a non-null
      ``harness_version``, ``started_at``, and ``ended_at``,
    * every recorded event has a non-null ``timestamp``,
    * the terminal ``outcome.status`` is one of
      ``{"success", "truncated", "failed"}``,
    * no secret-shaped substring survives in any ``tool_result`` payload,
    * ``run_evolution_step`` returns an :class:`EvolutionResult` with a
      valid ``failure_report``,
    * the ``failure_report`` has a valid ``proposed_class``,
    * if edits are proposed, the Critic returns a verdict with a boolean
      ``verdict`` attribute.
    """
    if not HARNESS_DIR.is_dir():
        pytest.skip(f"harness directory {HARNESS_DIR} is not present")
    try:
        validate_harness_layout(HARNESS_DIR)
    except Exception as exc:
        pytest.skip(f"harness layout at {HARNESS_DIR} is invalid: {exc}")

    trace_path = Path(os.environ.get("FOUNDRY_TRACE_PATH", str(DEFAULT_TRACE_PATH))).resolve()
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        trace_path.unlink()

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
    harness_version = resolve_harness_version(HARNESS_DIR) or "evolution_full_loop_real_model"

    try:
        with logger.session(
            harness_version=harness_version, model_id="evolution_full_loop_real_model"
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

    # --- Runner → trace plumbing assertions (issue #175) --------------------

    assert trace_path.exists(), f"trace database was not created at {trace_path}"

    sessions = logger.list_sessions()
    assert len(sessions) == 1, (
        f"expected exactly one TraceSession, got {len(sessions)}: {sessions!r}"
    )
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
    assert len(outcome_events) == 1, (
        f"expected exactly one outcome event, got {len(outcome_events)}"
    )
    status = outcome_events[0].payload.get("status")
    assert status in {"success", "truncated", "failed"}, (
        f"unexpected outcome.status={status!r}; the Digester (ADR-0007) "
        "only buckets these three values"
    )

    tool_result_events = [event for event in events if event.kind == "tool_result"]
    for event in tool_result_events:
        scrubbed = _redact(event.payload)
        assert scrubbed == event.payload, (
            f"tool_result event {event.event_id} contains a secret-shaped "
            f"substring that should have been redacted at write time; "
            f"payload={event.payload!r}"
        )

    # --- trace → run_evolution_step assertions (issue #484) ---------------

    result = run_evolution_step(session.session_id, events, HARNESS_DIR)

    assert result.session_id == session.session_id, (
        "EvolutionResult.session_id must match the trace session_id"
    )
    assert result.failure_report is not None, "EvolutionResult.failure_report must not be None"
    assert result.failure_report.session_id == session.session_id, (
        "FailureReport.session_id must match the trace session_id"
    )
    assert result.failure_report.proposed_class in {
        "clean",
        "wrong-tool",
        "bad-prompt",
        "state-leak",
        "tool-error",
        "injection-attempt",
    }, f"unexpected proposed_class={result.failure_report.proposed_class!r}"

    if result.failure_report.proposed_class != "clean":
        assert result.proposed_edits is not None, (
            "EvolutionResult.proposed_edits must not be None when report is not clean"
        )

        if result.proposed_edits:
            assert result.verdict is not None, (
                "EvolutionResult.verdict must not be None when edits are proposed"
            )
            assert hasattr(result.verdict, "verdict"), (
                "CriticVerdict must have a 'verdict' attribute (bool)"
            )
            assert isinstance(result.verdict.verdict, bool), (
                f"CriticVerdict.verdict must be bool, got {type(result.verdict.verdict)}"
            )
