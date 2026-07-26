"""Benchmark task: assert pruning preserves tool_result and user_prompt (issue #929).

Exercises ``ContextPruningHook`` in the Runner loop and asserts the
``_PRESERVE_KINDS`` invariant: events whose ``kind`` is ``tool_result`` or
``user_prompt`` survive pruning, while droppable noise (``model_request``) is
reduced by exactly the ``dropped`` count recorded on the ``context_pruned``
event.

The two existing pruning benchmarks
(``test_context_pruning_benchmark.py`` #618,
``test_token_aware_pruning_benchmark.py`` #732) plant ONLY non-preserved
event kinds and assert ``dropped >= 1``. Their ``_plant`` helpers explicitly
document "None of the planted events use tool_result or user_prompt" -- they
cover the DROP path but never the PRESERVE path. A grep for
``_PRESERVE_KINDS`` across ``tests/`` and ``benchmarks/`` returned zero
matches before this task, so an Evolver-produced diff that removed
``tool_result`` / ``user_prompt`` from ``_PRESERVE_KINDS``
(``harness/hooks/context_pruning.py:68``) would silently degrade agent
context with no benchmark catching the regression. This task closes that gap
(ADR-0004 -- Critic gate gains a regression target for the preservation
invariant; ADR-0005 -- pytest-based regression target).
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.execution.model_adapter import (
    ModelMessage,
    ModelResponse,
    ModelResponseChunk,
    ModelToolCall,
    ModelToolCallChunk,
    ToolCallFunction,
    ToolCallFunctionChunk,
)
from foundry_x.execution.runner import run_task
from foundry_x.trace.logger import TraceLogger


TASK = BenchmarkTask(
    name="prune_preserves_kinds",
    description=(
        "Drive Runner.run_task with ContextPruningHook and interleaved "
        "tool_result/user_prompt/model_request events; assert the preserved "
        "kinds survive pruning while model_request noise is reduced by the "
        "dropped count (issue #929)."
    ),
    tags=["agent-loop", "context-pruning", "preserve-kinds"],
    difficulty_tier="easy",
)

# Hook fires when the session's event count exceeds ``_THRESHOLD``. The plant
# interleaves four slots per cycle: ``tool_result`` (preserved),
# ``user_prompt`` (preserved), and two ``model_request`` (droppable noise).
# With ``_PLANTED=72`` that yields 18 tool_result, 18 user_prompt, and 36
# model_request events.
#
# Sizing rationale for the exact criterion-4 assertion
# (``NOISE_PLANTED - surviving_noise == dropped``): the pruner deletes only
# non-preserved events, oldest first. The planted events are older than any
# runner-emitted event (they are recorded before ``run_task`` starts), so the
# hook's first ``pre_tool`` deletion set is drawn entirely from the planted
# ``model_request`` events as long as ``dropped <= NOISE_PLANTED``. The
# runner records ~4 events before ``pre_tool`` fires (``user_prompt`` plus
# ``model_request`` / ``token_usage_missing`` / ``model_response``), so the
# session holds ~76 events and the hook drops ~26. Because
# ``preserved_planted (36) + runner_events (~4) = 40 <= _THRESHOLD (50)``,
# ``dropped`` (26) stays well below ``NOISE_PLANTED`` (36) -- a margin of 10
# that absorbs runner event-count drift.
_THRESHOLD = 50
_PLANTED = 72
_NOISE_KIND = "model_request"
_NOISE_MARKER = "noise"
_PRESERVE_MARKER = "preserve"


class _StubAdapter:
    """Stub ``ModelAdapter`` that emits one tool_call then a final answer.

    Same shape as ``_StubAdapter`` in ``test_context_pruning_benchmark``; one
    step is enough to trigger the hook's first ``pre_tool`` and fire pruning.
    """

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tools=None, **kwargs):  # noqa: ANN001, ARG002
        self.calls += 1
        if self.calls == 1:
            tool_call = ModelToolCall(
                id="call_read",
                type="function",
                function=ToolCallFunction(
                    name="read_file",
                    arguments=json.dumps({"path": "/tmp/x"}),
                ),
            )
            return ModelResponse(
                message=ModelMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                ),
                tool_calls=[tool_call],
                finish_reason="tool_calls",
            )
        if self.calls == 2:
            return ModelResponse(
                message=ModelMessage(role="assistant", content="done"),
                finish_reason="stop",
            )
        raise RuntimeError(
            f"_StubAdapter exhausted after 2 scripted responses; loop called "
            f"complete() {self.calls} times"
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


def _stub_harness(harness_dir: Path) -> Path:
    """Build a minimal valid harness layout under ``harness_dir``."""
    harness_dir.mkdir(parents=True, exist_ok=True)
    (harness_dir / "system_prompt.txt").write_text("stub harness for prune_preserves_kinds\n")
    (harness_dir / "hooks").mkdir(exist_ok=True)
    (harness_dir / "skills").mkdir(exist_ok=True)
    return harness_dir


def _sqlite_pruner(db_path: Path):  # noqa: ANN401
    """Build a ``Pruner`` callable backed by direct SQLite.

    Mirrors the implementation in ``tests/harness/test_context_pruning.py`` and
    the sibling pruning benchmarks.
    """

    def _drop(session_id: str, keep_kinds: frozenset[str], target_count: int) -> int:
        not_in_clause = ", ".join("?" for _ in keep_kinds)
        with sqlite3.connect(db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM events WHERE session_id = ?",
                (session_id,),
            ).fetchone()[0]
            if total <= target_count:
                return 0
            to_drop = total - target_count
            params: list[object] = [session_id, *keep_kinds, to_drop]
            cursor = conn.execute(
                "SELECT event_id FROM events "
                "WHERE session_id = ? AND kind NOT IN (" + not_in_clause + ") "
                "ORDER BY timestamp LIMIT ?",
                params,
            )
            ids = [row[0] for row in cursor.fetchall()]
            if not ids:
                return 0
            placeholders = ", ".join("?" for _ in ids)
            conn.execute(
                "DELETE FROM events WHERE event_id IN (" + placeholders + ")",
                ids,
            )
            return len(ids)

    return _drop


def _plant(logger: TraceLogger, session_id: str, n: int) -> None:
    """Plant ``n`` interleaved preserved + noise events on ``session_id``.

    Every 4-event cycle carries one ``tool_result`` (preserved), one
    ``user_prompt`` (preserved), and two ``model_request`` (droppable noise).
    The preserved events occupy the OLDEST indices on purpose: a pruner that
    ignored ``_PRESERVE_KINDS`` would delete them first (oldest-first deletion
    order), so this layout makes the preservation assertion a genuine
    regression target. Each event carries a ``marker`` payload so the post-run
    analysis can distinguish planted events from runner-emitted ones.
    """
    for i in range(n):
        role = i % 4
        if role == 0:
            kind = "tool_result"
            payload: dict[str, object] = {"index": i, "marker": _PRESERVE_MARKER}
        elif role == 1:
            kind = "user_prompt"
            payload = {"index": i, "marker": _PRESERVE_MARKER}
        else:
            kind = _NOISE_KIND
            payload = {"index": i, "marker": _NOISE_MARKER}
        logger.record(session_id, kind=kind, payload=payload)


def _install_on_error_tracker(tracker):  # noqa: ANN001
    """Install ``tracker`` on the default ``HookRegistry`` for the test."""
    from harness.hooks import get_registry
    from harness.hooks.base import reset_default_registry

    reset_default_registry()
    registry = get_registry()
    registry._on_error = tracker  # type: ignore[assignment]
    return registry


@pytest.mark.benchmark
def test_prune_preserves_kinds(benchmark_workspace: Path) -> None:
    """Pruning preserves tool_result and user_prompt events (issue #929).

    Drives ``Runner.run_task`` with a stub ``ModelAdapter`` (one tool_call
    then final answer) and ``ContextPruningHook`` registered on the default
    registry. Before the run, plants ``_PLANTED`` interleaved
    tool_result / user_prompt / model_request events into the trace database
    so the hook's first ``pre_tool`` call fires pruning.

    Asserts the four behavioural acceptance criteria:
    1. ``tool_result`` events are present after pruning (fully preserved).
    2. ``user_prompt`` events are present after pruning (fully preserved).
    3. ``dropped`` count > 0 (pruning actually fired).
    4. ``model_request`` noise is reduced by exactly the ``dropped`` count.
    """
    db = benchmark_workspace / "traces.db"
    harness_dir = benchmark_workspace / "harness"
    _stub_harness(harness_dir)

    hook_failures: list[tuple[str, int, str, str]] = []

    def _track_failure(slot: str, index: int, name: str, exc: BaseException) -> None:
        hook_failures.append((slot, index, name, repr(exc)))

    registry = _install_on_error_tracker(_track_failure)

    try:
        adapter = _StubAdapter()
        pruner = _sqlite_pruner(db)

        async def _drive(registry: Any) -> None:
            logger = TraceLogger(db)
            with logger.session(harness_version="0.1.0") as sid:
                _plant(logger, sid, _PLANTED)

                def _tracer(sid: str, kind: str, payload: dict) -> None:
                    logger.record(sid, kind=kind, payload=payload)

                from harness.hooks.context_pruning import ContextPruningHook

                hook = ContextPruningHook(
                    session_id=sid,
                    threshold=_THRESHOLD,
                    pruner=pruner,
                    tracer=_tracer,
                )
                registry.register(hook)

                await run_task(
                    "prune-preserves-kinds",
                    harness_dir,
                    logger,
                    sid,
                    model_adapter=adapter,
                )

        asyncio.run(_drive(registry))

        # --- Trace analysis ------------------------------------------------
        logger = TraceLogger(db)
        sid = logger.list_sessions()[0].session_id
        events = logger.load_session(sid)

        # --- context_pruned event -----------------------------------------
        prune_events = [e for e in events if e.kind == "context_pruned"]
        assert len(prune_events) == 1, (
            f"expected exactly 1 context_pruned event; got {len(prune_events)}: "
            f"{[e.payload for e in prune_events]!r}"
        )
        dropped = prune_events[0].payload["dropped"]

        # Criterion 3: pruning actually fired.
        assert dropped >= 1, f"expected dropped >= 1 (hook must fire); got {dropped!r}"

        # --- planted-event populations ------------------------------------
        # Filter by marker so runner-emitted events of the same kind do not
        # pollute the planted counts.
        preserved_tr = [
            e
            for e in events
            if e.kind == "tool_result" and e.payload.get("marker") == _PRESERVE_MARKER
        ]
        preserved_up = [
            e
            for e in events
            if e.kind == "user_prompt" and e.payload.get("marker") == _PRESERVE_MARKER
        ]
        noise = [
            e for e in events if e.kind == _NOISE_KIND and e.payload.get("marker") == _NOISE_MARKER
        ]

        preserved_each = _PLANTED // 4  # 18 tool_result + 18 user_prompt
        noise_planted = _PLANTED // 2  # 36 model_request

        # Criteria 1 & 2: preserved kinds survive pruning fully intact. An
        # Evolver diff that dropped tool_result/user_prompt from
        # _PRESERVE_KINDS would delete these (they sit at the oldest indices)
        # and fail these equality assertions.
        assert len(preserved_tr) == preserved_each, (
            f"expected all {preserved_each} planted tool_result events to "
            f"survive pruning; got {len(preserved_tr)} (dropped={dropped})"
        )
        assert len(preserved_up) == preserved_each, (
            f"expected all {preserved_each} planted user_prompt events to "
            f"survive pruning; got {len(preserved_up)} (dropped={dropped})"
        )

        # Criterion 4: noise reduced by exactly the dropped count. The pruner
        # deletes only non-preserved events oldest-first, and the planted
        # noise events are older than any runner event, so the deletion set is
        # drawn entirely from planted model_request events (sized so
        # ``dropped <= noise_planted`` -- see module docstring).
        assert noise_planted - len(noise) == dropped, (
            f"expected model_request noise reduced by exactly dropped "
            f"({dropped}); planted {noise_planted}, surviving {len(noise)}, "
            f"reduction {noise_planted - len(noise)}"
        )

        # --- outcome (sanity) ---------------------------------------------
        outcome_event = next(e for e in events if e.kind == "outcome")
        assert outcome_event.payload["status"] == "success", (
            f"expected outcome.status='success'; got {outcome_event.payload!r}"
        )
        assert outcome_event.payload["reason"] == "final_answer"

        # --- Hook isolation ------------------------------------------------
        assert hook_failures == [], (
            f"expected zero HookRegistry._on_error calls; got {hook_failures!r}"
        )

        # --- Adapter exhaustion guard --------------------------------------
        assert adapter.calls == 2, (
            f"expected exactly 2 model round-trips (tool_call then "
            f"final_answer); got {adapter.calls}"
        )
    finally:
        from harness.hooks.base import reset_default_registry

        del registry
        reset_default_registry()
