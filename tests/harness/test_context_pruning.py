"""Tests for ``harness/hooks/context_pruning.py`` (issue #106).

Acceptance criteria from the issue body:

* ``harness/hooks/context_pruning.py`` exists implementing the Hook
  protocol; its ``pre_tool`` method, when the running session's
  accumulated event count exceeds a configurable threshold (default
  200), drops the oldest non-``tool_result`` / non-``user_prompt``
  events and records a ``context_pruned`` trace event with the
  dropped count.
* Writes 250 synthetic TraceEvents into a tmp_path TraceLogger, runs
  the hook once, asserts the post-prune count is ``<= threshold`` and
  that a ``context_pruned`` event was recorded with the correct
  dropped count.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3

import pytest

from foundry_x.trace.logger import TraceLogger
from harness.hooks.base import ToolCall
from harness.hooks.context_pruning import (
    DEFAULT_THRESHOLD,
    ContextPruningHook,
    Pruner,
    Tracer,
    TokenAwarePruningHook,
    register_into,
    register_token_aware_into,
    resolve_token_threshold,
)
from harness.hooks.base import HookRegistry


_PLANTS = 250


def _sqlite_pruner(db_path) -> Pruner:
    """Build a ``Pruner`` callable backed by direct SQLite.

    The hook is decoupled from :class:`TraceLogger` (AGENTS.md §7
    self-reference loop), so the test wires a minimal SQLite closure
    rather than expanding ``TraceLogger`` with a per-event delete
    method (out of scope for issue #106). The closure enforces the
    same contract documented on :data:`harness.hooks.context_pruning.Pruner`:
    drop the oldest events whose ``kind`` is not in ``keep_kinds``
    until the session's event count is at most ``target_count``, and
    return the number actually dropped.
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


def _tracer_for(logger: TraceLogger, session_id: str) -> tuple[Tracer, list[dict]]:
    """Build a ``Tracer`` callable that records into ``logger``.

    Returns ``(tracer, captured)``; ``captured`` accumulates every
    payload the tracer was asked to record so tests can assert the
    ``context_pruned`` payload shape without re-reading the trace.
    """

    captured: list[dict] = []

    def _record(_sid: str, kind: str, payload: dict) -> None:
        captured.append({"session_id": _sid, "kind": kind, "payload": dict(payload)})
        logger.record(session_id, kind=kind, payload=payload)

    return _record, captured


def _plant(logger: TraceLogger, session_id: str, n: int) -> None:
    """Plant ``n`` synthetic events on ``session_id``.

    None of the planted events use ``tool_result`` or ``user_prompt``,
    so every one is eligible for pruning and the post-prune math is
    unambiguous: ``n - DEFAULT_THRESHOLD`` events are dropped.
    """
    kinds = (
        "tool_call",
        "task_received",
        "model_request",
        "model_response",
        "critic_verdict",
    )
    for i in range(n):
        logger.record(
            session_id,
            kind=kinds[i % len(kinds)],
            payload={"index": i, "marker": "synthetic"},
        )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Acceptance: 250 events → prune → <= threshold + context_pruned recorded
# ---------------------------------------------------------------------------


def test_pre_tool_prunes_when_over_threshold(tmp_path) -> None:
    """250 planted events, one ``pre_tool`` call: post-prune count of
    non-bookkeeping events must be ``<= threshold`` and the
    ``context_pruned`` event must record the exact dropped count."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )

        _run(hook.pre_tool(ToolCall(name="read_file", arguments={"path": "/tmp/x"})))

        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    real_events = [e for e in surviving if e.kind != "context_pruned"]
    assert len(real_events) <= DEFAULT_THRESHOLD
    assert len(real_events) == DEFAULT_THRESHOLD
    assert len(prune_events) == 1
    prune_event = prune_events[0]
    assert prune_event.payload == {
        "dropped": _PLANTS - DEFAULT_THRESHOLD,
        "threshold": DEFAULT_THRESHOLD,
    }
    assert captured == [
        {
            "session_id": sid,
            "kind": "context_pruned",
            "payload": {
                "dropped": _PLANTS - DEFAULT_THRESHOLD,
                "threshold": DEFAULT_THRESHOLD,
            },
        }
    ]


def test_pre_tool_does_not_prune_when_under_threshold(tmp_path) -> None:
    """When the session is at or below the threshold the hook must be a
    no-op: no ``context_pruned`` event is recorded, no events are
    dropped, and the original ``ToolCall`` is returned unchanged."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, DEFAULT_THRESHOLD - 1)
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )
        call = ToolCall(name="read_file", arguments={"path": "/tmp/x"})

        returned = _run(hook.pre_tool(call))
        surviving = logger.load_session(sid)

    assert returned is call
    assert len(surviving) == DEFAULT_THRESHOLD - 1
    assert all(e.kind != "context_pruned" for e in surviving)
    assert captured == []


def test_pre_tool_preserves_tool_result_and_user_prompt(tmp_path) -> None:
    """``tool_result`` and ``user_prompt`` events must never be dropped,
    even when the protected set alone exceeds the threshold. The hook
    drops as many prunable events as it can; if all prunable events
    are exhausted, the count stays above threshold but every protected
    event survives intact."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        for _ in range(150):
            logger.record(sid, kind="tool_result", payload={"i": 0})
        for _ in range(50):
            logger.record(sid, kind="user_prompt", payload={"i": 0})
        for _ in range(100):
            logger.record(sid, kind="tool_call", payload={"i": 0})
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    real_events = [e for e in surviving if e.kind != "context_pruned"]
    kinds = [e.kind for e in real_events]
    assert kinds.count("tool_result") == 150
    assert kinds.count("user_prompt") == 50
    assert kinds.count("tool_call") == 0
    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 1
    assert prune_events[0].payload == {
        "dropped": 100,
        "threshold": DEFAULT_THRESHOLD,
    }
    assert captured[0]["payload"] == {
        "dropped": 100,
        "threshold": DEFAULT_THRESHOLD,
    }


def test_pre_tool_leaves_session_above_threshold_when_all_protected(tmp_path) -> None:
    """When the prunable pool is empty, the hook leaves the session above
    threshold rather than dropping a protected event. The contract is:
    drop as many prunable events as needed; never drop protected ones.
    A subsequent hook invocation with a still-empty prunable pool must
    return ``dropped=0`` and record no new ``context_pruned`` event."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        for _ in range(DEFAULT_THRESHOLD + 100):
            logger.record(sid, kind="tool_result", payload={"i": 0})
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    assert all(e.kind == "tool_result" for e in surviving)
    assert len(surviving) == DEFAULT_THRESHOLD + 100
    assert captured == []


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_constructor_rejects_invalid_threshold(tmp_path) -> None:
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        pass
    with pytest.raises(ValueError, match="threshold must be >= 1"):
        ContextPruningHook(
            session_id=sid,
            threshold=0,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
        )


def test_constructor_rejects_empty_session_id(tmp_path) -> None:
    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        ContextPruningHook(
            session_id="",
            threshold=DEFAULT_THRESHOLD,
            pruner=lambda *a, **k: 0,
            tracer=lambda *a, **k: None,
        )


# ---------------------------------------------------------------------------
# Critic-sandbox entry point: register_into(targeted_registry)
# ---------------------------------------------------------------------------


def test_register_into_installs_into_targeted_registry(tmp_path) -> None:
    """``register_into(registry, ...)`` must install the hook into the
    supplied registry and not the process default (ADR-0004)."""
    from harness.hooks.base import get_registry

    default_before = get_registry()
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        fresh = HookRegistry()
        hook = register_into(
            fresh,
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
        )

        assert hook in fresh._hooks  # noqa: SLF001 — internal inspection only here
        assert hook not in default_before._hooks  # noqa: SLF001
        assert hook.session_id == sid
        assert hook.threshold == DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Hook protocol conformance
# ---------------------------------------------------------------------------


def test_post_tool_is_pass_through(tmp_path) -> None:
    """``post_tool`` must return the result untouched. The pruning check
    runs in ``pre_tool`` only; doubling it on every result would burn
    an extra query per call without buying anything on a long session."""
    from harness.hooks.base import ToolResult

    hook = ContextPruningHook(
        session_id="any",
        threshold=DEFAULT_THRESHOLD,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
    )
    call = ToolCall(name="read_file", arguments={})
    result = ToolResult(name="read_file", output="hello")

    returned = _run(hook.post_tool(call, result))
    assert returned is result


def test_payload_is_json_round_tripable(tmp_path) -> None:
    """The ``context_pruned`` payload follows the existing TraceEvent
    payload pattern (ADR-0006): it must round-trip through JSON so the
    Digester can read it back without special-casing."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        pruner = _sqlite_pruner(db)
        tracer, _ = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_event = next(e for e in surviving if e.kind == "context_pruned")
    encoded = prune_event.model_dump_json()
    decoded = json.loads(encoded)
    assert decoded["kind"] == "context_pruned"
    assert decoded["payload"]["dropped"] == _PLANTS - DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# TokenAwarePruningHook tests (issue #465)
# ---------------------------------------------------------------------------


def test_token_aware_prunes_when_over_threshold(tmp_path) -> None:
    """TokenAwarePruningHook prunes when get_tokens() exceeds token_threshold."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)

        def _get_tokens(s: str) -> int:
            return 50_000

        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=40_000,
            get_tokens=_get_tokens,
            pruner=pruner,
            tracer=tracer,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 1
    prune_payload = prune_events[0].payload
    assert prune_payload["dropped"] == _PLANTS - DEFAULT_THRESHOLD
    assert prune_payload["threshold_tokens"] == 40_000
    assert prune_payload["session_tokens"] == 50_000
    assert captured[0]["payload"] == prune_payload


def test_token_aware_does_not_prune_when_under_threshold(tmp_path) -> None:
    """When session_tokens <= token_threshold, no pruning occurs."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)

        def _get_tokens(s: str) -> int:
            return 30_000

        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=40_000,
            get_tokens=_get_tokens,
            pruner=pruner,
            tracer=tracer,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    assert len(surviving) == _PLANTS
    assert all(e.kind != "context_pruned" for e in surviving)
    assert captured == []


def test_token_aware_preserves_tool_result_and_user_prompt(tmp_path) -> None:
    """TokenAwarePruningHook never drops tool_result or user_prompt."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        for _ in range(150):
            logger.record(sid, kind="tool_result", payload={"i": 0})
        for _ in range(50):
            logger.record(sid, kind="user_prompt", payload={"i": 0})
        for _ in range(100):
            logger.record(sid, kind="tool_call", payload={"i": 0})
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)

        def _get_tokens(s: str) -> int:
            return 80_000

        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=40_000,
            get_tokens=_get_tokens,
            pruner=pruner,
            tracer=tracer,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    real_events = [e for e in surviving if e.kind != "context_pruned"]
    kinds = [e.kind for e in real_events]
    assert kinds.count("tool_result") == 150
    assert kinds.count("user_prompt") == 50
    assert kinds.count("tool_call") == 0
    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 1
    assert prune_events[0].payload["dropped"] == 100
    assert prune_events[0].payload["threshold_tokens"] == 40_000
    assert prune_events[0].payload["session_tokens"] == 80_000


def test_token_aware_payload_is_json_round_tripable(tmp_path) -> None:
    """TokenAwarePruningHook context_pruned payload round-trips through JSON."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        pruner = _sqlite_pruner(db)
        tracer, _ = _tracer_for(logger, sid)

        def _get_tokens(s: str) -> int:
            return 60_000

        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=30_000,
            get_tokens=_get_tokens,
            pruner=pruner,
            tracer=tracer,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_event = next(e for e in surviving if e.kind == "context_pruned")
    encoded = prune_event.model_dump_json()
    decoded = json.loads(encoded)
    assert decoded["kind"] == "context_pruned"
    assert decoded["payload"]["threshold_tokens"] == 30_000
    assert decoded["payload"]["session_tokens"] == 60_000
    assert "dropped" in decoded["payload"]


def test_token_aware_constructor_rejects_invalid_threshold(tmp_path) -> None:
    """token_threshold must be >= 1."""
    with pytest.raises(ValueError, match="token_threshold must be >= 1"):
        TokenAwarePruningHook(
            session_id="any",
            token_threshold=0,
            get_tokens=lambda s: 0,
            pruner=lambda *a, **k: 0,
            tracer=lambda *a, **k: None,
        )


def test_token_aware_post_tool_is_pass_through(tmp_path) -> None:
    """post_tool returns the result untouched (same contract as ContextPruningHook)."""
    hook = TokenAwarePruningHook(
        session_id="any",
        token_threshold=40_000,
        get_tokens=lambda s: 30_000,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
    )
    call = ToolCall(name="read_file", arguments={})
    from harness.hooks.base import ToolResult

    result = ToolResult(name="read_file", output="hello")
    returned = _run(hook.post_tool(call, result))
    assert returned is result


def test_resolve_token_threshold_returns_int() -> None:
    """resolve_token_threshold returns the integer value when set."""
    value = resolve_token_threshold({"FOUNDRY_CONTEXT_TOKENS": "50000"})
    assert value == 50_000


def test_resolve_token_threshold_returns_none_when_absent() -> None:
    """resolve_token_threshold returns None when the env var is absent."""
    value = resolve_token_threshold({})
    assert value is None


def test_resolve_token_threshold_returns_none_when_empty() -> None:
    """resolve_token_threshold returns None when the env var is empty."""
    value = resolve_token_threshold({"FOUNDRY_CONTEXT_TOKENS": ""})
    assert value is None


def test_resolve_token_threshold_rejects_non_positive() -> None:
    """resolve_token_threshold raises ValueError for non-positive values."""
    with pytest.raises(ValueError, match="FOUNDRY_CONTEXT_TOKENS must be a positive integer"):
        resolve_token_threshold({"FOUNDRY_CONTEXT_TOKENS": "0"})
    with pytest.raises(ValueError, match="FOUNDRY_CONTEXT_TOKENS must be a positive integer"):
        resolve_token_threshold({"FOUNDRY_CONTEXT_TOKENS": "-100"})


def test_register_token_aware_into_installs_into_targeted_registry(tmp_path) -> None:
    """register_token_aware_into installs TokenAwarePruningHook into the registry."""
    default_before = HookRegistry()
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        fresh = HookRegistry()
        hook = register_token_aware_into(
            fresh,
            session_id=sid,
            token_threshold=40_000,
            get_tokens=lambda s: 30_000,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
        )
        assert isinstance(hook, TokenAwarePruningHook)
        assert hook in fresh._hooks  # noqa: SLF001
        assert hook not in default_before._hooks  # noqa: SLF001
        assert hook.session_id == sid
        assert hook.token_threshold == 40_000
