"""Tests for ``harness/hooks/context_pruning.py`` (issue #106, #465).

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

Token-aware pruning (issue #465):

* ``TokenAwarePruningHook`` drops events when cumulative tokens exceed
  the configured threshold (``FOUNDRY_CONTEXT_TOKENS``).
* ``context_pruned`` payload includes ``threshold_tokens`` and
  ``session_tokens`` fields.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from foundry_x.trace.logger import TraceLogger
from harness.hooks.base import ToolCall, HookRegistry, ToolResult
from harness.hooks.context_pruning import (
    DEFAULT_THRESHOLD,
    DEFAULT_TOKEN_THRESHOLD,
    ContextPruningHook,
    Tracer,
    TokenAwarePruningHook,
    TokenCounter,
    _sqlite_pruner,
    register_into,
    register_token_aware_into,
    resolve_context_tokens_threshold,
)


_PLANTS = 250


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
        "token_threshold": DEFAULT_TOKEN_THRESHOLD,
    }
    assert captured == [
        {
            "session_id": sid,
            "kind": "context_pruned",
            "payload": {
                "dropped": _PLANTS - DEFAULT_THRESHOLD,
                "threshold": DEFAULT_THRESHOLD,
                "token_threshold": DEFAULT_TOKEN_THRESHOLD,
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
        "token_threshold": DEFAULT_TOKEN_THRESHOLD,
    }
    assert captured[0]["payload"] == {
        "dropped": 100,
        "threshold": DEFAULT_THRESHOLD,
        "token_threshold": DEFAULT_TOKEN_THRESHOLD,
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


def test_constructor_rejects_invalid_token_threshold(tmp_path) -> None:
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        pass
    with pytest.raises(ValueError, match="token_threshold must be >= 1"):
        ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            token_threshold=0,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
        )


def test_token_threshold_property() -> None:
    hook = ContextPruningHook(
        session_id="any",
        threshold=DEFAULT_THRESHOLD,
        token_threshold=4096,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
    )
    assert hook.token_threshold == 4096


def test_token_threshold_defaults_to_default() -> None:
    hook = ContextPruningHook(
        session_id="any",
        threshold=DEFAULT_THRESHOLD,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
    )
    assert hook.token_threshold == DEFAULT_TOKEN_THRESHOLD


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
            token_threshold=DEFAULT_TOKEN_THRESHOLD,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
        )

        assert hook in fresh._hooks  # noqa: SLF001 — internal inspection only here
        assert hook not in default_before._hooks  # noqa: SLF001
        assert hook.session_id == sid
        assert hook.threshold == DEFAULT_THRESHOLD
        assert hook.token_threshold == DEFAULT_TOKEN_THRESHOLD


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
# Token-aware pruning tests (issue #465)
# ---------------------------------------------------------------------------


def _fake_token_counter(token_map: dict[str, int]) -> TokenCounter:
    """Build a TokenCounter that returns values from a dict."""
    return lambda sid: token_map.get(sid, 0)


def test_token_aware_prunes_when_over_token_threshold(tmp_path) -> None:
    """When session_tokens exceeds token_threshold, the hook must prune
    down to event_threshold (not zero) and record the token-aware payload."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        token_map[sid] = 3000
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        get_tokens = _fake_token_counter(token_map)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=1000,
            event_threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
            get_tokens=get_tokens,
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 1
    prune_event = prune_events[0]
    expected_dropped = _PLANTS - DEFAULT_THRESHOLD
    assert prune_event.payload == {
        "dropped": expected_dropped,
        "threshold_tokens": 1000,
        "session_tokens": 3000,
    }
    assert captured[0]["payload"] == {
        "dropped": expected_dropped,
        "threshold_tokens": 1000,
        "session_tokens": 3000,
    }


def test_token_aware_does_not_prune_when_under_token_threshold(tmp_path) -> None:
    """When session_tokens is at or below token_threshold, the hook must
    be a no-op."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        token_map[sid] = 500
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=1000,
            pruner=pruner,
            tracer=tracer,
            get_tokens=_fake_token_counter(token_map),
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 0
    assert captured == []
    assert len(surviving) == _PLANTS


def test_token_aware_preserves_tool_result_and_user_prompt(tmp_path) -> None:
    """``tool_result`` and ``user_prompt`` events must never be dropped,
    even when token pruning triggers."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        for _ in range(150):
            logger.record(sid, kind="tool_result", payload={"i": 0})
        for _ in range(50):
            logger.record(sid, kind="user_prompt", payload={"i": 0})
        for _ in range(100):
            logger.record(sid, kind="tool_call", payload={"i": 0})
        token_map[sid] = 3000
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=1000,
            pruner=pruner,
            tracer=tracer,
            get_tokens=_fake_token_counter(token_map),
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


def test_token_aware_constructor_rejects_invalid_threshold(tmp_path) -> None:
    """token_threshold < 1 must raise ValueError."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        pass
    with pytest.raises(ValueError, match="token_threshold must be >= 1"):
        TokenAwarePruningHook(
            session_id=sid,
            token_threshold=0,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
            get_tokens=lambda s: 0,
        )


def test_token_aware_constructor_rejects_empty_session_id(tmp_path) -> None:
    """session_id must be a non-empty string."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as _:
        pass
    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        TokenAwarePruningHook(
            session_id="",
            token_threshold=1000,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
            get_tokens=lambda s: 0,
        )


def test_token_aware_constructor_rejects_invalid_event_threshold(tmp_path) -> None:
    """event_threshold < 1 must raise ValueError (issue #581)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        pass
    with pytest.raises(ValueError, match="event_threshold must be >= 1"):
        TokenAwarePruningHook(
            session_id=sid,
            token_threshold=1000,
            event_threshold=0,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
            get_tokens=lambda s: 0,
        )


def test_token_aware_event_threshold_property() -> None:
    """TokenAwarePruningHook must expose event_threshold as a property."""
    hook = TokenAwarePruningHook(
        session_id="any",
        token_threshold=16384,
        event_threshold=300,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
        get_tokens=lambda s: 0,
    )
    assert hook.event_threshold == 300


def test_token_aware_event_threshold_defaults_to_default_threshold() -> None:
    """event_threshold defaults to DEFAULT_THRESHOLD when not specified."""
    hook = TokenAwarePruningHook(
        session_id="any",
        token_threshold=16384,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
        get_tokens=lambda s: 0,
    )
    assert hook.event_threshold == DEFAULT_THRESHOLD


def test_token_aware_post_tool_is_pass_through(tmp_path) -> None:
    """``post_tool`` must return the result untouched."""
    hook = TokenAwarePruningHook(
        session_id="any",
        token_threshold=1000,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
        get_tokens=lambda s: 0,
    )
    call = ToolCall(name="read_file", arguments={})
    result = ToolResult(name="read_file", output="hello")

    returned = _run(hook.post_tool(call, result))
    assert returned is result


def test_token_aware_register_token_aware_into_installs(tmp_path) -> None:
    """``register_token_aware_into`` must install the hook into the
    supplied registry and not the process default."""
    from harness.hooks.base import get_registry

    default_before = get_registry()
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        fresh = HookRegistry()
        hook = register_token_aware_into(
            fresh,
            session_id=sid,
            token_threshold=1000,
            pruner=_sqlite_pruner(db),
            tracer=lambda *a, **k: None,
            get_tokens=lambda s: 0,
        )
        assert hook in fresh._hooks  # noqa: SLF001
        assert hook not in default_before._hooks  # noqa: SLF001
        assert hook.session_id == sid
        assert hook.token_threshold == 1000


def test_resolve_context_tokens_threshold_returns_int(monkeypatch) -> None:
    """When FOUNDRY_CONTEXT_TOKENS is set, returns int."""
    monkeypatch.setenv("FOUNDRY_CONTEXT_TOKENS", "5000")
    assert resolve_context_tokens_threshold() == 5000


def test_resolve_context_tokens_threshold_returns_none_when_unset(monkeypatch) -> None:
    """When FOUNDRY_CONTEXT_TOKENS is absent/empty, returns None."""
    monkeypatch.delenv("FOUNDRY_CONTEXT_TOKENS", raising=False)
    assert resolve_context_tokens_threshold() is None


def test_token_aware_payload_json_round_tripable(tmp_path) -> None:
    """The token-aware context_pruned payload must round-trip through JSON."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        token_map[sid] = 3000
        pruner = _sqlite_pruner(db)
        tracer, _ = _tracer_for(logger, sid)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=1000,
            event_threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
            get_tokens=_fake_token_counter(token_map),
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_event = next(e for e in surviving if e.kind == "context_pruned")
    encoded = prune_event.model_dump_json()
    decoded = json.loads(encoded)
    assert decoded["kind"] == "context_pruned"
    assert decoded["payload"]["dropped"] == _PLANTS - DEFAULT_THRESHOLD
    assert decoded["payload"]["threshold_tokens"] == 1000
    assert decoded["payload"]["session_tokens"] == 3000


# ---------------------------------------------------------------------------
# Issue #492: Large-session pruning validation
# ---------------------------------------------------------------------------


def test_pre_tool_prunes_1000_plus_events(tmp_path) -> None:
    """1000 planted events, one ``pre_tool`` call: post-prune count must
    be ``<= threshold`` and ``context_pruned`` event must record the
    exact dropped count."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, 1000)
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
    assert prune_events[0].payload == {
        "dropped": 1000 - DEFAULT_THRESHOLD,
        "threshold": DEFAULT_THRESHOLD,
        "token_threshold": DEFAULT_TOKEN_THRESHOLD,
    }
    assert captured[0]["payload"] == {
        "dropped": 1000 - DEFAULT_THRESHOLD,
        "threshold": DEFAULT_THRESHOLD,
        "token_threshold": DEFAULT_TOKEN_THRESHOLD,
    }


def test_pre_tool_prunes_2000_events_session(tmp_path) -> None:
    """2000 planted events, one ``pre_tool`` call: post-prune count must
    be ``<= threshold`` and ``context_pruned`` event must record the
    exact dropped count. This validates pruning at scale (issue #492)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, 2000)
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )

        _run(hook.pre_tool(ToolCall(name="bash", arguments={"command": "ls"})))

        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    real_events = [e for e in surviving if e.kind != "context_pruned"]
    assert len(real_events) == DEFAULT_THRESHOLD
    assert len(prune_events) == 1
    assert prune_events[0].payload["dropped"] == 2000 - DEFAULT_THRESHOLD


def test_pre_tool_preserves_all_tool_results_regardless_of_age(tmp_path) -> None:
    """All ``tool_result`` events must survive pruning regardless of age.
    Plant 100 tool_results spread across a 500-event session; after
    pruning to DEFAULT_THRESHOLD all 100 tool_results must survive."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        for i in range(100):
            logger.record(sid, kind="tool_result", payload={"index": i, "age": "old"})
        for _ in range(400):
            logger.record(sid, kind="tool_call", payload={"index": 0})
        for i in range(100, 200):
            logger.record(sid, kind="tool_result", payload={"index": i, "age": "new"})
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
    assert kinds.count("tool_result") == 200
    assert kinds.count("tool_call") == 0
    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 1
    assert prune_events[0].payload["dropped"] == 400


def test_dropped_events_recorded_correctly_in_trace(tmp_path) -> None:
    """The ``context_pruned`` event payload must record the exact number
    of events that were dropped, enabling trace consumers to verify
    pruning happened and to account for missing events."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, 500)
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

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 1
    expected_dropped = 500 - DEFAULT_THRESHOLD
    assert prune_events[0].payload["dropped"] == expected_dropped
    assert prune_events[0].payload["threshold"] == DEFAULT_THRESHOLD
    assert captured[0]["payload"]["dropped"] == expected_dropped
    assert captured[0]["payload"]["threshold"] == DEFAULT_THRESHOLD


def test_pruning_performance_reasonable_for_1000_events(tmp_path) -> None:
    """Pruning a 1000-event session must complete within 1 second.
    This is a sanity-check bound; the actual requirement is that
    pruning adds negligible latency to the tool-call pre_tool hook."""
    import time

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, 1000)
        pruner = _sqlite_pruner(db)
        tracer, _ = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )

        start = time.perf_counter()
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"pruning took {elapsed:.3f}s, expected < 1.0s"


def test_pruning_performance_reasonable_for_2000_events(tmp_path) -> None:
    """Pruning a 2000-event session must complete within 2 seconds.
    Linear scaling is acceptable; this validates pruning at scale
    does not become a bottleneck (issue #492)."""
    import time

    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, 2000)
        pruner = _sqlite_pruner(db)
        tracer, _ = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )

        start = time.perf_counter()
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"pruning took {elapsed:.3f}s, expected < 2.0s"


def test_multiple_prune_calls_accumulate_correctly(tmp_path) -> None:
    """Multiple ``pre_tool`` calls on a session that stays over threshold
    must record a ``context_pruned`` event on each call with the
    correct dropped count for that specific pruning operation."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, 600)
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = ContextPruningHook(
            session_id=sid,
            threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
        )

        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        _run(hook.pre_tool(ToolCall(name="bash", arguments={})))
        _run(hook.pre_tool(ToolCall(name="list_dir", arguments={})))

        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 3
    for pe in prune_events:
        assert pe.payload["threshold"] == DEFAULT_THRESHOLD


# ---------------------------------------------------------------------------
# Token-aware pruning: large-session and edge cases (issue #492, #553)
# ---------------------------------------------------------------------------


def test_token_aware_prunes_multiple_times_when_over_threshold(tmp_path) -> None:
    """Multiple ``pre_tool`` calls on a session that stays over the token
    threshold must record a ``context_pruned`` event on each call, as long
    as there are events left to drop. Prunes to event_threshold (not zero)
    per issue #581 fix."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        token_map[sid] = 5000
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=1000,
            event_threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
            get_tokens=_fake_token_counter(token_map),
        )

        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        _run(hook.pre_tool(ToolCall(name="bash", arguments={})))
        _run(hook.pre_tool(ToolCall(name="list_dir", arguments={})))

        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 3
    assert all(p.payload["threshold_tokens"] == 1000 for p in prune_events)
    assert all(p.payload["session_tokens"] == 5000 for p in prune_events)


def test_token_aware_does_not_prune_at_exact_threshold(tmp_path) -> None:
    """When session_tokens equals (not exceeds) token_threshold, the hook
    must be a no-op. The condition is strictly greater-than."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, _PLANTS)
        token_map[sid] = 1000
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=1000,
            pruner=pruner,
            tracer=tracer,
            get_tokens=_fake_token_counter(token_map),
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 0
    assert captured == []
    assert len(surviving) == _PLANTS


def test_token_aware_prunes_with_large_token_counts(tmp_path) -> None:
    """Token-aware pruning must handle sessions with very large token
    counts (e.g. 50000 tokens on a 6600 XT with --ctx-size 8192).
    Pruning fires and prunes down to event_threshold, not zero (issue #581)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        _plant(logger, sid, 500)
        token_map[sid] = 50000
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=8192,
            event_threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
            get_tokens=_fake_token_counter(token_map),
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    real_events = [e for e in surviving if e.kind != "context_pruned"]
    assert len(prune_events) == 1
    assert len(real_events) == DEFAULT_THRESHOLD
    assert prune_events[0].payload["threshold_tokens"] == 8192
    assert prune_events[0].payload["session_tokens"] == 50000
    assert prune_events[0].payload["dropped"] == 500 - DEFAULT_THRESHOLD


def test_token_aware_token_threshold_property() -> None:
    """TokenAwarePruningHook must expose token_threshold as a property."""
    hook = TokenAwarePruningHook(
        session_id="any",
        token_threshold=16384,
        pruner=lambda *a, **k: 0,
        tracer=lambda *a, **k: None,
        get_tokens=lambda s: 0,
    )
    assert hook.token_threshold == 16384


def test_resolve_context_tokens_threshold_parses_int_correctly(monkeypatch) -> None:
    """``resolve_context_tokens_threshold`` must return the exact int value."""
    monkeypatch.setenv("FOUNDRY_CONTEXT_TOKENS", "16384")
    assert resolve_context_tokens_threshold() == 16384


def test_resolve_context_tokens_threshold_rejects_non_digit(monkeypatch) -> None:
    """Non-numeric FOUNDRY_CONTEXT_TOKENS must raise ValueError."""
    monkeypatch.setenv("FOUNDRY_CONTEXT_TOKENS", "not_a_number")
    with pytest.raises(ValueError, match="invalid literal"):
        resolve_context_tokens_threshold()


def test_token_aware_pruning_at_5600g_6600xt_context_window(tmp_path) -> None:
    """Simulate the 6600 XT context window (8192 tokens). When session_tokens
    is 9000 (exceeds context window) and FOUNDRY_CONTEXT_TOKENS is 8192,
    pruning must fire and preserve tool_result and user_prompt events.
    Prunes to event_threshold (200), not zero (issue #581 fix)."""
    db = tmp_path / "traces.db"
    logger = TraceLogger(db)
    token_map: dict[str, int] = {}
    with logger.session(harness_version="test-0.0") as sid:
        for _ in range(100):
            logger.record(sid, kind="tool_result", payload={"i": 0})
        for _ in range(50):
            logger.record(sid, kind="user_prompt", payload={"i": 0})
        for _ in range(200):
            logger.record(sid, kind="tool_call", payload={"i": 0})
        token_map[sid] = 9000
        pruner = _sqlite_pruner(db)
        tracer, captured = _tracer_for(logger, sid)
        hook = TokenAwarePruningHook(
            session_id=sid,
            token_threshold=8192,
            event_threshold=DEFAULT_THRESHOLD,
            pruner=pruner,
            tracer=tracer,
            get_tokens=_fake_token_counter(token_map),
        )
        _run(hook.pre_tool(ToolCall(name="read_file", arguments={})))
        surviving = logger.load_session(sid)

    real_events = [e for e in surviving if e.kind != "context_pruned"]
    kinds = [e.kind for e in real_events]
    assert kinds.count("tool_result") == 100
    assert kinds.count("user_prompt") == 50
    assert kinds.count("tool_call") == 50
    prune_events = [e for e in surviving if e.kind == "context_pruned"]
    assert len(prune_events) == 1
    assert prune_events[0].payload["dropped"] == 150
    assert prune_events[0].payload["threshold_tokens"] == 8192
    assert prune_events[0].payload["session_tokens"] == 9000
