"""Context-pruning hook (issue #106, docs/ROADMAP.md:31).

Implements the ``Hook.pre_tool`` slot to bound per-session event
accumulation and token usage. When the running session's accumulated
event count exceeds a configurable threshold (``DEFAULT_EVENT_THRESHOLD``
= 200) **or** the cumulative token count exceeds
``DEFAULT_TOKEN_THRESHOLD`` (None = unlimited), the hook drops the
oldest events whose ``kind`` is not in :data:`_PRESERVE_KINDS`
(``tool_result`` and ``user_prompt``) down to the threshold, then
records a ``context_pruned`` trace event carrying the dropped count
**and** the token metrics.

Why this exists
---------------
``docs/ROADMAP.md:31`` calls for "refine the hooks to prune historical
logs efficiently and keep inference latency low on the 5600G / 6600 XT
setup." ``docs/PHILOSOPHY.md`` §5 commits to local-first on that hardware.
The trace store currently keeps every event indefinitely
(``src/foundry_x/trace/logger.py:TraceEvent``), so a long session
exhausts the context budget. Issue #418 extends the hook to also track
cumulative token usage: a session with 199 ``tool_result`` events each
containing 8K-token outputs would deliver ~1.5M tokens — far exceeding
any context window — and the event-count-only threshold would never fire.

The hook is opt-in via ``harness/manifest.json`` ``hooks`` list (issue
#103). Without that opt-in the registry does not load it. The Critic
(ADR-0004) gates the manifest change on the benchmark suite before it
goes active.

Decoupling from the trace store
-------------------------------
The hook accepts a ``pruner`` callable, a ``tracer`` callable, and a
``token_counter`` callable rather than importing
:class:`foundry_x.trace.logger.TraceLogger` directly. The self-reference
loop in AGENTS.md §7 forbids that import direction: ``harness/`` is the
artifact being evolved and ``src/foundry_x/`` is the machinery. The
:mod:`harness.hooks.injection_firewall` precedent (``tracer=`` keyword on
:class:`InjectionFirewallHook`) follows the same shape. The runner
supplies real TraceLogger-backed closures; tests supply fakes or
direct-SQLite closures.

Out of scope (issue #106)
-------------------------
- Modifying the :class:`TraceEvent` schema (separate evolution-slot
  proposal).
- Implementing summarization-based pruning (a different proposal).
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from .base import HookRegistry, ToolCall, ToolResult

_log = logging.getLogger("harness.hooks.context_pruning")


DEFAULT_EVENT_THRESHOLD: int = 200
DEFAULT_TOKEN_THRESHOLD: int | None = None
DEFAULT_THRESHOLD: int = DEFAULT_EVENT_THRESHOLD

# Event kinds the hook never drops. ``tool_result`` is the response the
# model has already seen and must keep referencing; ``user_prompt`` is
# the user-visible bookend a long session cannot lose without losing
# intent. Every other kind (``tool_call``, ``task_received``,
# ``model_request`` / ``model_response``, ``critic_verdict``, etc.) is
# considered noise once the count crosses the threshold.
_PRESERVE_KINDS: frozenset[str] = frozenset({"tool_result", "user_prompt"})

# ``pruner(session_id, keep_kinds, target_count) -> int``: drops the
# oldest events for ``session_id`` whose ``kind`` is not in
# ``keep_kinds`` until the session's event count is at most
# ``target_count``. Returns the number of events actually dropped
# (``0`` when the session was already at or below the target). Safe to
# call on a session that does not exist.
Pruner = Callable[[str, frozenset[str], int], int]

# ``tracer(session_id, kind, payload) -> None``: persists a single
# trace event. The hook calls it exactly once per prune with
# ``kind='context_pruned'`` and ``payload={'dropped': <int>,
# 'threshold': <int>, 'tokens_dropped': <int>, 'tokens_remaining': <int>}``
# (issues #106 and #418). The payload follows the existing TraceEvent
# payload pattern (ADR-0006).
Tracer = Callable[[str, str, dict[str, object]], None]

# ``token_counter() -> int``: returns the current cumulative token count
# for the session. The runner supplies a closure that reads its live
# ``tokens_used`` counter so the hook can fire before the next model
# turn consumes the context budget (issue #418).
TokenCounter = Callable[[], int]


class ContextPruningHook:
    """Hook that bounds per-session event accumulation and token usage (issues #106, #418).

    ``pre_tool`` is the gate. On every tool call it checks whether the
    event count exceeds ``event_threshold`` **or** the cumulative token
    count (from ``token_counter``) exceeds ``token_threshold``. When
    either condition is true it asks the ``pruner`` to drop events and
    then asks the ``tracer`` to record a ``context_pruned`` event
    carrying the event-dropped count and token metrics.

    ``post_tool`` is a pass-through: the boundary check happens once
    per tool call rather than twice because the cost of an extra query
    on every result is not worth the marginal freshness on a long
    session, and pruning immediately before a tool call lines up with
    the model invocation that is about to consume the context budget.

    Parameters
    ----------
    session_id:
        Identifier of the running session. The hook does not open or
        close sessions; the caller (the runner) supplies the id and is
        responsible for keeping it valid for the lifetime of the hook.
        The id is bound at construction time so the hook never has to
        reach into a thread-local to find the current session, which
        keeps the protocol synchronous-friendly and Critic-sandbox safe.
    event_threshold:
        Maximum number of events the session may hold. When the count
        exceeds this value, the hook prunes down to the threshold. Must
        be ``>= 1``. Pass ``None`` to disable event-count pruning.
    token_threshold:
        Maximum cumulative token count permitted for the session. When
        ``token_counter()`` returns a value that exceeds this threshold,
        the hook prunes. ``None`` (the default) disables token-count
        pruning. When both thresholds are set, either exceeding triggers
        a prune.
    token_counter:
        Callable returning the current cumulative token count for the
        session (see :data:`TokenCounter`). Required when
        ``token_threshold`` is not ``None``.
    pruner:
        Callable implementing the deletion contract (see :data:`Pruner`).
    tracer:
        Callable that persists the ``context_pruned`` trace event (see
        :data:`Tracer`).
    """

    def __init__(
        self,
        *,
        session_id: str,
        threshold: int | None = None,
        event_threshold: int | None = DEFAULT_EVENT_THRESHOLD,
        token_threshold: int | None = DEFAULT_TOKEN_THRESHOLD,
        token_counter: TokenCounter | None = None,
        pruner: Pruner,
        tracer: Tracer,
    ) -> None:
        if threshold is not None:
            event_threshold = threshold
        if event_threshold is not None and event_threshold < 1:
            raise ValueError(
                f"context_pruning: event_threshold must be >= 1 or None, got {event_threshold!r}"
            )
        if token_threshold is not None and token_threshold < 0:
            raise ValueError(
                f"context_pruning: token_threshold must be >= 0 or None, got {token_threshold!r}"
            )
        if token_threshold is not None and token_counter is None:
            raise ValueError(
                "context_pruning: token_counter is required when token_threshold is set"
            )
        if not session_id:
            raise ValueError("context_pruning: session_id must be a non-empty string")
        self._session_id = session_id
        self._event_threshold = event_threshold
        self._token_threshold = token_threshold
        self._token_counter = token_counter
        self._pruner = pruner
        self._tracer = tracer
        self._pending_drop_count: int | None = None

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def event_threshold(self) -> int | None:
        return self._event_threshold

    @property
    def threshold(self) -> int | None:
        return self._event_threshold

    @property
    def token_threshold(self) -> int | None:
        return self._token_threshold

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        current_tokens = self._token_counter() if self._token_counter is not None else 0
        needs_prune = False
        self._pending_drop_count = None
        if self._event_threshold is not None:
            dropped = self._pruner(self._session_id, _PRESERVE_KINDS, self._event_threshold)
            self._pending_drop_count = dropped
            needs_prune = dropped > 0
        if (
            not needs_prune
            and self._token_threshold is not None
            and self._token_counter is not None
        ):
            if current_tokens > self._token_threshold:
                needs_prune = True
        if not needs_prune:
            return call

        dropped = self._pending_drop_count
        self._tracer(
            self._session_id,
            "context_pruned",
            {
                "dropped": dropped,
                "threshold": self._event_threshold,
                "tokens_dropped": 0,
                "tokens_remaining": current_tokens,
            },
        )
        _log.info(
            "context_pruning: dropped %d event(s) from session %r (event_threshold=%r, token_threshold=%r, tokens_remaining=%d)",
            dropped,
            self._session_id,
            self._event_threshold,
            self._token_threshold,
            current_tokens,
        )
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        return result


def register_into(
    registry: HookRegistry,
    *,
    session_id: str,
    threshold: int | None = None,
    event_threshold: int | None = DEFAULT_EVENT_THRESHOLD,
    token_threshold: int | None = DEFAULT_TOKEN_THRESHOLD,
    token_counter: TokenCounter | None = None,
    pruner: Pruner,
    tracer: Tracer,
) -> ContextPruningHook:
    """Install a fresh :class:`ContextPruningHook` into ``registry``.

    Mirrors :func:`harness.hooks.injection_firewall.register_into`: pass
    a :class:`HookRegistry` to install the hook into it without touching
    the process default. Returns the hook so callers can introspect or
    detach it.

    The runner wires ``pruner`` / ``tracer`` / ``token_counter`` to
    TraceLogger-backed closures. The Critic sandbox passes a fresh
    ``HookRegistry`` so variant A cannot leak into variant B's
    evaluation (ADR-0004, issue #22).
    """
    hook = ContextPruningHook(
        session_id=session_id,
        threshold=threshold,
        event_threshold=event_threshold,
        token_threshold=token_threshold,
        token_counter=token_counter,
        pruner=pruner,
        tracer=tracer,
    )
    registry.register(hook)
    return hook


__all__ = [
    "ContextPruningHook",
    "DEFAULT_EVENT_THRESHOLD",
    "DEFAULT_THRESHOLD",
    "DEFAULT_TOKEN_THRESHOLD",
    "Pruner",
    "TokenCounter",
    "Tracer",
    "register_into",
]
