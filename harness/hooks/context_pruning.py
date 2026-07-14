"""Context-pruning hook (issue #106, docs/ROADMAP.md:31).

Implements the ``Hook.pre_tool`` slot to bound per-session event
accumulation. When the running session's accumulated event count exceeds
a configurable threshold (``DEFAULT_THRESHOLD`` = 200), the hook drops
the oldest events whose ``kind`` is not in :data:`_PRESERVE_KINDS`
(``tool_result`` and ``user_prompt``) down to the threshold, then records
a ``context_pruned`` trace event carrying the dropped count.

Why this exists
---------------
``docs/ROADMAP.md:31`` calls for "refine the hooks to prune historical
logs efficiently and keep inference latency low on the 5600G / 6600 XT
setup." ``docs/PHILOSOPHY.md`` §5 commits to local-first on that hardware.
The trace store currently keeps every event indefinitely
(``src/foundry_x/trace/logger.py:TraceEvent``), so a long session
exhausts the context budget. This hook is the first step: drop the noisy
middle of the trace while preserving the user-visible bookends (the user
prompts and tool results the model has to keep seeing).

The hook is opt-in via ``harness/manifest.json`` ``hooks`` list (issue
#103). Without that opt-in the registry does not load it. The Critic
(ADR-0004) gates the manifest change on the benchmark suite before it
goes active.

Decoupling from the trace store
-------------------------------
The hook accepts a ``pruner`` callable and a ``tracer`` callable rather
than importing :class:`foundry_x.trace.logger.TraceLogger` directly. The
self-reference loop in AGENTS.md §7 forbids that import direction:
``harness/`` is the artifact being evolved and ``src/foundry_x/`` is the
machinery. The :mod:`harness.hooks.injection_firewall` precedent
(``tracer=`` keyword on :class:`InjectionFirewallHook`) follows the same
shape. The runner supplies real TraceLogger-backed closures; tests supply
fakes or direct-SQLite closures.

Out of scope (issue #106)
-------------------------
- Modifying the :class:`TraceEvent` schema (separate evolution-slot
  proposal).
- Implementing summarization-based pruning (a different proposal).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from .base import HookRegistry, ToolCall, ToolResult

_log = logging.getLogger("harness.hooks.context_pruning")

_FOUNDRY_CONTEXT_TOKENS_ENV = "FOUNDRY_CONTEXT_TOKENS"


DEFAULT_THRESHOLD: int = 200

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
# ``kind='context_pruned'``.
#
# For :class:`ContextPruningHook`: ``payload={'dropped': <int>,
# 'threshold': <int>}`` (issue #106 acceptance: the dropped-count
# payload follows the existing TraceEvent payload pattern; ADR-0006).
#
# For :class:`TokenAwarePruningHook`: ``payload={'dropped': <int>,
# 'threshold_tokens': <int>, 'session_tokens': <int>}`` so the KPI
# layer can report context efficiency (issue #465 acceptance).
Tracer = Callable[[str, str, dict[str, object]], None]


class ContextPruningHook:
    """Hook that bounds per-session event accumulation (issue #106).

    ``pre_tool`` is the gate. On every tool call it asks the injected
    ``pruner`` how many events would be dropped to bring the session
    down to ``threshold``. When the answer is positive it asks the
    ``tracer`` to record a ``context_pruned`` event carrying the count.
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
    threshold:
        Maximum number of events the session may hold. When the count
        exceeds this value, the hook prunes down to the threshold. Must
        be ``>= 1``.
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
        threshold: int = DEFAULT_THRESHOLD,
        pruner: Pruner,
        tracer: Tracer,
    ) -> None:
        if threshold < 1:
            raise ValueError(f"context_pruning: threshold must be >= 1, got {threshold!r}")
        if not session_id:
            raise ValueError("context_pruning: session_id must be a non-empty string")
        self._session_id = session_id
        self._threshold = threshold
        self._pruner = pruner
        self._tracer = tracer

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def threshold(self) -> int:
        return self._threshold

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        dropped = self._pruner(self._session_id, _PRESERVE_KINDS, self._threshold)
        if dropped > 0:
            self._tracer(
                self._session_id,
                "context_pruned",
                {"dropped": dropped, "threshold": self._threshold},
            )
            _log.info(
                "context_pruning: dropped %d event(s) from session %r (threshold=%d)",
                dropped,
                self._session_id,
                self._threshold,
            )
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        return result


TokenCounter = Callable[[str], int]


class TokenAwarePruningHook:
    """Token-aware variant of :class:`ContextPruningHook` (issue #465).

    Instead of counting events, this hook compares the session's accumulated
    token count against a ``token_threshold``. The accumulated token count
    is obtained by calling the injected ``get_tokens`` callable, which the
    runner implements by returning its running ``tokens_used`` counter.

    The ``context_pruned`` event payload includes
    ``{{"dropped": N, "threshold_tokens": M, "session_tokens": P}}``
    so the KPI layer can report context efficiency (issue #465 acceptance).

    Parameters
    ----------
    session_id:
        Identifier of the running session (same contract as
        :class:`ContextPruningHook`).
    token_threshold:
        Maximum cumulative token count permitted before pruning triggers.
        Must be ``>= 1``.
    get_tokens:
        Callable accepting ``session_id`` and returning the current
        cumulative token count for that session. The runner passes a
        closure that returns its running ``tokens_used`` counter.
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
        token_threshold: int,
        get_tokens: TokenCounter,
        pruner: Pruner,
        tracer: Tracer,
    ) -> None:
        if token_threshold < 1:
            raise ValueError(
                f"context_pruning: token_threshold must be >= 1, got {token_threshold!r}"
            )
        if not session_id:
            raise ValueError("context_pruning: session_id must be a non-empty string")
        self._session_id = session_id
        self._token_threshold = token_threshold
        self._get_tokens = get_tokens
        self._pruner = pruner
        self._tracer = tracer

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def token_threshold(self) -> int:
        return self._token_threshold

    async def pre_tool(self, call: ToolCall) -> ToolCall:
        session_tokens = self._get_tokens(self._session_id)
        if session_tokens <= self._token_threshold:
            return call
        dropped = self._pruner(self._session_id, _PRESERVE_KINDS, DEFAULT_THRESHOLD)
        if dropped > 0:
            self._tracer(
                self._session_id,
                "context_pruned",
                {
                    "dropped": dropped,
                    "threshold_tokens": self._token_threshold,
                    "session_tokens": session_tokens,
                },
            )
            _log.info(
                "context_pruning: dropped %d event(s) from session %r "
                "(token_threshold=%d, session_tokens=%d)",
                dropped,
                self._session_id,
                self._token_threshold,
                session_tokens,
            )
        return call

    async def post_tool(self, call: ToolCall, result: ToolResult) -> ToolResult:
        return result


def register_into(
    registry: HookRegistry,
    *,
    session_id: str,
    threshold: int = DEFAULT_THRESHOLD,
    pruner: Pruner,
    tracer: Tracer,
) -> ContextPruningHook:
    """Install a fresh :class:`ContextPruningHook` into ``registry``.

    Mirrors :func:`harness.hooks.injection_firewall.register_into`: pass
    a :class:`HookRegistry` to install the hook into it without touching
    the process default. Returns the hook so callers can introspect or
    detach it.

    The runner wires ``pruner`` / ``tracer`` to TraceLogger-backed
    closures. The Critic sandbox passes a fresh ``HookRegistry`` so
    variant A cannot leak into variant B's evaluation (ADR-0004, issue
    #22).
    """
    hook = ContextPruningHook(
        session_id=session_id,
        threshold=threshold,
        pruner=pruner,
        tracer=tracer,
    )
    registry.register(hook)
    return hook


def register_token_aware_into(
    registry: HookRegistry,
    *,
    session_id: str,
    token_threshold: int,
    get_tokens: TokenCounter,
    pruner: Pruner,
    tracer: Tracer,
) -> TokenAwarePruningHook:
    """Install a fresh :class:`TokenAwarePruningHook` into ``registry``.

    Mirrors :func:`register_into` but for the token-aware variant.
    The runner's ``get_tokens`` closure is passed directly so the hook
    can query the runner's running token counter on each ``pre_tool``
    call (issue #465).
    """
    hook = TokenAwarePruningHook(
        session_id=session_id,
        token_threshold=token_threshold,
        get_tokens=get_tokens,
        pruner=pruner,
        tracer=tracer,
    )
    registry.register(hook)
    return hook


def resolve_token_threshold(env: dict[str, str] | None = None) -> int | None:
    """Resolve the token threshold from ``FOUNDRY_CONTEXT_TOKENS``.

    Returns ``None`` when the env var is absent or empty; callers should
    fall back to event-count :class:`ContextPruningHook` in that case.
    Raises :class:`ValueError` for non-positive integer values (fail-fast
    per AGENTS.md §2).
    """
    source = env if env is not None else os.environ
    raw = source.get(_FOUNDRY_CONTEXT_TOKENS_ENV, "").strip()
    if not raw:
        return None
    value = int(raw)
    if value < 1:
        raise ValueError(f"{_FOUNDRY_CONTEXT_TOKENS_ENV} must be a positive integer, got {value!r}")
    return value


__all__ = [
    "ContextPruningHook",
    "DEFAULT_THRESHOLD",
    "Pruner",
    "Tracer",
    "TokenCounter",
    "TokenAwarePruningHook",
    "register_into",
    "register_token_aware_into",
    "resolve_token_threshold",
    "_FOUNDRY_CONTEXT_TOKENS_ENV",
]
