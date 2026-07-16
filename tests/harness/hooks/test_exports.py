"""Tests for ``harness/hooks/__init__.py`` exports (issue #578).

Verifies that ``context_pruning`` symbols are properly exported from
``harness.hooks`` so that code doing ``import harness.hooks`` can access
``ContextPruningHook`` and related symbols without ImportError.

Acceptance criteria from issue #578:

1. ``import harness.hooks; harness.hooks.ContextPruningHook`` works
2. ``import harness.hooks; harness.hooks.TokenAwarePruningHook`` works
   (note: TokenAwarePruningHook does not exist in context_pruning.py;
   only ContextPruningHook is exported per the proposed edit)

3. Existing tests in ``tests/harness/hooks/`` pass
4. ``harness.hooks.__all__`` contains the context_pruning symbols
"""

from __future__ import annotations

import harness.hooks


def test_ContextPruningHook_is_exported() -> None:
    """ContextPruningHook must be accessible as harness.hooks.ContextPruningHook."""
    assert hasattr(harness.hooks, "ContextPruningHook")
    assert harness.hooks.ContextPruningHook.__name__ == "ContextPruningHook"


def test_DEFAULT_THRESHOLD_is_exported() -> None:
    """DEFAULT_THRESHOLD must be accessible as harness.hooks.DEFAULT_THRESHOLD."""
    assert hasattr(harness.hooks, "DEFAULT_THRESHOLD")
    assert isinstance(harness.hooks.DEFAULT_THRESHOLD, int)
    assert harness.hooks.DEFAULT_THRESHOLD == 200


def test_Pruner_is_exported() -> None:
    """Pruner type alias must be accessible as harness.hooks.Pruner."""
    assert hasattr(harness.hooks, "Pruner")


def test_Tracer_is_exported() -> None:
    """Tracer type alias must be accessible as harness.hooks.Tracer."""
    assert hasattr(harness.hooks, "Tracer")


def test_register_into_is_exported() -> None:
    """register_into must be accessible as harness.hooks.register_into."""
    assert hasattr(harness.hooks, "register_into")
    assert callable(harness.hooks.register_into)


def test_context_pruning_in_all() -> None:
    """All context_pruning symbols must appear in harness.hooks.__all__."""
    all_names = harness.hooks.__all__
    expected = {"ContextPruningHook", "DEFAULT_THRESHOLD", "Pruner", "Tracer", "register_into"}
    missing = expected - set(all_names)
    assert not missing, f"context_pruning exports missing from __all__: {missing}"


def test_injection_firewall_still_exported() -> None:
    """Existing injection_firewall exports must still work after the fix."""
    assert hasattr(harness.hooks, "InjectionFirewallHook")
    assert hasattr(harness.hooks, "INJECTION_PATTERNS")


def test_base_still_exported() -> None:
    """Existing base exports must still work after the fix."""
    assert hasattr(harness.hooks, "Hook")
    assert hasattr(harness.hooks, "HookRegistry")
    assert hasattr(harness.hooks, "get_registry")
    assert hasattr(harness.hooks, "register_hook")
