"""Tests for ``harness/hooks/__init__.py`` (issue #712 / #634).

Verifies that every hook listed in ``harness/manifest.json`` is importable
from the ``harness.hooks`` public interface, matching the manifest
declarations. Issue #634 identified that ``context_pruning`` and ``rate_limit``
were not exported from ``harness/hooks/__init__.py`` even though
``manifest.json`` declared them. This caused ``load_check.py`` to pass while
the hooks were not actually importable at runtime.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "harness" / "manifest.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    assert MANIFEST_PATH.exists(), f"manifest missing at {MANIFEST_PATH}"
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def manifest_hooks(manifest: dict) -> list[str]:
    hooks = manifest.get("hooks", [])
    assert isinstance(hooks, list), "manifest.hooks must be a list"
    return hooks


def test_context_pruning_imports() -> None:
    from harness.hooks import (
        ContextPruningHook,
        DEFAULT_THRESHOLD,
        Pruner,
        Tracer,
        register_into,
    )

    assert ContextPruningHook is not None
    assert isinstance(DEFAULT_THRESHOLD, int)
    assert callable(Pruner)
    assert callable(Tracer)
    assert callable(register_into)


def test_rate_limit_imports() -> None:
    from harness.hooks import (
        RateLimitHook,
        DEFAULT_MAX_DIFF_LINES,
        DEFAULT_MAX_PROPOSALS_PER_HOUR,
        DEFAULT_RATE_WINDOW_HOURS,
        get_default_max_diff_lines,
        get_default_max_proposals,
        get_default_rate_window_hours,
    )

    assert RateLimitHook is not None
    assert isinstance(DEFAULT_MAX_DIFF_LINES, int)
    assert isinstance(DEFAULT_MAX_PROPOSALS_PER_HOUR, int)
    assert isinstance(DEFAULT_RATE_WINDOW_HOURS, int)
    assert callable(get_default_max_diff_lines)
    assert callable(get_default_max_proposals)
    assert callable(get_default_rate_window_hours)


def test_manifest_imports(manifest_hooks: list[str]) -> None:
    """Every hook listed in manifest.json must be importable from harness.hooks.

    This catches the bug reported in issue #634 where load_check.py passed
    but the hooks were not actually importable because they were not exported
    from the __init__.py.
    """
    import harness.hooks

    missing: list[str] = []
    for hook_name in manifest_hooks:
        try:
            getattr(harness.hooks, hook_name)
        except AttributeError:
            missing.append(hook_name)

    assert not missing, (
        f"The following hooks are listed in manifest.json but are not "
        f"importable from harness.hooks: {missing}"
    )
