"""Token-aware pruning hook (issue #465).

This module exists to satisfy the ``manifest.json`` hook-declaration
contract: each hook listed in ``manifest.json`` must correspond to a
``harness/hooks/<name>.py`` file. The ``TokenAwarePruningHook`` class
lives in ``context_pruning.py`` and is re-exported here.
"""

from .context_pruning import (
    TokenAwarePruningHook,
    register_token_aware_into,
)

__all__ = [
    "TokenAwarePruningHook",
    "register_token_aware_into",
]
