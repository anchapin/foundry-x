"""Callee module for the multi_file_rename benchmark (seeded/broken).

Defines ``compute_value`` -- the OLD function name.  The caller has
already been migrated to the new name ``calculate_value``, but this
callee was not renamed, so ``python -m caller`` raises
``AttributeError`` until the callee is renamed to match.

This file is the *seeded* (broken) state.  The golden rename lives in
``benchmarks/tasks/test_multi_file_rename.py`` (``GOLDEN_CALLEE``).
"""


def compute_value() -> str:
    """Return the canonical value string (rename target: calculate_value)."""
    return "value=7"
