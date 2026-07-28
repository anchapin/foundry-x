"""stats module for refactor_api_with_constraints (seeded/stale caller).

Calls ``clamp`` with two positional arguments -- the OLD signature.
"""

from lib.math_utils import clamp


def clamp_mean(values: list[float]) -> float:
    """Return the mean of *values* clamped to [0, 1000]."""
    m = sum(values) / len(values)
    return clamp(m, 0, 1000)  # stale positional call
