"""calc module for refactor_api_with_constraints (seeded/stale caller).

Calls ``clamp`` with two positional arguments -- the OLD signature.
Since ``clamp`` is now keyword-only, this raises ``TypeError`` at call
time.
"""

from lib.math_utils import clamp


def bounded_sum(a: float, b: float) -> float:
    """Return ``a + b`` clamped to [0, 100]."""
    return clamp(a + b, 0, 100)  # stale positional call
