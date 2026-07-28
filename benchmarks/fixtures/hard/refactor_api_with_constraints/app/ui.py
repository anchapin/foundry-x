"""ui module for refactor_api_with_constraints (seeded/stale caller).

Calls ``clamp`` with two positional arguments -- the OLD signature.
"""

from lib.math_utils import clamp


def display_bar(value: float, width: int = 5) -> str:
    """Render *value* clamped to [1, width] as a row of '#' characters."""
    v = clamp(value, 1, width)  # stale positional call
    return "#" * int(v)
