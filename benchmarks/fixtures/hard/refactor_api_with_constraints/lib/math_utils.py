"""math_utils library for refactor_api_with_constraints (seeded/half-migrated).

The public ``clamp`` API has ALREADY been migrated to keyword-only
parameters (``clamp(value, *, lo, hi)``).  The three application callers
under ``app/`` still invoke it positionally, so importing the app raises
``TypeError`` at call time and the test suite fails.  The agent must
discover every positional caller and update it to the keyword-only form.
"""


def clamp(value: float, *, lo: float, hi: float) -> float:
    """Clamp *value* to the inclusive ``[lo, hi]`` range.

    ``lo`` and ``hi`` are keyword-only: callers must name them, which is
    the backward-incompatible contract change the callers must adopt.
    """
    if lo > hi:
        raise ValueError(f"lo ({lo}) must be <= hi ({hi})")
    return max(lo, min(hi, value))
