"""Rate-limit defaults mirror (SECURITY.md "Rate limits", issue #206).

``docs/SECURITY.md:50-52`` states that the Evolver rate-limit defaults --
"max N proposals per hour, max M lines of harness diff per proposal" --
"live in ``harness/hooks/``." Before this module those defaults lived
only in ``src/foundry_x/evolution/evolver.py`` (``Evolver.__init__``).
This module is the thin mirror that makes the SECURITY.md claim true:
the same numeric defaults are now discoverable from the harness side.

Self-reference constraint (AGENTS.md section 7)
-----------------------------------------------
This module imports **nothing** from ``src/foundry_x/``. The harness is
the artifact being evolved; the foundry is the machinery that evolves
it. A harness-to-foundry import would close that loop in the wrong
direction. The numeric equality between this mirror and the
foundry-resident defaults is asserted by
``tests/harness/test_rate_limit_mirror.py`` so a drift between the two
surfaces as a test failure rather than a silent divergence.

Seed status (issue #206)
------------------------
This file is a **seed** created per issue #206's explicit request. It is
a new file landing the cap's declarative form, not hand-edited DNA in
the sense of AGENTS.md section 2. Future evolution runs may modify it
through the normal ``Evolver`` -> ``Critic`` pipeline (ADR-0004).

Out of scope (issue #206)
-------------------------
- Moving the runtime cap from ``src/foundry_x/`` to ``harness/hooks/``.
- Adding a runtime hook invocation in ``main()`` (separate runner-slot
  proposal).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Default cap values -- mirror of Evolver.__init__ defaults
# ---------------------------------------------------------------------------
# These constants are the single source of truth on the harness side. The
# foundry-side runtime defaults live in
# ``src/foundry_x/evolution/evolver.py`` (``Evolver(max_proposals_per_hour=10,
# max_diff_lines=200)`` plus ``_RATE_WINDOW = timedelta(hours=1)``). The two
# MUST agree; ``tests/harness/test_rate_limit_mirror.py`` enforces that.

DEFAULT_MAX_PROPOSALS_PER_HOUR: int = 10
DEFAULT_MAX_DIFF_LINES: int = 200
DEFAULT_RATE_WINDOW_HOURS: int = 1


def get_default_max_proposals() -> int:
    """Return the default cap on proposals per rolling hour.

    Mirrors ``Evolver.__init__(max_proposals_per_hour=10)`` in
    ``src/foundry_x/evolution/evolver.py``. SECURITY.md "Rate limits":
    "max N proposals per hour."
    """
    return DEFAULT_MAX_PROPOSALS_PER_HOUR


def get_default_max_diff_lines() -> int:
    """Return the default cap on unified-diff lines per proposal.

    Mirrors ``Evolver.__init__(max_diff_lines=200)`` in
    ``src/foundry_x/evolution/evolver.py``. SECURITY.md "Rate limits":
    "max M lines of harness diff per proposal."
    """
    return DEFAULT_MAX_DIFF_LINES


def get_default_rate_window_hours() -> int:
    """Return the sliding-window length in hours for the proposal cap.

    Mirrors ``_RATE_WINDOW = timedelta(hours=1)`` in
    ``src/foundry_x/evolution/evolver.py``.
    """
    return DEFAULT_RATE_WINDOW_HOURS


__all__ = [
    "DEFAULT_MAX_DIFF_LINES",
    "DEFAULT_MAX_PROPOSALS_PER_HOUR",
    "DEFAULT_RATE_WINDOW_HOURS",
    "get_default_max_diff_lines",
    "get_default_max_proposals",
    "get_default_rate_window_hours",
]
