"""Mirror test: ``harness/hooks/rate_limit.py`` matches the foundry-resident
Evolver defaults (issue #206).

``docs/SECURITY.md:50-52`` claims the rate-limit defaults "live in
``harness/hooks/``." This test enforces numerical equality between the
harness-side mirror (``harness/hooks/rate_limit.py``) and the
foundry-side runtime defaults
(``src/foundry_x/evolution/evolver.py:Evolver.__init__``). A drift
between the two surfaces here as a test failure rather than a silent
divergence from the SECURITY.md prose.

Acceptance criteria from issue #206:

* ``get_default_max_proposals()`` matches the foundry-resident default
  (numerical equality, both values logged).
* The mirror imports nothing from ``src/foundry_x/`` (AGENTS.md section 7
  self-reference loop).
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

from foundry_x.evolution.evolver import Evolver, _RATE_WINDOW
from harness.hooks.rate_limit import (
    DEFAULT_MAX_DIFF_LINES,
    DEFAULT_MAX_PROPOSALS_PER_HOUR,
    DEFAULT_RATE_WINDOW_HOURS,
    get_default_max_diff_lines,
    get_default_max_proposals,
    get_default_rate_window_hours,
)

_log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
RATE_LIMIT_SOURCE = REPO_ROOT / "harness" / "hooks" / "rate_limit.py"


def test_get_default_max_proposals_matches_evolver() -> None:
    """``get_default_max_proposals()`` must equal the Evolver's default."""
    mirror = get_default_max_proposals()
    foundry = Evolver().max_proposals_per_hour
    _log.info(
        "rate_limit mirror max_proposals=%d, evolver default=%d",
        mirror,
        foundry,
    )
    assert mirror == foundry, (
        f"harness/hooks/rate_limit.py DEFAULT_MAX_PROPOSALS_PER_HOUR={mirror} "
        f"diverges from Evolver.max_proposals_per_hour={foundry}"
    )
    assert mirror == DEFAULT_MAX_PROPOSALS_PER_HOUR


def test_get_default_max_diff_lines_matches_evolver() -> None:
    """``get_default_max_diff_lines()`` must equal the Evolver's default."""
    mirror = get_default_max_diff_lines()
    foundry = Evolver().max_diff_lines
    _log.info(
        "rate_limit mirror max_diff_lines=%d, evolver default=%d",
        mirror,
        foundry,
    )
    assert mirror == foundry, (
        f"harness/hooks/rate_limit.py DEFAULT_MAX_DIFF_LINES={mirror} "
        f"diverges from Evolver.max_diff_lines={foundry}"
    )
    assert mirror == DEFAULT_MAX_DIFF_LINES


def test_get_default_rate_window_hours_matches_evolver() -> None:
    """``get_default_rate_window_hours()`` must equal the Evolver's window."""
    mirror = get_default_rate_window_hours()
    foundry = int(_RATE_WINDOW.total_seconds() // 3600)
    _log.info(
        "rate_limit mirror rate_window_hours=%d, evolver window=%dh",
        mirror,
        foundry,
    )
    assert mirror == foundry, (
        f"harness/hooks/rate_limit.py DEFAULT_RATE_WINDOW_HOURS={mirror} "
        f"diverges from Evolver _RATE_WINDOW={foundry}h"
    )
    assert mirror == DEFAULT_RATE_WINDOW_HOURS


def test_rate_limit_source_imports_nothing_from_foundry_x() -> None:
    """The mirror must not import from ``src/foundry_x/`` (AGENTS.md section 7).

    Walks the AST of ``harness/hooks/rate_limit.py`` and rejects any
    ``import foundry_x...`` or ``from foundry_x... import ...`` statement.
    A harness-to-foundry import closes the self-reference loop in the
    forbidden direction.
    """
    source = RATE_LIMIT_SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not alias.name.startswith("foundry_x"), (
                    f"harness/hooks/rate_limit.py imports {alias.name!r} "
                    f"-- forbidden by AGENTS.md section 7"
                )
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert not module.startswith("foundry_x"), (
                f"harness/hooks/rate_limit.py imports from {module!r} "
                f"-- forbidden by AGENTS.md section 7"
            )
