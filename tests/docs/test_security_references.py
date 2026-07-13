"""Verify SECURITY.md guardrail references resolve to real tools and paths.

SECURITY.md is the threat-model source of truth. Two of its guardrail
bullets name a tool and a file path: the pre-commit secret scanner and
the Evolver rate-limiter defaults. If the prose drifts from reality
(e.g. naming ``git-secrets`` when the hook is actually ``gitleaks``, or
pointing rate limits at ``harness/hooks/`` when the runtime cap lives in
``src/foundry_x/evolution/evolver.py``) a contributor hardening those
guardrails looks in the wrong place and finds nothing.

This module pins the two references to the codebase so a future drift
surfaces as a test failure rather than a stale pointer. See issue #288
and ADR-0009.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SECURITY_MD = REPO_ROOT / "docs" / "SECURITY.md"
PRECOMMIT_CONFIG = REPO_ROOT / ".pre-commit-config.yaml"
EVOLVER_PATH = REPO_ROOT / "src" / "foundry_x" / "evolution" / "evolver.py"


def test_precommit_hook_bullet_names_gitleaks() -> None:
    """The 'Pre-commit hooks' bullet must name ``gitleaks``, not ``git-secrets``.

    The actual hook configured in ``.pre-commit-config.yaml`` is gitleaks
    (issue #126). Naming a different tool in SECURITY.md sends contributors
    to a non-existent hook.
    """
    text = SECURITY_MD.read_text(encoding="utf-8")
    assert "git-secrets" not in text, (
        "SECURITY.md still references the stale 'git-secrets' tool; the "
        "configured pre-commit secret scanner is 'gitleaks' "
        "(.pre-commit-config.yaml)."
    )
    assert "gitleaks" in text, (
        "SECURITY.md 'Pre-commit hooks' bullet must name 'gitleaks' to match "
        ".pre-commit-config.yaml."
    )


def test_security_md_gitleaks_matches_precommit_config() -> None:
    """The gitleaks claim in SECURITY.md must match the configured hook.

    If the pre-commit configuration ever swaps the scanner, SECURITY.md must
    be updated in lock-step; this test fails first.
    """
    config = PRECOMMIT_CONFIG.read_text(encoding="utf-8")
    assert "gitleaks" in config, (
        ".pre-commit-config.yaml no longer configures gitleaks; update "
        "SECURITY.md's 'Pre-commit hooks' bullet to name the real scanner."
    )


def test_rate_limits_bullet_points_at_evolver() -> None:
    """The 'Rate limits' bullet must point at the Evolver source.

    The runtime cap (``max_proposals_per_hour`` / ``max_diff_lines``) and the
    ``_check_rate_limit`` enforcement live in
    ``src/foundry_x/evolution/evolver.py``. SECURITY.md must not claim the
    defaults live under ``harness/hooks/`` (issue #288, out-of-scope: moving
    the limiter into harness/hooks/).
    """
    text = SECURITY_MD.read_text(encoding="utf-8")
    assert "harness/hooks/" not in text, (
        "SECURITY.md still references 'harness/hooks/' for the rate limiter; "
        "the runtime cap is enforced in "
        "src/foundry_x/evolution/evolver.py (issue #288)."
    )
    assert "src/foundry_x/evolution/evolver.py" in text, (
        "SECURITY.md 'Rate limits' bullet must point at "
        "src/foundry_x/evolution/evolver.py where the runtime cap is enforced."
    )
    assert EVOLVER_PATH.is_file(), (
        f"{EVOLVER_PATH.relative_to(REPO_ROOT)} is missing; update "
        "SECURITY.md's 'Rate limits' bullet to the real location."
    )
    evolver = EVOLVER_PATH.read_text(encoding="utf-8")
    assert "max_proposals_per_hour" in evolver, (
        "evolver.py no longer defines max_proposals_per_hour; update "
        "SECURITY.md's 'Rate limits' bullet."
    )
