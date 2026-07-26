"""Static validation of ``.github/workflows/audit.yml`` paths filter (issue #1012).

Verifies that ``audit.yml`` triggers its ``pull_request`` job only when
dependency-affecting files change, consistent with the pattern established
by ``docker.yml`` and ``critic.yml``.  The ``pip-audit`` scan is only
meaningful when Python dependencies may have changed; running it on
docs-only or CI-config-only PRs wastes CI minutes with no security signal.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
AUDIT_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "audit.yml"

# Files that affect Python dependencies and should trigger pip-audit.
DEPENDENCY_PATHS = (
    "pyproject.toml",
    "uv.lock",
    "**/requirements*.txt",
)


@pytest.fixture(scope="module")
def config() -> dict:
    assert AUDIT_WORKFLOW.exists(), f"audit.yml missing at {AUDIT_WORKFLOW}"
    data = yaml.safe_load(AUDIT_WORKFLOW.read_text())
    assert isinstance(data, dict), "audit.yml must parse to a mapping."
    return data


def _pull_request_trigger(config: dict) -> dict:
    """Return the ``on.pull_request`` value, handling YAML's ``on`` → ``True`` quirk."""
    on = config.get("on") or config.get(True) or {}
    pr = on.get("pull_request")
    assert pr is not None, "audit.yml must have an on.pull_request trigger."
    assert isinstance(pr, dict), "on.pull_request must be a mapping."
    return pr


def test_pull_request_trigger_has_paths_key(config: dict) -> None:
    """``on.pull_request`` must declare a ``paths`` key (issue #1012)."""
    pr_trigger = _pull_request_trigger(config)
    assert "paths" in pr_trigger, (
        "on.pull_request must have a paths key to avoid running pip-audit "
        "on dependency-irrelevant changes (issue #1012)."
    )


def test_paths_cover_dependency_files(config: dict) -> None:
    """``paths`` must cover all files that affect Python dependencies."""
    pr_trigger = _pull_request_trigger(config)
    paths: list[str] = pr_trigger.get("paths", [])
    for dep_file in DEPENDENCY_PATHS:
        assert any(
            dep_file in p for p in paths
        ), (
            f"paths must include {dep_file!r} to trigger pip-audit only when "
            f"dependencies change (issue #1012). Current paths: {paths}"
        )
