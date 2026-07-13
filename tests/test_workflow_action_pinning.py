"""Static validation of GitHub Actions pinning in workflow files.

Per docs/SECURITY.md threat #3, all external GitHub Actions used in
workflows MUST be pinned to a 40-char commit SHA with a trailing
comment indicating the tag they resolve to. This prevents a compromised
or yanked movable tag (@v4, @main, @latest) from silently changing
the checkout action across every CI run (issue #282).

Guards against supply-chain attacks on the GitHub Actions ecosystem.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS_DIR = REPO_ROOT / ".github" / "workflows"

# Pattern: uses: <owner>/<repo>@<ref> or uses: ./.github/actions/<name>
# For external actions, <ref> must be a 40-char hex SHA.
EXTERNAL_ACTION_RE = re.compile(
    r"uses:\s*(?!\./)([a-zA-Z0-9\-_.]+/[a-zA-Z0-9\-_.]+)@([a-zA-Z0-9]+)"
)

# A 40-char lowercase hex SHA
SHA_40_RE = re.compile(r"^[a-f0-9]{40}$")

# Movable tags that must NOT be used
MOVABLE_TAGS = {"v4", "v3", "v2", "v1", "main", "master", "latest"}


@pytest.fixture(scope="module")
def workflow_files() -> list[Path]:
    """Collect all YAML workflow files."""
    if not WORKFLOWS_DIR.exists():
        pytest.skip("no .github/workflows directory")
    return sorted(WORKFLOWS_DIR.glob("*.yml"))


@pytest.fixture(scope="module")
def all_uses_lines(workflow_files: list[Path]) -> list[tuple[Path, int, str]]:
    """Extract all `uses:` lines from workflow files with file, line number, and content."""
    results: list[tuple[Path, int, str]] = []
    for wf in workflow_files:
        for i, line in enumerate(wf.read_text().splitlines(), start=1):
            if "uses:" in line and line.strip().startswith("uses:"):
                results.append((wf, i, line.strip()))
    return results


def test_all_external_actions_pinned_to_sha(
    all_uses_lines: list[tuple[Path, int, str]],
) -> None:
    """Every external `uses:` line MUST reference a 40-char commit SHA.

    Local actions (./.github/actions/*) are excluded since they are
    already pinned by the repository commit itself.
    """
    violations: list[str] = []
    for wf, lineno, line in all_uses_lines:
        # Skip local actions
        if "./" in line:
            continue
        m = EXTERNAL_ACTION_RE.search(line)
        if not m:
            # Shouldn't happen for well-formed uses: lines, but skip gracefully
            continue
        action_ref = m.group(2)
        if not SHA_40_RE.match(action_ref):
            violations.append(f"{wf.name}:{lineno}: {line} (ref={action_ref!r})")

    assert not violations, (
        "External GitHub Actions must be pinned to a 40-char commit SHA, "
        "not a movable tag. Found violations:\n" + "\n".join(f"  - {v}" for v in violations)
    )


def test_no_movable_tag_references(
    all_uses_lines: list[tuple[Path, int, str]],
) -> None:
    """No `uses:` line may reference a movable tag like @v4, @main, @latest."""
    violations: list[str] = []
    for wf, lineno, line in all_uses_lines:
        if "./" in line:
            continue
        m = EXTERNAL_ACTION_RE.search(line)
        if not m:
            continue
        action_ref = m.group(2)
        if action_ref in MOVABLE_TAGS:
            violations.append(f"{wf.name}:{lineno}: {line}")

    assert not violations, (
        "External GitHub Actions must NOT use movable tags. Found:\n"
        + "\n".join(f"  - {v}" for v in violations)
        + "\nPin to a 40-char commit SHA with a trailing comment (e.g. "
        + "uses: actions/checkout@abc123... # v4)"
    )


def test_sha_pins_have_tag_comment(
    all_uses_lines: list[tuple[Path, int, str]],
) -> None:
    """Every SHA-pinned `uses:` line SHOULD have a trailing comment
    indicating which tag the SHA resolves to, for auditability."""
    warnings: list[str] = []
    for wf, lineno, line in all_uses_lines:
        if "./" in line:
            continue
        m = EXTERNAL_ACTION_RE.search(line)
        if not m:
            continue
        action_ref = m.group(2)
        if SHA_40_RE.match(action_ref) and "#" not in line:
            warnings.append(f"{wf.name}:{lineno}: {line}")

    assert not warnings, (
        "SHA-pinned `uses:` lines should have a trailing comment "
        "indicating the resolved tag (e.g. # v4). Found:\n"
        + "\n".join(f"  - {w}" for w in warnings)
    )
