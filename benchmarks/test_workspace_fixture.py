"""Infrastructure tests for the ``benchmark_workspace`` fixture.

These are NOT benchmark tasks for the agent (they carry no ``benchmark``
marker); they verify the isolation contract declared in
``benchmarks/conftest.py`` so that task authors in a later issue can rely on
it. They run as part of the full suite (``uv run pytest``).

Acceptance for issue #29: a test using the fixture writes a file to the
yielded path and asserts it exists, and nothing lands in the repository
tree after the run (``git status`` clean).
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.conftest import FIXTURES_ROOT, _seed_workspace

#: Absolute path to the repository root (parent of ``benchmarks/``).
REPO_ROOT = Path(__file__).resolve().parents[1]


def test_workspace_is_writable_and_persists_within_test(
    benchmark_workspace: Path,
) -> None:
    """A file written to the workspace is readable for the rest of the test."""
    out = benchmark_workspace / "agent_output.txt"
    out.write_text("result\n", encoding="utf-8")
    assert out.exists()
    assert out.read_text(encoding="utf-8") == "result\n"


def test_workspace_is_empty_by_default(benchmark_workspace: Path) -> None:
    """With no indirect parameter the workspace starts empty."""
    assert list(benchmark_workspace.iterdir()) == []


def test_workspace_lives_outside_the_repository_tree(
    benchmark_workspace: Path,
) -> None:
    """The workspace is under the OS temp dir, never inside the repo."""
    repo = REPO_ROOT
    ws = benchmark_workspace.resolve()
    assert ws != repo
    assert repo not in ws.parents


def test_workspace_is_unique_per_test(benchmark_workspace: Path) -> None:
    """A file written by another test must not leak into this workspace."""
    leaked = benchmark_workspace / "agent_output.txt"
    assert not leaked.exists()


@pytest.mark.parametrize("benchmark_workspace", ["reverse_string"], indirect=True)
def test_workspace_seeds_from_fixtures(benchmark_workspace: Path) -> None:
    """An indirect param seeds the workspace from ``fixtures/<name>/``."""
    expected = FIXTURES_ROOT / "reverse_string" / "expected.txt"
    assert (benchmark_workspace / "expected.txt").read_text() == expected.read_text()


def test_seed_workspace_raises_on_missing_directory(tmp_path: Path) -> None:
    """A missing fixture directory fails fast instead of seeding nothing."""
    with pytest.raises(FileNotFoundError):
        _seed_workspace(tmp_path, "does_not_exist")
