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

from benchmarks.conftest import DEFAULT_SKIP, FIXTURES_ROOT, _seed_workspace

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


@pytest.mark.parametrize("benchmark_workspace", ["sample"], indirect=True)
def test_workspace_seeds_from_fixtures(benchmark_workspace: Path) -> None:
    """An indirect param seeds the workspace from ``fixtures/<name>/``."""
    expected = FIXTURES_ROOT / "sample" / "expected.txt"
    assert (benchmark_workspace / "expected.txt").read_text() == expected.read_text()


def test_seed_workspace_raises_on_missing_directory(tmp_path: Path) -> None:
    """A missing fixture directory fails fast instead of seeding nothing."""
    with pytest.raises(FileNotFoundError):
        _seed_workspace(tmp_path, "does_not_exist")


# ---------------------------------------------------------------------------
# skip parameter
# ---------------------------------------------------------------------------


def test_seed_workspace_default_skip_excludes_pycache(tmp_path: Path) -> None:
    """The default skip set excludes ``__pycache__`` etc."""
    workspace = tmp_path / "ws"
    _seed_workspace(workspace, "skip_demo")
    assert (workspace / "keep.txt").exists()


def test_seed_workspace_skip_excludes_named_dirs(tmp_path: Path) -> None:
    """An explicit ``skip`` set excludes matching directory/file names."""
    workspace = tmp_path / "ws"
    _seed_workspace(
        workspace,
        "skip_demo",
        skip={"__pycache__", ".git", ".venv", "excluded"},
    )
    assert (workspace / "keep.txt").exists()
    assert not (workspace / "excluded").exists()


def test_seed_workspace_manifest_skip(tmp_path: Path) -> None:
    """A ``fixture.toml`` manifest provides skip patterns without explicit skip."""
    workspace = tmp_path / "ws"
    _seed_workspace(workspace, "skip_demo")
    assert (workspace / "keep.txt").exists()
    assert not (workspace / "excluded").exists()


def test_seed_workspace_metadata_files_not_copied(tmp_path: Path) -> None:
    """Seed-control files never leak into the workspace."""
    workspace = tmp_path / "ws"
    _seed_workspace(workspace, "skip_demo")
    assert not (workspace / "fixture.toml").exists()


# ---------------------------------------------------------------------------
# template.env expansion
# ---------------------------------------------------------------------------


def test_seed_workspace_template_expansion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``{{VAR}}`` placeholders are replaced with env-var values."""
    monkeypatch.setenv("SEED_FEATURE_VAR", "expanded_value")
    workspace = tmp_path / "ws"
    _seed_workspace(workspace, "template_demo")
    content = (workspace / "config.txt").read_text(encoding="utf-8")
    assert "expanded_value" in content
    assert "{{" not in content


def test_seed_workspace_template_unset_var_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An unset placeholder variable raises ``KeyError``."""
    monkeypatch.delenv("SEED_FEATURE_VAR", raising=False)
    workspace = tmp_path / "ws"
    with pytest.raises(KeyError, match="SEED_FEATURE_VAR"):
        _seed_workspace(workspace, "template_demo")


def test_default_skip_constant() -> None:
    """``DEFAULT_SKIP`` contains the documented defaults."""
    assert DEFAULT_SKIP == frozenset({"__pycache__", ".git", ".venv"})
