"""Tests for the file-operation skill executors (issue #259).

Exercises the four executors wired into ``run_task`` against a temp workspace:
edit_file, write_file, list_dir, and grep_search. Each test verifies the
acceptance criteria from issue #259 against a synthetic fixture.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from foundry_x.execution.runner import (
    _exec_edit_file,
    _exec_grep_search,
    _exec_list_dir,
    _exec_write_file,
    _resolve_path,
    _resolve_workspace_root,
)


class TestResolvePath:
    def test_relative_path_resolved_against_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "file.txt").write_text("hello", encoding="utf-8")

        result = _resolve_path("file.txt", ws)
        assert result == ws / "file.txt"

    def test_absolute_path_within_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        target = ws / "file.txt"
        target.write_text("hello", encoding="utf-8")

        result = _resolve_path(str(target), ws)
        assert result == target

    def test_absolute_path_outside_workspace_raises(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        outside = tmp_path / "outside.txt"
        outside.write_text("secret", encoding="utf-8")

        with pytest.raises(ValueError, match="escapes workspace root"):
            _resolve_path(str(outside), ws)

    def test_path_with_dotdot_escapes_workspace(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "file.txt").write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="escapes workspace root"):
            _resolve_path("..", ws)

    def test_path_with_dotdot_in_subdir_escapes(self, tmp_path: Path) -> None:
        ws = tmp_path / "ws"
        ws.mkdir()
        sub = ws / "sub"
        sub.mkdir()
        (sub / "file.txt").write_text("hello", encoding="utf-8")

        with pytest.raises(ValueError, match="escapes workspace root"):
            _resolve_path("sub/../../outside.txt", ws)


class TestExecListDir:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "a.py").write_text("x = 1\n", encoding="utf-8")
        (ws / "b.py").write_text("y = 2\n", encoding="utf-8")
        (ws / "c.txt").write_text("not py\n", encoding="utf-8")
        (ws / ".hidden").write_text("secret\n", encoding="utf-8")
        (ws / "subdir").mkdir()
        (ws / "subdir" / "nested.py").write_text("z = 3\n", encoding="utf-8")
        return ws

    @pytest.mark.asyncio
    async def test_returns_entries_with_correct_kind(self, workspace: Path) -> None:
        result = await _exec_list_dir({"path": str(workspace)}, workspace)

        assert result["truncated"] is False
        names = {e["name"] for e in result["entries"]}
        assert "a.py" in names
        assert "b.py" in names
        assert "c.txt" in names
        for entry in result["entries"]:
            assert entry["kind"] in {"file", "dir", "symlink", "other"}
            assert isinstance(entry["size"], int)
            assert entry["size"] >= 0

    @pytest.mark.asyncio
    async def test_excludes_hidden_by_default(self, workspace: Path) -> None:
        result = await _exec_list_dir({"path": str(workspace)}, workspace)

        names = {e["name"] for e in result["entries"]}
        assert ".hidden" not in names

    @pytest.mark.asyncio
    async def test_include_hidden_reveals_dotfiles(self, workspace: Path) -> None:
        result = await _exec_list_dir(
            {"path": str(workspace), "include_hidden": True},
            workspace,
        )

        names = {e["name"] for e in result["entries"]}
        assert ".hidden" in names

    @pytest.mark.asyncio
    async def test_glob_filters_entries(self, workspace: Path) -> None:
        result = await _exec_list_dir(
            {"path": str(workspace), "glob": "*.py"},
            workspace,
        )

        names = sorted(e["name"] for e in result["entries"])
        assert names == ["a.py", "b.py"]

    @pytest.mark.asyncio
    async def test_max_entries_caps_and_truncates(self, workspace: Path) -> None:
        result = await _exec_list_dir(
            {"path": str(workspace), "max_entries": 2},
            workspace,
        )

        assert result["truncated"] is True
        assert len(result["entries"]) == 2

    @pytest.mark.asyncio
    async def test_missing_directory_returns_empty(self, workspace: Path) -> None:
        result = await _exec_list_dir(
            {"path": str(workspace / "nonexistent")},
            workspace,
        )

        assert result["entries"] == []
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_path_escape_returns_error(self, workspace: Path) -> None:
        result = await _exec_list_dir(
            {"path": str(workspace / ".." / "outside.txt")},
            workspace,
        )

        assert "error" in result
        assert "escapes workspace root" in result["error"]

    @pytest.mark.asyncio
    async def test_sorting_is_deterministic_by_name(self, workspace: Path) -> None:
        result = await _exec_list_dir({"path": str(workspace)}, workspace)

        names = [e["name"] for e in result["entries"]]
        assert names == sorted(names)


class TestExecWriteFile:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "ws"
        ws.mkdir()
        return ws

    @pytest.mark.asyncio
    async def test_creates_file_and_returns_bytes_written(self, workspace: Path) -> None:
        result = await _exec_write_file(
            {"path": "new.txt", "content": "hello world\n"},
            workspace,
        )

        assert "error" not in result
        assert result["bytes_written"] == len("hello world\n".encode("utf-8"))
        assert result["sha256"] == hashlib.sha256("hello world\n".encode("utf-8")).hexdigest()
        assert (workspace / "new.txt").read_text(encoding="utf-8") == "hello world\n"

    @pytest.mark.asyncio
    async def test_overwrites_existing_file(self, workspace: Path) -> None:
        target = workspace / "existing.txt"
        target.write_text("old content\n", encoding="utf-8")

        result = await _exec_write_file(
            {"path": "existing.txt", "content": "new content\n"},
            workspace,
        )

        assert result["bytes_written"] == len("new content\n".encode("utf-8"))
        assert target.read_text(encoding="utf-8") == "new content\n"

    @pytest.mark.asyncio
    async def test_creates_parent_directories(self, workspace: Path) -> None:
        result = await _exec_write_file(
            {"path": "deep/nested/file.txt", "content": "deep\n"},
            workspace,
        )

        assert "error" not in result
        assert (workspace / "deep" / "nested" / "file.txt").read_text(encoding="utf-8") == "deep\n"

    @pytest.mark.asyncio
    async def test_absolute_path_within_workspace(self, workspace: Path) -> None:
        target = workspace / "abs.txt"
        result = await _exec_write_file(
            {"path": str(target), "content": "absolute\n"},
            workspace,
        )

        assert "error" not in result
        assert target.read_text(encoding="utf-8") == "absolute\n"

    @pytest.mark.asyncio
    async def test_absolute_path_outside_workspace_rejected(self, workspace: Path) -> None:
        outside = workspace.parent / "outside.txt"
        result = await _exec_write_file(
            {"path": str(outside), "content": "oops\n"},
            workspace,
        )

        assert "error" in result
        assert "escapes workspace root" in result["error"]

    @pytest.mark.asyncio
    async def test_empty_content(self, workspace: Path) -> None:
        result = await _exec_write_file(
            {"path": "empty.txt", "content": ""},
            workspace,
        )

        assert "error" not in result
        assert result["bytes_written"] == 0
        assert (workspace / "empty.txt").read_text(encoding="utf-8") == ""


class TestExecEditFile:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "ws"
        ws.mkdir()
        target = ws / "target.py"
        target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        return ws

    @pytest.mark.asyncio
    async def test_replaces_old_string_and_returns_sha256(self, workspace: Path) -> None:
        result = await _exec_edit_file(
            {
                "path": "target.py",
                "old_string": "def add(a, b):",
                "new_string": "def subtract(a, b):",
            },
            workspace,
        )

        assert result["replacements_made"] == 1
        new_content = (workspace / "target.py").read_text(encoding="utf-8")
        assert "def subtract(a, b):" in new_content
        assert "def add(a, b):" not in new_content
        assert result["sha256"] == hashlib.sha256(new_content.encode("utf-8")).hexdigest()

    @pytest.mark.asyncio
    async def test_old_string_not_found_returns_zero_replacements(self, workspace: Path) -> None:
        original = (workspace / "target.py").read_text(encoding="utf-8")
        result = await _exec_edit_file(
            {
                "path": "target.py",
                "old_string": "not present",
                "new_string": "replacement",
            },
            workspace,
        )

        assert result["replacements_made"] == 0
        assert (workspace / "target.py").read_text(encoding="utf-8") == original

    @pytest.mark.asyncio
    async def test_file_not_found(self, workspace: Path) -> None:
        result = await _exec_edit_file(
            {
                "path": "nonexistent.py",
                "old_string": "anything",
                "new_string": "replacement",
            },
            workspace,
        )

        assert result["replacements_made"] == 0
        assert "error" in result

    @pytest.mark.asyncio
    async def test_path_escape_rejected(self, workspace: Path) -> None:
        result = await _exec_edit_file(
            {
                "path": "../outside.txt",
                "old_string": "anything",
                "new_string": "replacement",
            },
            workspace,
        )

        assert result["replacements_made"] == 0
        assert "error" in result
        assert "escapes workspace root" in result["error"]


class TestExecGrepSearch:
    @pytest.fixture
    def workspace(self, tmp_path: Path) -> Path:
        ws = tmp_path / "ws"
        ws.mkdir()
        (ws / "a.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
        (ws / "b.py").write_text("def sub(a, b):\n    return a - b\n", encoding="utf-8")
        (ws / "c.txt").write_text("not code here\n", encoding="utf-8")
        (ws / ".hidden.py").write_text("secret = 1\n", encoding="utf-8")
        (ws / "subdir").mkdir()
        (ws / "subdir" / "nested.py").write_text(
            "def mul(a, b):\n    return a * b\n", encoding="utf-8"
        )
        return ws

    @pytest.mark.asyncio
    async def test_finds_matches_in_python_files(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"def \w+\(", "path": str(workspace)},
            workspace,
        )

        assert result["truncated"] is False
        matches = result["matches"]
        assert len(matches) == 3
        files = {m["file"] for m in matches}
        assert "a.py" in files
        assert "b.py" in files
        assert "subdir/nested.py" in files
        assert "c.txt" not in files

    @pytest.mark.asyncio
    async def test_glob_filters_by_filename(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"def \w+\(", "path": str(workspace), "glob": "*.py"},
            workspace,
        )

        files = {m["file"] for m in result["matches"]}
        assert "c.txt" not in files
        assert all(f.endswith(".py") for f in files)

    @pytest.mark.asyncio
    async def test_max_matches_caps_and_truncates(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"def \w+\(", "path": str(workspace), "max_matches": 2},
            workspace,
        )

        assert result["truncated"] is True
        assert len(result["matches"]) == 2

    @pytest.mark.asyncio
    async def test_context_lines_included(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"def add\(", "path": str(workspace), "context_lines": 1},
            workspace,
        )

        match = next(m for m in result["matches"] if m["file"] == "a.py")
        assert "def add(a, b):" in match["text"]

    @pytest.mark.asyncio
    async def test_missing_directory_returns_empty(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": "def", "path": str(workspace / "nonexistent")},
            workspace,
        )

        assert result["matches"] == []
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_path_escape_rejected(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": "def", "path": str(workspace / ".." / "outside")},
            workspace,
        )

        assert "error" in result
        assert "escapes workspace root" in result["error"]

    @pytest.mark.asyncio
    async def test_excludes_hidden_by_default(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"secret", "path": str(workspace)},
            workspace,
        )

        files = {m["file"] for m in result["matches"]}
        assert ".hidden.py" not in files

    @pytest.mark.asyncio
    async def test_include_hidden_reveals_dotfiles(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"secret", "path": str(workspace), "include_hidden": True},
            workspace,
        )

        files = {m["file"] for m in result["matches"]}
        assert ".hidden.py" in files

    @pytest.mark.asyncio
    async def test_pattern_too_long_rejected(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": "x" * 5000, "path": str(workspace)},
            workspace,
        )

        assert "error" in result
        assert "4096" in result["error"]

    @pytest.mark.asyncio
    async def test_invalid_regex_rejected(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"[invalid", "path": str(workspace)},
            workspace,
        )

        assert "error" in result
        assert "invalid regex" in result["error"].lower()

    @pytest.mark.asyncio
    async def test_skips_large_files(self, workspace: Path) -> None:
        big = workspace / "big.py"
        big.write_text("x\n" * 100000, encoding="utf-8")

        result = await _exec_grep_search(
            {"pattern": r"x", "path": str(workspace), "max_file_bytes": 500},
            workspace,
        )

        skipped = [m for m in result["matches"] if "exceeds" in m["text"]]
        assert len(skipped) == 1
        assert skipped[0]["file"] == "big.py"

    @pytest.mark.asyncio
    async def test_results_relative_to_search_path(self, workspace: Path) -> None:
        result = await _exec_grep_search(
            {"pattern": r"def mul\(", "path": str(workspace)},
            workspace,
        )

        match = result["matches"][0]
        assert match["file"] == "subdir/nested.py"
        assert not match["file"].startswith("/")


class TestWorkspaceRootResolution:
    def test_resolve_workspace_root_from_env(self, monkeypatch) -> None:
        monkeypatch.setenv("FOUNDRY_WORKSPACE_ROOT", "/tmp/test_ws")
        result = _resolve_workspace_root()
        assert result == Path("/tmp/test_ws")

    def test_resolve_workspace_root_defaults_to_cwd(self, monkeypatch) -> None:
        monkeypatch.delenv("FOUNDRY_WORKSPACE_ROOT", raising=False)
        result = _resolve_workspace_root()
        assert result == Path.cwd()

    def test_resolve_workspace_root_from_dict(self, monkeypatch) -> None:
        result = _resolve_workspace_root({"FOUNDRY_WORKSPACE_ROOT": "/custom/path"})
        assert result == Path("/custom/path")

    def test_resolve_workspace_root_empty_string_defaults_to_cwd(self, monkeypatch) -> None:
        result = _resolve_workspace_root({"FOUNDRY_WORKSPACE_ROOT": ""})
        assert result == Path.cwd()
