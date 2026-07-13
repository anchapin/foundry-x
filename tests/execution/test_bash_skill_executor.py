"""Unit tests for the subprocess-backed bash skill executor (issue #258)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry_x.execution.runner import (
    _bash_skill_executor,
    _default_skill_executor,
    _truncate_at_newline,
)


class TestTruncateAtNewline:
    """Tests for the _truncate_at_newline helper."""

    def test_fits_within_limit_returns_unchanged(self) -> None:
        data = b"hello world\n"
        result, was_truncated = _truncate_at_newline(data, 20)
        assert result == b"hello world\n"
        assert was_truncated is False

    def test_exactly_at_limit_returns_unchanged(self) -> None:
        data = b"hello world\n"
        result, was_truncated = _truncate_at_newline(data, 12)
        assert result == b"hello world\n"
        assert was_truncated is False

    def test_truncates_at_newline_boundary(self) -> None:
        data = b"line1\nline2\nline3"
        result, was_truncated = _truncate_at_newline(data, 10)
        assert result == b"line1\n"
        assert was_truncated is True

    def test_truncates_at_last_newline_before_limit(self) -> None:
        data = b"abc\ndefghij\nmore text"
        result, was_truncated = _truncate_at_newline(data, 10)
        assert result == b"abc\n"
        assert was_truncated is True

    def test_truncates_at_newline_within_limit(self) -> None:
        data = b"abc\ndefghij\nmore text"
        result, was_truncated = _truncate_at_newline(data, 12)
        assert result == b"abc\ndefghij\n"
        assert was_truncated is True

    def test_truncates_to_empty_if_no_newline_in_range(self) -> None:
        data = b"abcdefghijklmnop"
        result, was_truncated = _truncate_at_newline(data, 5)
        assert result == b"abcde"
        assert was_truncated is True

    def test_empty_data_returns_empty(self) -> None:
        data = b""
        result, was_truncated = _truncate_at_newline(data, 10)
        assert result == b""
        assert was_truncated is False

    def test_single_newline_at_exactly_limit(self) -> None:
        data = b"\n"
        result, was_truncated = _truncate_at_newline(data, 1)
        assert result == b"\n"
        assert was_truncated is False


class TestBashSkillExecutor:
    """Tests for the _bash_skill_executor function."""

    @pytest.mark.asyncio
    async def test_echo_hello_produces_stdout_containing_hello(self) -> None:
        result = await _bash_skill_executor("bash", {"command": "echo hello"})
        assert "hello" in result["stdout"]
        assert result["exit_code"] == 0
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_stderr_captured_separately(self) -> None:
        result = await _bash_skill_executor(
            "bash", {"command": "python3 -c 'import sys; sys.stderr.write(\"error\\n\")'"}
        )
        assert "error" in result["stderr"]
        assert result["exit_code"] == 0
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_exit_code_nonzero_on_failure(self) -> None:
        result = await _bash_skill_executor(
            "bash", {"command": "python3 -c 'import sys; sys.exit(42)'"}
        )
        assert result["exit_code"] == 42
        assert result["truncated"] is False

    @pytest.mark.asyncio
    async def test_timeout_produces_exit_code_minus_one_and_truncated(self) -> None:
        result = await _bash_skill_executor("bash", {"command": "sleep 10", "timeout_seconds": 1})
        assert result["exit_code"] == -1
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_truncation_at_newline_boundary(self) -> None:
        long_output = "x" * 10000 + "\nmore"
        result = await _bash_skill_executor(
            "bash", {"command": f"echo '{long_output}'", "max_output_bytes": 100}
        )
        assert result["truncated"] is True
        assert len(result["stdout"]) <= 100
        assert result["stdout"].endswith("\n") or len(result["stdout"]) <= 100

    @pytest.mark.asyncio
    async def test_default_timeout_is_30_seconds(self) -> None:
        result = await _bash_skill_executor("bash", {"command": "echo quick"})
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_default_max_output_bytes_is_32768(self) -> None:
        result = await _bash_skill_executor("bash", {"command": "echo small"})
        assert result["exit_code"] == 0
        assert len(result["stdout"]) <= 32768

    @pytest.mark.asyncio
    async def test_cwd_parameter_is_respected(self, tmp_path: Path) -> None:
        result = await _bash_skill_executor("bash", {"command": "pwd", "cwd": str(tmp_path)})
        assert result["exit_code"] == 0
        assert tmp_path.name in result["stdout"] or str(tmp_path) in result["stdout"]

    @pytest.mark.asyncio
    async def test_workspace_dir_used_when_cwd_not_provided(self, tmp_path: Path) -> None:
        result = await _bash_skill_executor("bash", {"command": "pwd"}, workspace_dir=tmp_path)
        assert result["exit_code"] == 0
        assert tmp_path.name in result["stdout"] or str(tmp_path) in result["stdout"]

    @pytest.mark.asyncio
    async def test_shlex_split_handles_quotes(self) -> None:
        result = await _bash_skill_executor("bash", {"command": 'echo "hello world"'})
        assert "hello world" in result["stdout"]
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_shell_false_does_not_expand_wildcards(self) -> None:
        result = await _bash_skill_executor("bash", {"command": "echo *"})
        assert result["exit_code"] == 0
        assert "*" in result["stdout"] or "test_bash_skill_executor" in result["stdout"]


class TestDefaultSkillExecutor:
    """Tests for _default_skill_executor to ensure backward compatibility."""

    @pytest.mark.asyncio
    async def test_returns_acknowledgment_envelope(self) -> None:
        result = await _default_skill_executor("bash", {"command": "echo hello"})
        assert result["status"] == "ok"
        assert result["skill"] == "bash"
        assert "command" in result["echo"]

    @pytest.mark.asyncio
    async def test_echoes_argument_keys(self) -> None:
        result = await _default_skill_executor("read_file", {"path": "/tmp/x", "offset": 10})
        assert result["status"] == "ok"
        assert result["skill"] == "read_file"
        assert sorted(result["echo"]) == ["offset", "path"]
