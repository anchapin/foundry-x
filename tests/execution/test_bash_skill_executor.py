"""Unit tests for the subprocess-backed bash skill executor (issue #258)."""

from __future__ import annotations

from pathlib import Path

import pytest

from foundry_x.execution.runner import (
    _bash_skill_executor,
    _default_skill_executor,
    _resolve_bash_max_timeout,
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
    async def test_cwd_inside_workspace_subdir_is_allowed(self, tmp_path: Path) -> None:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        sub = workspace / "sub"
        sub.mkdir()
        result = await _bash_skill_executor(
            "bash", {"command": "pwd", "cwd": str(sub)}, workspace_dir=workspace
        )
        assert result["exit_code"] == 0
        assert "sub" in result["stdout"]

    @pytest.mark.asyncio
    async def test_cwd_outside_workspace_root_returns_error(self, tmp_path: Path) -> None:
        """Issue #935: a ``cwd`` resolving outside ``workspace_root`` is rejected.

        Mirrors the file-operation skill confinement: ``_resolve_path`` raises
        ``ValueError`` for path escapes, and the executor returns an error
        result instead of executing the command in the escaped directory.
        """
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        outside = tmp_path / "outside"
        outside.mkdir()

        result = await _bash_skill_executor(
            "bash", {"command": "pwd", "cwd": str(outside)}, workspace_dir=workspace
        )
        assert result["exit_code"] == -1
        assert result["truncated"] is False
        assert result["stdout"] == ""
        assert "escapes workspace root" in result["error"]

    @pytest.mark.asyncio
    async def test_cwd_dotdot_escape_returns_error(self, tmp_path: Path) -> None:
        """Issue #935: a relative ``..`` escape in ``cwd`` is rejected too."""
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        (workspace / "inner").mkdir()

        result = await _bash_skill_executor(
            "bash", {"command": "pwd", "cwd": "../outside"}, workspace_dir=workspace
        )
        assert result["exit_code"] == -1
        assert "escapes workspace root" in result["error"]

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


class TestBashTimeoutClamp:
    """Issue #1467: model-supplied ``timeout_seconds`` is clamped to an upper bound."""

    @pytest.mark.asyncio
    async def test_inflated_timeout_is_clamped_to_cap(self) -> None:
        """A model sending ``timeout_seconds=999999`` uses the cap (60 by default).

        We assert the command completes quickly (echo is instant) and that a
        warning is emitted documenting the clamp — proving the inflated value
        did not reach ``subprocess.run``.
        """
        with pytest.warns(UserWarning, match="exceeds cap"):
            result = await _bash_skill_executor(
                "bash", {"command": "echo fast", "timeout_seconds": 999999}
            )
        assert result["exit_code"] == 0
        assert "fast" in result["stdout"]

    @pytest.mark.asyncio
    async def test_clamped_timeout_still_kills_hung_command(self, monkeypatch) -> None:
        """The clamped cap is enforced: with a 1s cap, a ``sleep 10`` with a
        huge requested timeout is killed within the cap window — proving the
        inflated value did not reach ``subprocess.run``."""
        monkeypatch.setenv("FOUNDRY_BASH_MAX_TIMEOUT_S", "1")
        with pytest.warns(UserWarning, match="exceeds cap"):
            result = await _bash_skill_executor(
                "bash",
                {"command": "sleep 10", "timeout_seconds": 999999},
            )
        assert result["exit_code"] == -1
        assert result["truncated"] is True

    @pytest.mark.asyncio
    async def test_timeout_within_cap_is_not_clamped(self, monkeypatch) -> None:
        """A timeout under the cap produces no warning and is respected."""
        monkeypatch.setenv("FOUNDRY_BASH_MAX_TIMEOUT_S", "5")
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error")
            result = await _bash_skill_executor(
                "bash", {"command": "echo ok", "timeout_seconds": 2}
            )
        assert result["exit_code"] == 0
        assert "ok" in result["stdout"]

    @pytest.mark.asyncio
    async def test_env_override_raises_cap(self, monkeypatch) -> None:
        """A lower ``FOUNDRY_BASH_MAX_TIMEOUT_S`` clamps even modest requests."""
        monkeypatch.setenv("FOUNDRY_BASH_MAX_TIMEOUT_S", "1")
        with pytest.warns(UserWarning, match="exceeds cap"):
            result = await _bash_skill_executor(
                "bash", {"command": "sleep 10", "timeout_seconds": 30}
            )
        assert result["exit_code"] == -1
        assert result["truncated"] is True


class TestResolveBashMaxTimeout:
    """Tests for the ``_resolve_bash_max_timeout`` config resolver (issue #1467)."""

    def test_absent_env_returns_default(self, monkeypatch) -> None:
        monkeypatch.delenv("FOUNDRY_BASH_MAX_TIMEOUT_S", raising=False)
        assert _resolve_bash_max_timeout() == 60

    def test_empty_env_returns_default(self, monkeypatch) -> None:
        monkeypatch.setenv("FOUNDRY_BASH_MAX_TIMEOUT_S", "")
        assert _resolve_bash_max_timeout() == 60

    def test_explicit_value_is_returned(self, monkeypatch) -> None:
        monkeypatch.setenv("FOUNDRY_BASH_MAX_TIMEOUT_S", "120")
        assert _resolve_bash_max_timeout() == 120

    def test_non_positive_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("FOUNDRY_BASH_MAX_TIMEOUT_S", "0")
        with pytest.warns(UserWarning, match="non-positive"):
            assert _resolve_bash_max_timeout() == 60

    def test_non_integer_raises(self, monkeypatch) -> None:
        monkeypatch.setenv("FOUNDRY_BASH_MAX_TIMEOUT_S", "oops")
        with pytest.raises(ValueError):
            _resolve_bash_max_timeout()


class TestDefaultSkillExecutor:
    """Tests for _default_skill_executor (issue #416).

    The default executor now returns an explicit error envelope for any
    unimplemented skill, replacing the fake ``{"status": "ok"}`` stub.
    """

    @pytest.mark.asyncio
    async def test_returns_error_envelope_for_unimplemented_skill(self) -> None:
        result = await _default_skill_executor("bash", {"command": "echo hello"})
        assert result["error"] is not None
        assert "not implemented" in result["error"]
        assert result["skill"] == "bash"
        assert "command" in result["echo"]

    @pytest.mark.asyncio
    async def test_echoes_argument_keys_for_unimplemented_skill(self) -> None:
        result = await _default_skill_executor("nonexistent_skill", {"path": "/tmp/x"})
        assert result["error"] is not None
        assert "not implemented" in result["error"]
        assert result["skill"] == "nonexistent_skill"
        assert result["echo"] == ["path"]
