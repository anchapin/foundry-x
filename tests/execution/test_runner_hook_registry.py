"""Tests for hook registry resolution in ``runner._resolve_hook_registry`` (issue #260).

The hook registry is resolved lazily so the runner can import without the
harness package on ``sys.path``. When the harness *is* importable but
``get_registry()`` raises, the session must continue in degraded mode (no
hooks) **and** record a ``hook_registry_error`` trace event so the Digester
and operator observe that the security-critical injection firewall is off.
"""

from unittest.mock import patch

import pytest

from foundry_x.execution.runner import _resolve_hook_registry
from foundry_x.trace.logger import TraceLogger


@pytest.fixture
def trace_logger(tmp_path):
    db = tmp_path / "traces.db"
    return TraceLogger(db)


@pytest.fixture
def session_id(trace_logger):
    with trace_logger.session(harness_version="test") as sid:
        yield sid


class TestResolveHookRegistry:
    """Tests for :func:`_resolve_hook_registry`."""

    def test_returns_none_when_harness_not_importable(
        self, trace_logger, session_id, monkeypatch
    ):
        """Missing harness (ImportError) is a legitimate degraded state —
        returns ``None`` without a trace event.
        """
        # Mock the import to raise ImportError
        def raise_import_error(*args, **kwargs):
            if args[0] == "harness.hooks":
                raise ImportError("No module named 'harness.hooks'")
            return __import__(*args, **kwargs)

        monkeypatch.setattr("builtins.__import__", raise_import_error)

        registry = _resolve_hook_registry(trace_logger, session_id)
        assert registry is None

        events = trace_logger.load_session(session_id)
        hook_errors = [e for e in events if e.kind == "hook_registry_error"]
        assert not hook_errors, "ImportError must not emit hook_registry_error"

    def test_records_hook_registry_error_when_get_registry_raises(
        self, trace_logger, session_id
    ):
        """When ``get_registry()`` raises, a ``hook_registry_error`` event
        is recorded with ``error_type`` and ``message``.
        """
        exc = RuntimeError("registry initialization failed")

        with patch("harness.hooks.get_registry", side_effect=exc):
            registry = _resolve_hook_registry(trace_logger, session_id)

        assert registry is None, "must return None so session continues in degraded mode"

        events = trace_logger.load_session(session_id)
        hook_errors = [e for e in events if e.kind == "hook_registry_error"]
        assert len(hook_errors) == 1, "exactly one hook_registry_error event expected"
        payload = hook_errors[0].payload
        assert payload["error_type"] == "RuntimeError"
        assert payload["message"] == "registry initialization failed"

    def test_records_hook_registry_error_for_any_exception(
        self, trace_logger, session_id
    ):
        """Any ``Exception`` subclass from ``get_registry()`` is captured."""
        exc = ValueError("config invalid")

        with patch("harness.hooks.get_registry", side_effect=exc):
            registry = _resolve_hook_registry(trace_logger, session_id)

        assert registry is None
        events = trace_logger.load_session(session_id)
        hook_errors = [e for e in events if e.kind == "hook_registry_error"]
        assert len(hook_errors) == 1
        payload = hook_errors[0].payload
        assert payload["error_type"] == "ValueError"
        assert payload["message"] == "config invalid"

    def test_session_completes_in_degraded_mode(
        self, trace_logger, session_id, tmp_path, monkeypatch
    ):
        """Even when registry resolution fails, the session can complete
        (degraded mode — no hooks run).
        """
        harness_dir = tmp_path / "harness"
        harness_dir.mkdir()
        (harness_dir / "system_prompt.txt").write_text("test\n")
        (harness_dir / "hooks").mkdir()
        (harness_dir / "skills").mkdir()

        with patch("harness.hooks.get_registry", side_effect=RuntimeError("boom")):
            registry = _resolve_hook_registry(trace_logger, session_id)

        assert registry is None
        # The runner's run_task would proceed with registry=None and skip
        # hook fan-out. This test just confirms the function doesn't raise.

    def test_existing_registry_none_path_still_works(
        self, trace_logger, session_id, monkeypatch
    ):
        """When ``get_registry()`` returns a registry normally, it is
        returned (not None). This ensures the happy path is unchanged.
        """
        from harness.hooks import HookRegistry

        mock_registry = HookRegistry()

        with patch("harness.hooks.get_registry", return_value=mock_registry):
            registry = _resolve_hook_registry(trace_logger, session_id)

        assert registry is mock_registry
        events = trace_logger.load_session(session_id)
        hook_errors = [e for e in events if e.kind == "hook_registry_error"]
        assert not hook_errors


if __name__ == "__main__":
<<<<<<< HEAD
    pytest.main([__file__, "-v"])
=======
    pytest.main([__file__, "-v"])
>>>>>>> 66ceef6 (fix: resolve #260 — emit trace event when hook registry resolution fails)
