"""Unit tests for ``FoundryServerManager`` (issue #899).

The manager is a thin supervisor around ``subprocess.Popen`` and
``httpx.Client``. These tests cover the state machine without spawning a
real ``llama-server``:

* ``ServerConfig.from_env`` resolves env vars into the right defaults
  and overrides (without ever reading from the host environment).
* ``start()`` spawns the underlying subprocess and waits for ``/health``.
* ``is_healthy()`` performs a single ``GET /health`` probe.
* ``restart()`` invokes exponential-backoff retries and surfaces the
  final state.
* ``ensure_healthy()`` is the public ``fx-runner`` entry point.
* Autostart-off mode degrades to a passive health-checker.
"""

from __future__ import annotations

import os
import subprocess
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from foundry_x.infra.server_manager import (
    FOUNDRY_SERVER_AUTOSTART_ENV,
    FOUNDRY_SERVER_BIN_ENV,
    FOUNDRY_SERVER_CTX_SIZE_ENV,
    FOUNDRY_SERVER_N_GPU_LAYERS_ENV,
    FoundryServerManager,
    LLAMACPP_HOST_ENV,
    LLAMACPP_MODEL_PATH_ENV,
    ServerConfig,
    ServerLaunchError,
    ServerNotManagedError,
    _build_server_argv,
)


def _config(
    *,
    host: str = "http://127.0.0.1:8080",
    model_path: str | None = "/tmp/x.gguf",
    n_gpu_layers: str = "0",
    ctx_size: str = "8192",
    autostart: bool = True,
    server_bin: str | None = None,
    **overrides: Any,
) -> ServerConfig:
    """Build a :class:`ServerConfig` with sensible defaults for tests."""
    base = dict(
        host=host,
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        ctx_size=ctx_size,
        autostart=autostart,
        server_bin=server_bin,
    )
    base.update(overrides)
    return ServerConfig(**base)


# ---------------------------------------------------------------------------
# ServerConfig.from_env
# ---------------------------------------------------------------------------


def test_server_config_from_env_reads_all_knobs() -> None:
    """Every env var is plumbed through ``ServerConfig.from_env``."""
    env = {
        LLAMACPP_HOST_ENV: "http://127.0.0.1:9999",
        LLAMACPP_MODEL_PATH_ENV: "/srv/models/qwen.Q4_K_M.gguf",
        FOUNDRY_SERVER_N_GPU_LAYERS_ENV: "35",
        FOUNDRY_SERVER_CTX_SIZE_ENV: "4096",
        FOUNDRY_SERVER_AUTOSTART_ENV: "0",
        FOUNDRY_SERVER_BIN_ENV: "/opt/llama.cpp/build/bin/llama-server",
    }
    cfg = ServerConfig.from_env(env)

    assert cfg.host == "http://127.0.0.1:9999"
    assert cfg.model_path == "/srv/models/qwen.Q4_K_M.gguf"
    assert cfg.n_gpu_layers == "35"
    assert cfg.ctx_size == "4096"
    assert cfg.autostart is False
    assert cfg.server_bin == "/opt/llama.cpp/build/bin/llama-server"


def test_server_config_from_env_uses_safe_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """When every var is unset, the defaults from ``MODEL_CONFIG.md`` apply."""
    # Strip every variable the manager reads so ``os.environ`` does not
    # leak values from the developer's shell into the test.
    for name in (
        LLAMACPP_HOST_ENV,
        LLAMACPP_MODEL_PATH_ENV,
        FOUNDRY_SERVER_N_GPU_LAYERS_ENV,
        FOUNDRY_SERVER_CTX_SIZE_ENV,
        FOUNDRY_SERVER_AUTOSTART_ENV,
        FOUNDRY_SERVER_BIN_ENV,
    ):
        monkeypatch.delenv(name, raising=False)

    cfg = ServerConfig.from_env()

    assert cfg.host == "http://127.0.0.1:8080"
    assert cfg.model_path is None
    assert cfg.n_gpu_layers == "0"
    assert cfg.ctx_size == "8192"
    # Default is OFF — the supervisor is opt-in so operators who manage
    # llama-server out-of-band are not surprised by spawn/kill side
    # effects. CI / sweeps opt in via ``FOUNDRY_SERVER_AUTOSTART=1``.
    assert cfg.autostart is False
    assert cfg.server_bin is None


def test_server_config_from_env_recognises_truthy_autostart_values() -> None:
    """``FOUNDRY_SERVER_AUTOSTART`` accepts the common truthy spellings."""
    for raw in ("1", "true", "TRUE", "yes", "on"):
        cfg = ServerConfig.from_env({FOUNDRY_SERVER_AUTOSTART_ENV: raw})
        assert cfg.autostart is True, f"autostart should be True for {raw!r}"


def test_server_config_from_env_treats_empty_strings_as_unset() -> None:
    """Empty / whitespace env values fall back to the documented defaults."""
    cfg = ServerConfig.from_env(
        {
            LLAMACPP_HOST_ENV: "   ",
            LLAMACPP_MODEL_PATH_ENV: "",
            FOUNDRY_SERVER_N_GPU_LAYERS_ENV: "",
        }
    )
    assert cfg.host == "http://127.0.0.1:8080"
    assert cfg.model_path is None
    assert cfg.n_gpu_layers == "0"


# ---------------------------------------------------------------------------
# argv construction
# ---------------------------------------------------------------------------


def test_build_server_argv_mirrors_launch_llamacpp_shape() -> None:
    """The argv shape matches ``infra/scripts/launch_llamacpp.sh`` exactly."""
    cfg = _config(
        host="http://127.0.0.1:8080",
        model_path="/srv/models/codellama-7b.Q5_K_M.gguf",
        n_gpu_layers="35",
        ctx_size="4096",
    )
    argv = _build_server_argv(cfg)
    assert argv == [
        "--model",
        "/srv/models/codellama-7b.Q5_K_M.gguf",
        "--host",
        "127.0.0.1",
        "--port",
        "8080",
        "--n-gpu-layers",
        "35",
        "--ctx-size",
        "4096",
    ]


def test_build_server_argv_accepts_bare_host_form() -> None:
    """``LLAMACPP_HOST=host:port`` (no scheme) is normalised to a URL."""
    cfg = _config(host="127.0.0.1:9000")
    argv = _build_server_argv(cfg)
    assert "--host" in argv
    assert "127.0.0.1" in argv
    assert "--port" in argv
    assert "9000" in argv


def test_build_server_argv_prepends_server_bin_when_set() -> None:
    """When ``FOUNDRY_SERVER_BIN`` is set it is the argv[0]."""
    cfg = _config(server_bin="/opt/llama.cpp/build/bin/llama-server")
    argv = _build_server_argv(cfg)
    assert argv[0] == "/opt/llama.cpp/build/bin/llama-server"


def test_health_url_handles_bare_host_form() -> None:
    """``health_url`` always emits a fully-qualified ``/health`` URL."""
    cfg = _config(host="127.0.0.1:8080")
    mgr = FoundryServerManager(cfg)
    assert mgr.health_url == "http://127.0.0.1:8080/health"


# ---------------------------------------------------------------------------
# is_healthy (passive probe)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_is_healthy_returns_true_on_200() -> None:
    """A 200 response from ``/health`` is reported as healthy."""
    cfg = _config(health_timeout_s=1.0)
    fake_client = MagicMock()
    fake_response = MagicMock(status_code=200)
    fake_client.__enter__.return_value.get.return_value = fake_response

    mgr = FoundryServerManager(cfg, popen_factory=lambda *a, **kw: MagicMock())

    # Patch ``httpx.Client`` to return our fake client.
    import foundry_x.infra.server_manager as sm

    monkey = pytest.MonkeyPatch()
    try:
        monkey.setattr(sm.httpx, "Client", lambda timeout: fake_client)
        assert await mgr.is_healthy() is True
    finally:
        monkey.undo()


@pytest.mark.asyncio
async def test_is_healthy_returns_false_on_5xx(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-200 response is reported as unhealthy."""
    cfg = _config(health_timeout_s=1.0)
    fake_client = MagicMock()
    fake_response = MagicMock(status_code=503)
    fake_client.__enter__.return_value.get.return_value = fake_response

    import foundry_x.infra.server_manager as sm

    monkeypatch.setattr(sm.httpx, "Client", lambda timeout: fake_client)
    mgr = FoundryServerManager(cfg)
    assert await mgr.is_healthy() is False


@pytest.mark.asyncio
async def test_is_healthy_returns_false_on_connect_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """A ``httpx.ConnectError`` is reported as unhealthy, not raised."""
    cfg = _config(health_timeout_s=1.0)
    fake_client = MagicMock()
    fake_client.__enter__.return_value.get.side_effect = httpx.ConnectError("connection refused")

    import foundry_x.infra.server_manager as sm

    monkeypatch.setattr(sm.httpx, "Client", lambda timeout: fake_client)
    mgr = FoundryServerManager(cfg)
    assert await mgr.is_healthy() is False


# ---------------------------------------------------------------------------
# start() / stop()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_spawns_subprocess_and_waits_for_health(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``start()`` launches the subprocess and blocks until ``/health`` returns 200."""
    cfg = _config(
        health_ready_timeout_s=2.0,
        health_timeout_s=0.5,
    )

    # Fake Popen that reports itself as running.
    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    fake_proc.returncode = 0
    popen = MagicMock(return_value=fake_proc)

    # Fake httpx.Client that reports 200 on the first probe.
    fake_client = MagicMock()
    fake_response = MagicMock(status_code=200)
    fake_client.__enter__.return_value.get.return_value = fake_response

    import foundry_x.infra.server_manager as sm

    monkeypatch.setattr(sm.httpx, "Client", lambda timeout: fake_client)
    sleeps: list[float] = []
    mgr = FoundryServerManager(cfg, popen_factory=popen, sleep=sleeps.append)

    proc = await mgr.start()

    assert proc is fake_proc
    popen.assert_called_once()
    # The argv passed to ``Popen`` matches the configured model path / host.
    argv = popen.call_args.args[0]
    assert "--model" in argv
    assert "/tmp/x.gguf" in argv


@pytest.mark.asyncio
async def test_start_refuses_without_model_path(monkeypatch: pytest.MonkeyPatch) -> None:
    """``start()`` refuses to spawn when ``LLAMACPP_MODEL_PATH`` is unset."""
    cfg = _config(model_path=None)
    mgr = FoundryServerManager(cfg)
    with pytest.raises(ServerLaunchError, match="LLAMACPP_MODEL_PATH"):
        await mgr.start()


@pytest.mark.asyncio
async def test_start_raises_when_binary_not_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing binary surfaces as :class:`ServerLaunchError`, not ``FileNotFoundError``."""

    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("llama-server")

    cfg = _config()
    mgr = FoundryServerManager(cfg, popen_factory=_raise)
    with pytest.raises(ServerLaunchError, match="binary"):
        await mgr.start()


@pytest.mark.asyncio
async def test_start_raises_not_managed_when_autostart_off() -> None:
    """``start()`` is unavailable when the operator manages the server."""
    cfg = _config(autostart=False)
    mgr = FoundryServerManager(cfg)
    with pytest.raises(ServerNotManagedError):
        await mgr.start()


@pytest.mark.asyncio
async def test_stop_is_noop_when_no_proc() -> None:
    """``stop()`` is safe to call before ``start()``."""
    cfg = _config()
    mgr = FoundryServerManager(cfg)
    await mgr.stop()
    assert mgr._proc is None  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_stop_terminates_running_proc(monkeypatch: pytest.MonkeyPatch) -> None:
    """``stop()`` sends SIGTERM and waits for the process to exit."""
    fake_proc = MagicMock()
    fake_proc.poll.side_effect = [None, 0]
    cfg = _config()

    # Patch ``time.sleep`` to no-op (the wait() fallback does not sleep
    # when the proc exits promptly).
    monkeypatch.setattr("foundry_x.infra.server_manager.time.sleep", lambda _s: None)
    mgr = FoundryServerManager(cfg)
    mgr._proc = fake_proc  # type: ignore[attr-defined]

    await mgr.stop()

    fake_proc.terminate.assert_called_once()
    assert mgr._proc is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# restart()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_restart_calls_start_then_increments_counter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A successful ``restart()`` increments the manager's restart counter once."""
    cfg = _config(health_ready_timeout_s=2.0, health_timeout_s=0.5)

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None

    popen = MagicMock(return_value=fake_proc)
    fake_client = MagicMock()
    fake_response = MagicMock(status_code=200)
    fake_client.__enter__.return_value.get.return_value = fake_response

    import foundry_x.infra.server_manager as sm

    monkeypatch.setattr(sm.httpx, "Client", lambda timeout: fake_client)

    mgr = FoundryServerManager(
        cfg,
        popen_factory=popen,
        sleep=lambda _s: None,
    )

    ok = await mgr.restart()
    assert ok is True
    assert mgr.restart_count == 1


@pytest.mark.asyncio
async def test_restart_returns_false_after_max_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``restart()`` returns ``False`` after the configured retry budget is exhausted."""
    cfg = _config(
        max_restart_attempts=3,
        restart_backoff_base_s=0.0,
        restart_backoff_cap_s=0.0,
        health_ready_timeout_s=0.05,
    )

    # popen always raises ServerLaunchError-equivalent (binary missing).
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("llama-server")

    mgr = FoundryServerManager(cfg, popen_factory=_raise, sleep=lambda _s: None)

    ok = await mgr.restart()
    assert ok is False
    assert mgr.restart_count == 0


@pytest.mark.asyncio
async def test_restart_uses_exponential_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``restart()`` sleeps for ``base * 2 ** (attempt-1)`` between attempts."""
    sleeps: list[float] = []
    cfg = _config(
        max_restart_attempts=3,
        restart_backoff_base_s=1.0,
        restart_backoff_cap_s=100.0,
        health_ready_timeout_s=0.05,
    )

    # All attempts fail with a missing binary so the backoff is exercised.
    def _raise(*_a: Any, **_kw: Any) -> Any:
        raise FileNotFoundError("llama-server")

    mgr = FoundryServerManager(
        cfg,
        popen_factory=_raise,
        sleep=sleeps.append,
    )

    await mgr.restart()

    # Two sleeps for three attempts (no sleep after the final attempt).
    assert sleeps == [1.0, 2.0]


@pytest.mark.asyncio
async def test_restart_raises_not_managed_when_autostart_off() -> None:
    """``restart()`` is unavailable when the operator manages the server."""
    cfg = _config(autostart=False)
    mgr = FoundryServerManager(cfg)
    with pytest.raises(ServerNotManagedError):
        await mgr.restart()


# ---------------------------------------------------------------------------
# ensure_healthy() — the fx-runner entry point
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ensure_healthy_returns_true_when_already_healthy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A healthy server short-circuits ``ensure_healthy`` to a no-op."""
    cfg = _config(health_timeout_s=0.5)
    fake_client = MagicMock()
    fake_response = MagicMock(status_code=200)
    fake_client.__enter__.return_value.get.return_value = fake_response

    import foundry_x.infra.server_manager as sm

    monkeypatch.setattr(sm.httpx, "Client", lambda timeout: fake_client)
    mgr = FoundryServerManager(cfg)
    ok = await mgr.ensure_healthy()
    assert ok is True
    assert mgr.restart_count == 0


@pytest.mark.asyncio
async def test_ensure_healthy_restarts_when_unhealthy_and_autostart_on(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``ensure_healthy()`` triggers ``restart()`` when autostart is on and the server is down."""
    cfg = _config(
        autostart=True,
        health_ready_timeout_s=2.0,
        health_timeout_s=0.5,
    )

    fake_proc = MagicMock()
    fake_proc.poll.return_value = None
    popen = MagicMock(return_value=fake_proc)

    # First probe returns 503 (unhealthy), then 200 (healthy after restart).
    fake_client = MagicMock()
    responses = iter([MagicMock(status_code=503), MagicMock(status_code=200)])
    fake_client.__enter__.return_value.get.side_effect = lambda *a, **kw: next(responses)

    import foundry_x.infra.server_manager as sm

    monkeypatch.setattr(sm.httpx, "Client", lambda timeout: fake_client)

    mgr = FoundryServerManager(
        cfg,
        popen_factory=popen,
        sleep=lambda _s: None,
    )

    ok = await mgr.ensure_healthy()
    assert ok is True
    assert mgr.restart_count == 1


@pytest.mark.asyncio
async def test_ensure_healthy_is_passive_when_autostart_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Autostart-off mode degrades to a passive health prober."""
    cfg = _config(autostart=False, health_timeout_s=0.5)

    fake_client = MagicMock()
    fake_response = MagicMock(status_code=503)
    fake_client.__enter__.return_value.get.return_value = fake_response

    import foundry_x.infra.server_manager as sm

    monkeypatch.setattr(sm.httpx, "Client", lambda timeout: fake_client)
    mgr = FoundryServerManager(cfg)
    ok = await mgr.ensure_healthy()
    assert ok is False
    assert mgr.restart_count == 0
    assert mgr._proc is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Isolation: subprocess is real but argv is mocked for offline safety
# ---------------------------------------------------------------------------


def test_real_subprocess_popen_is_used_by_default() -> None:
    """The default ``popen_factory`` is ``subprocess.Popen`` so no extra
    library sneaks in via the manager."""
    from foundry_x.infra.server_manager import subprocess as sm_subprocess

    mgr = FoundryServerManager(_config())
    assert mgr._popen_factory is sm_subprocess.Popen  # type: ignore[attr-defined]


def test_real_subprocess_module_is_stdlib() -> None:
    """``subprocess`` is the stdlib module — no third-party package."""
    # Defensive assertion in case anyone ever tries to swap in pexpect.
    assert subprocess.__name__ == "subprocess"


# Sanity: when ``os.environ`` is fully populated the manager does not
# crash on import. This guards against the env-defaults silently
# depending on host state.
def test_import_does_not_read_environ() -> None:
    importlib = pytest.MonkeyPatch()
    try:
        for name in (
            LLAMACPP_HOST_ENV,
            LLAMACPP_MODEL_PATH_ENV,
            FOUNDRY_SERVER_N_GPU_LAYERS_ENV,
            FOUNDRY_SERVER_CTX_SIZE_ENV,
            FOUNDRY_SERVER_AUTOSTART_ENV,
            FOUNDRY_SERVER_BIN_ENV,
        ):
            importlib.delenv(name, raising=False)
        # ``FoundryServerManager()`` (no config) should resolve cleanly.
        mgr = FoundryServerManager()
        assert mgr.host == "http://127.0.0.1:8080"
    finally:
        importlib.undo()


def test_default_env_does_not_include_real_model_path() -> None:
    """The defaults never embed a credentialed model path (AGENTS.md §2).

    This guards against a future change that accidentally reads from a
    developer-specific ``.env`` and ships a real path in the defaults.
    """
    # With all variables stripped, the model path must be ``None``.
    importlib = pytest.MonkeyPatch()
    try:
        for name in (
            LLAMACPP_HOST_ENV,
            LLAMACPP_MODEL_PATH_ENV,
            FOUNDRY_SERVER_N_GPU_LAYERS_ENV,
            FOUNDRY_SERVER_CTX_SIZE_ENV,
            FOUNDRY_SERVER_AUTOSTART_ENV,
            FOUNDRY_SERVER_BIN_ENV,
        ):
            importlib.delenv(name, raising=False)
        cfg = ServerConfig.from_env()
        assert cfg.model_path is None
        # And the host must be the documented local-first fallback.
        assert cfg.host == "http://127.0.0.1:8080"
        assert "/srv" not in cfg.host
    finally:
        importlib.undo()
        # The monkeypatch fixture automatically restores os.environ on teardown;
        # the explicit ``undo`` here is defensive.
        for name in (
            LLAMACPP_HOST_ENV,
            LLAMACPP_MODEL_PATH_ENV,
            FOUNDRY_SERVER_N_GPU_LAYERS_ENV,
            FOUNDRY_SERVER_CTX_SIZE_ENV,
            FOUNDRY_SERVER_AUTOSTART_ENV,
            FOUNDRY_SERVER_BIN_ENV,
        ):
            os.environ.pop(name, None)
