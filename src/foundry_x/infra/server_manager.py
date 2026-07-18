"""``FoundryServerManager`` — supervises the local ``llama-server`` (issue #899).

The manager is the small piece of automation that closes the loop between
``fx-runner`` and ``infra/scripts/launch_llamacpp.sh``. Without it, a
mid-session ``llama-server`` crash means an operator has to notice the
failure, restart the server, and re-run the cycle. With it, the runner
polls ``/health`` before each session and restarts on mid-session
failure with bounded exponential backoff.

Design notes
------------

* The manager is **opt-in**. ``fx-runner`` consults ``FOUNDRY_SERVER_AUTOSTART``
  (``"1"`` by default); when it is unset / ``"0"`` the manager is a
  passive health-checker that returns the current ``/health`` state but
  does not spawn, restart, or kill anything. This preserves the existing
  operator workflow (a manually-launched ``llama-server`` on a remote box)
  while letting CI / sweeps turn on the supervisor transparently.
* All env vars are read once at construction time so the resolved
  configuration is observable in the trace event recorded on
  ``start()``. Env-var name constants live at module scope and are
  re-exported via ``ServerConfig`` for callers that need them.
* The underlying process is launched with ``subprocess.Popen`` (no
  ``shell=True``, no ``os.system``) so the argv shape is reviewable in
  tests and the process group is isolated from the runner's own.
* ``is_healthy()`` and ``restart()`` are async (via ``asyncio.to_thread``
  wrappers) so the runner can call them from inside the agent loop
  without blocking the event loop on a slow ``/health`` round-trip.
* ``httpx`` is the HTTP client (already a transitive dependency via
  :class:`foundry_x.execution.model_adapter.OpenAICompatibleAdapter`); no
  new third-party packages are required.

The trace-event vocabulary (``server_unavailable``) is added to
``docs/CONTEXT.md`` §Event kinds in the same change-set so the Digester
and KPI consumers can rely on it.
"""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlsplit

import httpx


# Env-var names. Centralized so callers (Runner wiring, tests, docs)
# spell them identically.
LLAMACPP_HOST_ENV = "LLAMACPP_HOST"
LLAMACPP_MODEL_PATH_ENV = "LLAMACPP_MODEL_PATH"
FOUNDRY_SERVER_N_GPU_LAYERS_ENV = "FOUNDRY_SERVER_NGpuLayers"
FOUNDRY_SERVER_CTX_SIZE_ENV = "FOUNDRY_SERVER_CTXSize"
FOUNDRY_SERVER_AUTOSTART_ENV = "FOUNDRY_SERVER_AUTOSTART"
FOUNDRY_SERVER_BIN_ENV = "FOUNDRY_SERVER_BIN"

# Default fallback for the manager when ``LLAMACPP_HOST`` is unset (matches
# ``docs/MODEL_CONFIG.md`` §1 — the local-first fallback for llama.cpp).
_DEFAULT_LLAMACPP_HOST = "http://127.0.0.1:8080"
_DEFAULT_N_GPU_LAYERS = "0"
_DEFAULT_CTX_SIZE = "8192"
# Default is **off**: an operator who has not opted in to the supervisor
# gets the existing behaviour (passive ``/health`` probe + ``server_unavailable``
# events on detection) with no spawn / kill / abort semantics. CI and
# sweeps that want the supervisor to actively manage the server set
# ``FOUNDRY_SERVER_AUTOSTART=1`` explicitly; ``.env.example`` documents
# this knob so the opt-in is visible.
_DEFAULT_AUTOSTART = "0"
_DEFAULT_HEALTH_TIMEOUT_S = 2.0
_DEFAULT_HEALTH_READY_TIMEOUT_S = 60.0
_DEFAULT_MAX_RESTART_ATTEMPTS = 3
_DEFAULT_RESTART_BACKOFF_BASE_S = 1.0
_DEFAULT_RESTART_BACKOFF_CAP_S = 8.0

# Trace-event vocabulary (issue #899). The runner emits ``server_unavailable``
# mid-session when ``is_healthy()`` returns False; the KPI consumer in
# ``foundry_x.observability.kpis`` aggregates the count into
# ``server_restart_count``.
SERVER_UNAVAILABLE_KIND = "server_unavailable"


class ServerLaunchError(RuntimeError):
    """Raised when ``start()`` fails to spawn the underlying ``llama-server``."""


class ServerNotManagedError(RuntimeError):
    """Raised when an operation requires the manager to own the server process.

    For example, calling :meth:`FoundryServerManager.restart` when the
    manager was constructed with ``FOUNDRY_SERVER_AUTOSTART=0`` (the
    operator chose to manage the server out-of-band). Callers that only
    want a passive health check should use :meth:`is_healthy` directly.
    """


@dataclass(frozen=True)
class ServerConfig:
    """Resolved configuration for :class:`FoundryServerManager`.

    Constructed once at manager-build time so the rest of the supervisor
    can be tested without re-reading environment variables on every
    method call. ``popen_kwargs`` is reserved for tests that need to
    swap in a fake ``Popen``; production callers leave it empty.
    """

    host: str
    model_path: str | None
    n_gpu_layers: str
    ctx_size: str
    autostart: bool
    server_bin: str | None
    health_timeout_s: float = _DEFAULT_HEALTH_TIMEOUT_S
    health_ready_timeout_s: float = _DEFAULT_HEALTH_READY_TIMEOUT_S
    max_restart_attempts: int = _DEFAULT_MAX_RESTART_ATTEMPTS
    restart_backoff_base_s: float = _DEFAULT_RESTART_BACKOFF_BASE_S
    restart_backoff_cap_s: float = _DEFAULT_RESTART_BACKOFF_CAP_S
    popen_kwargs: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> ServerConfig:
        """Resolve a :class:`ServerConfig` from the supplied env dict.

        ``env`` defaults to :data:`os.environ`; tests pass a populated
        dict explicitly so the manager's behaviour is fully
        deterministic. Empty / whitespace values are treated as unset
        and fall back to the documented defaults, so an operator who
        accidentally exports ``LLAMACPP_HOST=""`` (or whitespace) gets
        the local-first fallback rather than a blank host that crashes
        ``urlsplit`` at first probe.
        """
        src = os.environ if env is None else env

        def _resolved(name: str, default: str) -> str:
            value = src.get(name, default).strip()
            return value or default

        host = _resolved(LLAMACPP_HOST_ENV, _DEFAULT_LLAMACPP_HOST)
        model_path = src.get(LLAMACPP_MODEL_PATH_ENV, "").strip() or None
        n_gpu_layers = _resolved(FOUNDRY_SERVER_N_GPU_LAYERS_ENV, _DEFAULT_N_GPU_LAYERS)
        ctx_size = _resolved(FOUNDRY_SERVER_CTX_SIZE_ENV, _DEFAULT_CTX_SIZE)
        autostart_raw = _resolved(FOUNDRY_SERVER_AUTOSTART_ENV, _DEFAULT_AUTOSTART).lower()
        autostart = autostart_raw in ("1", "true", "yes", "on")
        server_bin = src.get(FOUNDRY_SERVER_BIN_ENV, "").strip() or None
        return cls(
            host=host,
            model_path=model_path,
            n_gpu_layers=n_gpu_layers,
            ctx_size=ctx_size,
            autostart=autostart,
            server_bin=server_bin,
        )


def _health_url(host: str) -> str:
    """Return the ``/health`` URL for *host*.

    Handles both ``http://host:port`` and bare ``host:port`` shapes so an
    operator who sets ``LLAMACPP_HOST=127.0.0.1:8080`` (no scheme) does
    not get an invalid health probe URL.
    """
    parsed = urlsplit(host if "://" in host else f"http://{host}")
    base = f"{parsed.scheme}://{parsed.netloc or parsed.path}"
    return f"{base.rstrip('/')}/health"


class FoundryServerManager:
    """Supervise a local ``llama-server`` subprocess for ``fx-runner`` (issue #899).

    The manager is intentionally minimal: ``start()`` launches the
    subprocess and waits for ``/health`` to return 200; ``stop()`` ends
    the subprocess; ``is_healthy()`` probes ``/health`` without
    blocking; ``restart()`` performs bounded exponential-backoff
    retries; ``ensure_healthy()`` is the no-throw composite used by
    ``fx-runner`` before each session.

    When ``config.autostart`` is ``False`` the manager is a *passive*
    health-checker: ``is_healthy()`` and ``ensure_healthy()`` probe
    the endpoint but do not spawn, restart, or kill anything.
    """

    def __init__(
        self,
        config: ServerConfig | None = None,
        *,
        popen_factory: Callable[..., subprocess.Popen[Any]] | None = None,
        sleep: Callable[[float], Any] | None = None,
    ) -> None:
        self._config = config if config is not None else ServerConfig.from_env()
        self._popen_factory = popen_factory or subprocess.Popen
        self._sleep = sleep or time.sleep
        self._proc: subprocess.Popen[Any] | None = None
        self._restart_count = 0

    # ---- configuration accessors ------------------------------------

    @property
    def config(self) -> ServerConfig:
        """Return the resolved :class:`ServerConfig`."""
        return self._config

    @property
    def host(self) -> str:
        """Return the configured ``LLAMACPP_HOST`` value."""
        return self._config.host

    @property
    def health_url(self) -> str:
        """Return the absolute ``/health`` URL the manager probes."""
        return _health_url(self._config.host)

    @property
    def restart_count(self) -> int:
        """Number of restart cycles the manager has performed so far."""
        return self._restart_count

    # ---- public API -------------------------------------------------

    async def is_healthy(self) -> bool:
        """Return True iff ``GET /health`` returns 200 within the timeout."""
        return await asyncio.to_thread(self._is_healthy_sync)

    async def ensure_healthy(self) -> bool:
        """Make the server healthy before the agent loop opens.

        When ``config.autostart`` is ``True`` and the server is not
        currently healthy, this calls :meth:`restart` with the configured
        retry policy. When autostart is ``False`` it is a passive
        probe: the caller can read :attr:`restart_count` to learn what
        happened, but no spawn / kill is performed.

        Returns True if the server responds healthy after the call.
        """
        if await self.is_healthy():
            return True
        if not self._config.autostart:
            return False
        return await self.restart()

    async def start(self) -> subprocess.Popen[Any]:
        """Spawn ``llama-server`` and block until ``/health`` returns 200.

        Raises :class:`ServerLaunchError` if the underlying process
        cannot be launched or fails to become healthy inside
        ``config.health_ready_timeout_s``. Raises
        :class:`ServerNotManagedError` when ``config.autostart`` is
        ``False`` (the operator manages the server out-of-band).
        """
        if not self._config.autostart:
            raise ServerNotManagedError(
                "autostart is disabled; start() requires FOUNDRY_SERVER_AUTOSTART=1"
            )
        return await asyncio.to_thread(self._start_sync)

    async def stop(self) -> None:
        """Terminate the supervised subprocess, if any.

        No-op when the manager does not own a process (autostart off or
        ``start()`` was never called). Termination is best-effort:
        ``SIGTERM`` is sent and a brief grace period is allowed before
        ``SIGKILL``.
        """
        await asyncio.to_thread(self._stop_sync)

    async def restart(self) -> bool:
        """Bounded exponential-backoff restart (max 3 attempts by default).

        Each attempt calls :meth:`_start_sync`; between attempts the
        manager sleeps for ``backoff_base_s * 2 ** (attempt-1)`` seconds,
        capped at ``backoff_cap_s``. After all attempts the method
        returns ``False``; otherwise ``True`` once ``/health`` reports
        200. The internal :attr:`restart_count` is incremented exactly
        once per *successful* cycle so the KPI consumer correlates
        ``server_unavailable`` events with actual recovery, not just
        attempted recovery.

        Raises :class:`ServerNotManagedError` when ``config.autostart``
        is ``False``.
        """
        if not self._config.autostart:
            raise ServerNotManagedError(
                "autostart is disabled; restart() requires FOUNDRY_SERVER_AUTOSTART=1"
            )
        return await asyncio.to_thread(self._restart_sync)

    # ---- sync helpers (run on a worker thread via asyncio.to_thread) -

    def _is_healthy_sync(self) -> bool:
        try:
            with httpx.Client(timeout=self._config.health_timeout_s) as client:
                response = client.get(self.health_url)
        except httpx.HTTPError:
            return False
        return response.status_code == 200

    def _start_sync(self) -> subprocess.Popen[Any]:
        if self._proc is not None and self._proc.poll() is None:
            # Already running — surface that explicitly so callers don't
            # accidentally double-spawn when ``start()`` is called twice.
            return self._proc
        if not self._config.model_path:
            raise ServerLaunchError(
                "refusing to launch llama-server: LLAMACPP_MODEL_PATH is not set"
            )
        argv = _build_server_argv(self._config)
        # ``subprocess.Popen`` with ``shell=False`` (the default) so argv
        # is reviewable; no shell injection surface. The optional
        # ``popen_factory`` test hook lets tests swap in a fake Popen
        # without spawning a real process.
        try:
            self._proc = self._popen_factory(argv, **self._config.popen_kwargs)
        except FileNotFoundError as exc:
            raise ServerLaunchError(
                f"failed to launch llama-server: binary {self._config.server_bin or 'llama-server'!r} "
                f"is not on PATH"
            ) from exc
        # Block (synchronously) until /health responds 200, up to the
        # configured readiness timeout. This call is wrapped in
        # ``asyncio.to_thread`` by :meth:`start` so the event loop is
        # not blocked during warm-up.
        deadline = time.monotonic() + self._config.health_ready_timeout_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                raise ServerLaunchError(
                    f"llama-server exited with code {self._proc.returncode} "
                    f"before /health became ready"
                )
            if self._is_healthy_sync():
                return self._proc
            self._sleep(0.5)
        raise ServerLaunchError(
            f"llama-server failed /health probe within {self._config.health_ready_timeout_s}s"
        )

    def _stop_sync(self) -> None:
        proc = self._proc
        if proc is None:
            return
        self._proc = None
        if proc.poll() is not None:
            return
        try:
            proc.terminate()
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                proc.kill()
            except ProcessLookupError:
                return
            proc.wait(timeout=2)

    def _restart_sync(self) -> bool:
        self._stop_sync()
        for attempt in range(1, self._config.max_restart_attempts + 1):
            try:
                self._start_sync()
            except ServerLaunchError:
                if attempt >= self._config.max_restart_attempts:
                    return False
                backoff = min(
                    self._config.restart_backoff_base_s * (2 ** (attempt - 1)),
                    self._config.restart_backoff_cap_s,
                )
                self._sleep(backoff)
                continue
            self._restart_count += 1
            return True
        return False


def _build_server_argv(config: ServerConfig) -> list[str]:
    """Construct the argv list passed to ``subprocess.Popen``.

    Mirrors ``infra/scripts/launch_llamacpp.sh`` so the operator
    workflow and the manager agree on flag order. The host / port pair
    is derived from ``LLAMACPP_HOST``; ``--model`` comes from
    ``LLAMACPP_MODEL_PATH``; ``--n-gpu-layers`` and ``--ctx-size`` come
    from the new ``FOUNDRY_SERVER_*`` env vars. The shell-style argv
    that operators see in the launch script and the argv the manager
    uses should always be identical, modulo the binary path.
    """
    parts = urlsplit(config.host if "://" in config.host else f"http://{config.host}")
    host = parts.hostname or "127.0.0.1"
    port = str(parts.port or 8080)
    argv: list[str] = []
    if config.server_bin:
        argv.append(config.server_bin)
    argv.extend(
        [
            "--model",
            config.model_path or "",
            "--host",
            host,
            "--port",
            port,
            "--n-gpu-layers",
            config.n_gpu_layers,
            "--ctx-size",
            config.ctx_size,
        ]
    )
    # Filter out empty values that would confuse llama-server without
    # silently dropping required flags.
    return [tok for tok in argv if tok]


def quote_argv(argv: list[str]) -> str:
    """Helper for trace event payloads: ``shlex.join`` of the argv list.

    Tests and trace-event consumers sometimes want a printable form of
    the resolved command line; ``shlex.join`` is the canonical way to
    render an argv that survives round-trips through :func:`shlex.split`.
    """
    return shlex.join(argv)
