"""Subprocess coverage for infra/scripts/wait_for_llamacpp.sh.

Tests follow the same pattern as tests/infra/test_rocm_setup.py: run the
real script in a subprocess against either a stub http.server (for the
success case) or an unreachable port (for the timeout case).
"""

from __future__ import annotations

import http.server
import os
import shutil
import socket
import subprocess
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "infra" / "scripts" / "wait_for_llamacpp.sh"


def _bash_binary() -> str:
    bash = shutil.which("bash")
    if bash is None:
        pytest.skip("bash is not on PATH; skipping wait_for_llamacpp.sh subprocess tests")
    return bash


def _run(
    *args: str, env: dict[str, str] | None = None, timeout: float = 30
) -> subprocess.CompletedProcess[str]:
    run_env = os.environ.copy()
    run_env.pop("LLAMACPP_WAIT_HOST", None)
    run_env.pop("LLAMACPP_WAIT_PORT", None)
    run_env.pop("LLAMACPP_WAIT_TIMEOUT", None)
    if env is not None:
        run_env.update(env)
    return subprocess.run(
        [_bash_binary(), str(SCRIPT), *args],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        env=run_env,
    )


class _StubHandler(http.server.BaseHTTPRequestHandler):
    """Always respond 200 OK on any GET — mimics llama-server /health."""

    def do_GET(self) -> None:  # noqa: N802 – required by BaseHTTPRequestHandler
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"status":"ok"}')

    def log_message(self, *args: object) -> None:
        pass


def _start_stub_server() -> tuple[int, threading.Thread, http.server.HTTPServer]:
    """Start a background http.server on a free port; return (port, thread, server)."""
    server = http.server.HTTPServer(("127.0.0.1", 0), _StubHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return port, thread, server


def _free_port() -> int:
    """Return a port number that is almost certainly unused (open+close a socket)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -- --help ------------------------------------------------------------------


def test_wait_for_llamacpp_help_exits_zero() -> None:
    result = _run("--help")

    assert result.returncode == 0
    assert "usage: wait_for_llamacpp.sh" in result.stdout
    assert "--host HOST" in result.stdout
    assert "--port PORT" in result.stdout
    assert "--timeout SECS" in result.stdout
    assert "--quiet" in result.stdout


def test_wait_for_llamacpp_help_short_flag() -> None:
    result = _run("-h")
    assert result.returncode == 0


# -- bad usage ---------------------------------------------------------------


def test_wait_for_llamacpp_unknown_flag_exits_two() -> None:
    result = _run("--bogus")
    assert result.returncode == 2
    assert "unknown argument" in result.stderr


def test_wait_for_llamacpp_host_missing_value_exits_two() -> None:
    result = _run("--host")
    assert result.returncode == 2
    assert "--host requires a value" in result.stderr


def test_wait_for_llamacpp_port_missing_value_exits_two() -> None:
    result = _run("--port")
    assert result.returncode == 2
    assert "--port requires a value" in result.stderr


def test_wait_for_llamacpp_timeout_missing_value_exits_two() -> None:
    result = _run("--timeout")
    assert result.returncode == 2
    assert "--timeout requires a value" in result.stderr


# -- exit 0 on 200 OK --------------------------------------------------------


def test_wait_for_llamacpp_exits_zero_on_200_ok() -> None:
    port, _thread, server = _start_stub_server()
    try:
        result = _run("--host", "127.0.0.1", "--port", str(port), "--timeout", "5")
    finally:
        server.shutdown()

    assert result.returncode == 0, result.stderr


def test_wait_for_llamacpp_quiet_suppresses_output_on_success() -> None:
    port, _thread, server = _start_stub_server()
    try:
        result = _run(
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--timeout",
            "5",
            "--quiet",
        )
    finally:
        server.shutdown()

    assert result.returncode == 0
    assert result.stderr == ""


# -- exit 1 on timeout -------------------------------------------------------


def test_wait_for_llamacpp_exits_one_on_timeout() -> None:
    port = _free_port()
    result = _run("--host", "127.0.0.1", "--port", str(port), "--timeout", "2")

    assert result.returncode == 1


def test_wait_for_llamacpp_quiet_suppresses_output_on_timeout() -> None:
    port = _free_port()
    result = _run(
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--timeout",
        "2",
        "--quiet",
    )

    assert result.returncode == 1
    assert result.stderr == ""
