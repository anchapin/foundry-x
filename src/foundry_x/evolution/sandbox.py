from __future__ import annotations

import subprocess
import uuid
from pathlib import Path
from typing import Any

import pydantic


class SandboxConfig(pydantic.BaseModel):
    """Configuration for a Docker sandbox container.

    Attributes:
        image: Docker image to run (must match the built foundryx image).
        logs_dir: Host path to mount as /app/logs inside the container.
        uv_cache_dir: Host path to mount as the UV cache tmpfs inside the
            container.  Defaults to a temporary directory that is cleaned up
            when the sandbox exits.
    """

    image: str = "foundryx:latest"
    logs_dir: Path | None = None
    uv_cache_dir: Path | None = None


class SandboxRuntimeError(RuntimeError):
    """Raised when the Docker container lifecycle cannot be completed."""


class DockerSandbox:
    """Per-evaluation Docker container lifecycle manager.

    Each ``evaluate`` call creates a *new* container, runs the evaluation
    steps inside it, and tears the container down before returning.  The
    live ``harness_dir`` is never mutated; all filesystem work happens in
    a temporary harness copy that is bind-mounted into the container.

    This class is a context manager.  Prefer using it with ``with``:

        with DockerSandbox(harness_dir, config) as sandbox:
            result = sandbox.run(["pytest", ...])
    """

    def __init__(
        self,
        harness_copy: Path,
        config: SandboxConfig | None = None,
    ) -> None:
        self._harness_copy = harness_copy
        self._config = config or SandboxConfig()
        self._container_name: str | None = None
        self._uv_cache: Path | None = None

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def run(
        self, argv: list[str], *, cwd: Path | str | None = None
    ) -> subprocess.CompletedProcess[str]:
        """Run *argv* inside the sandbox container via ``docker exec``.

        Args:
            argv: Command and arguments to execute.  The first element is
                resolved relative to the container's ``PATH`` (i.e. use
                ``["pytest", ...]`` not ``["/app/.venv/bin/pytest", ...]``).
            cwd: Working directory inside the container.  Defaults to
                ``/app`` (the image's ``WORKDIR``).

        Returns:
            A ``CompletedProcess`` with ``stdout``, ``stderr``, and
            ``returncode`` from the *argv* execution inside the container.

        Raises:
            SandboxRuntimeError: if no container is running or the exec
                command fails in a way that prevents further evaluation.
        """
        if self._container_name is None:
            raise SandboxRuntimeError("No container running.  Use `with DockerSandbox(...)`.")

        working_dir = str(cwd) if cwd is not None else "/app"
        exec_argv = ["docker", "exec", "--workdir", working_dir, self._container_name, *argv]
        return subprocess.run(
            exec_argv,
            capture_output=True,
            text=True,
        )

    def teardown(self) -> None:
        """Stop and remove the evaluation container, then clean up temp files."""
        if self._container_name is not None:
            subprocess.run(
                ["docker", "stop", "-t=10", self._container_name],
                capture_output=True,
            )
            subprocess.run(
                ["docker", "rm", "-f", self._container_name],
                capture_output=True,
            )
            self._container_name = None

    # -------------------------------------------------------------------------
    # Context manager
    # -------------------------------------------------------------------------

    def __enter__(self) -> DockerSandbox:
        self._start()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        self.teardown()

    # -------------------------------------------------------------------------
    # Container lifecycle (private)
    # -------------------------------------------------------------------------

    def _unique_name(self) -> str:
        return f"critic-eval-{uuid.uuid4().hex[:8]}"

    def _start(self) -> None:
        """Create and start the evaluation container.

        The container is named so it can be uniquely addressed and reliably
        torn down even if a previous step crashed.  The ``--rm`` flag is NOT
        used because we need the container to survive ``docker exec`` calls
        and only be removed in ``teardown()``.
        """
        name = self._unique_name()
        self._container_name = name

        volume_args = self._volume_args()

        run_argv = [
            "docker",
            "run",
            "--detach",
            "--name",
            name,
            *volume_args,
            self._config.image,
            "sleep",
            "infinity",  # Keep container alive for subsequent exec calls
        ]
        result = subprocess.run(run_argv, capture_output=True, text=True)
        if result.returncode != 0:
            self._container_name = None
            raise SandboxRuntimeError(
                f"docker run failed:\nstdout: {result.stdout}\nstderr: {result.stderr}"
            )

    def _volume_args(self) -> list[str]:
        """Build the ``-v host:container`` and ``--tmpfs`` arguments for docker run."""
        args: list[str] = []

        # Harness copy (read-only inside container so the eval cannot mutate it;
        # the host-side copy is already the temp sandbox).
        args.extend(
            [
                "-v",
                f"{self._harness_copy}:/app/harness:ro",
            ]
        )

        # Read-only source mounts: the container exercises the live source
        # at /app/src and /app/tests without receiving writes from the eval.
        repo_root = (
            self._harness_copy.parent.parent
        )  # <tmp>/critic-sandbox-<uuid>/harness -> <tmp>/critic-sandbox-<uuid> -> <tmp>
        args.extend(
            [
                "-v",
                f"{repo_root / 'src'}:/app/src:ro",
                "-v",
                f"{repo_root / 'tests'}:/app/tests:ro",
            ]
        )

        # Logs directory: trace events are persisted to the host so they are
        # visible after the container exits.
        if self._config.logs_dir is not None:
            args.extend(["-v", f"{self._config.logs_dir}:/app/logs"])
        else:
            # Fall back to a throwaway tmpfs when no logs_dir is configured;
            # evaluation traces are discarded but the run still succeeds.
            args.append("--tmpfs=/app/logs")

        # UV cache: bound to a host directory so uv's resolver state is
        # reused across eval runs and is not lost when the container exits.
        # When no explicit path is given, a private temp directory is created
        # and cleaned up in teardown().
        if self._config.uv_cache_dir is not None:
            uv_cache = self._config.uv_cache_dir
        else:
            if self._uv_cache is None:
                import tempfile

                self._uv_cache = Path(tempfile.mkdtemp(prefix="critic-uv-cache-"))
            uv_cache = self._uv_cache
        args.extend(["-v", f"{uv_cache}:/tmp/uv-cache"])

        # /tmp and /var/tmp as tmpfs (matches the compose file hardening;
        # required so Python tempfile and sqlite WAL work when the root FS
        # is read-only).
        args.extend(["--tmpfs", "/tmp:size=512m,mode=1777"])
        args.extend(["--tmpfs", "/var/tmp:size=256m,mode=1777"])

        return args
