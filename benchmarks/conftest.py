"""Shared pytest fixtures for the benchmark suite.

This conftest makes the ``benchmark_workspace`` fixture available to every
test under ``benchmarks/`` (and its subdirectories, e.g. ``tasks/``). It is
the isolation boundary the local development path relies on until full
Docker sandboxing lands (SECURITY.md, "Sandbox").

The ``benchmark_workspace`` fixture
-----------------------------------
``benchmark_workspace`` yields a :class:`pathlib.Path` to an empty, private
temporary directory that is removed automatically when the test finishes
(it is layered on pytest's built-in ``tmp_path``, so teardown is handled by
pytest itself). Benchmark tasks MUST treat this directory as the agent's
entire filesystem view: write outputs here, read seeded inputs here, and
never touch the repository tree.

Seeding static fixtures (optional)
----------------------------------
A task that needs static inputs from ``benchmarks/fixtures/<name>/`` asks
the fixture to copy them in via indirect parametrization::

    @pytest.mark.parametrize(
        "benchmark_workspace", ["sort_a_list"], indirect=True
    )
    @pytest.mark.benchmark
    def test_sort_a_list(benchmark_workspace: Path) -> None:
        assert (benchmark_workspace / "input.txt").exists()

When the parameter is omitted (the common case) the workspace is empty. If
the named fixture directory does not exist, the fixture raises
``FileNotFoundError`` at setup time so the mistake surfaces loudly rather
than silently producing a false pass (PHILOSOPHY.md, "Evidence over
opinion").

Isolation guarantee
-------------------
The yielded path is unique per test invocation; no two tests share it and
nothing escapes into the repository tree. That is what makes the local
development path safe in lieu of full Docker sandboxing.

See also ``benchmarks/README.md`` ("The benchmark_workspace fixture") and
ADR-0004 / ADR-0005.
"""

from __future__ import annotations

import os
import shutil
import time
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest

#: Root directory for static benchmark fixture data (``benchmarks/fixtures/``).
FIXTURES_ROOT = Path(__file__).parent / "fixtures"

#: Default timeout for llama-server /health polling.
_LLAMACPP_HEALTH_TIMEOUT = 60


def _seed_workspace(workspace: Path, seed_name: str) -> None:
    """Copy ``benchmarks/fixtures/<seed_name>/`` into ``workspace``.

    Raises ``FileNotFoundError`` if the named fixture directory is missing so
    that a typo in a task's ``indirect`` parameter fails fast instead of
    yielding an empty workspace that masks the mistake.
    """
    source = FIXTURES_ROOT / seed_name
    if not source.is_dir():
        raise FileNotFoundError(
            f"benchmark fixture directory not found: {source} "
            "(requested via @pytest.mark.parametrize(..., indirect=True))"
        )
    shutil.copytree(source, workspace, dirs_exist_ok=True)


@pytest.fixture
def benchmark_workspace(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[Path]:
    """Yield an isolated, per-test working directory for a benchmark task.

    Args:
        request: pytest request object. ``request.param`` may name a
            subdirectory of ``benchmarks/fixtures/`` whose contents are copied
            into the workspace before it is yielded (optional; the workspace
            is empty when unset).
        tmp_path: pytest's per-test temporary directory, used as the parent so
            pytest owns cleanup.

    Yields:
        An empty (or seeded) :class:`~pathlib.Path` to a temp directory.
    """
    workspace = tmp_path / "benchmark_workspace"
    workspace.mkdir()

    seed_name = getattr(request, "param", None)
    if seed_name:
        _seed_workspace(workspace, seed_name)

    yield workspace
    # ``tmp_path`` (and everything beneath it) is removed by pytest at the end
    # of the session, so explicit cleanup here is unnecessary.


@pytest.fixture(scope="session")
def llamacpp_server() -> Iterator[str]:
    """Confirm llama-server is reachable and yield its base URL.

    Session-scoped: the server is probed once and the result is shared across
    all benchmark tasks in the session. Skips if ``LLAMACPP_HOST`` is unset.
    Polls ``/health`` for up to 60 seconds before declaring the server
    unreachable.

    Benchmark tasks that require inference should depend on this fixture via
    ``pytest.mark.benchmark`` so they skip gracefully when the server is
    absent (issue #355).
    """
    host = os.environ.get("LLAMACPP_HOST")
    if not host:
        pytest.skip("LLAMACPP_HOST is not set; skipping inference-requiring benchmark")

    base_url = host.rstrip("/")
    health_url = f"{base_url}/health"

    with httpx.Client(timeout=5.0) as client:
        for attempt in range(_LLAMACPP_HEALTH_TIMEOUT):
            try:
                response = client.get(health_url)
                if response.status_code == 200 and '"ok"' in response.text:
                    break
            except httpx.RequestError:
                pass
            time.sleep(1)
        else:
            pytest.skip(
                f"llama-server not reachable at {base_url} after "
                f"{_LLAMACPP_HEALTH_TIMEOUT}s; skipping inference-requiring benchmark"
            )

    yield base_url
