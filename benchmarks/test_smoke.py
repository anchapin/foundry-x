"""Infrastructure smoke benchmark for the benchmark suite.

This is the canary that issue #27 asked for: a single trivial test, marked
``@pytest.mark.benchmark``, that asserts a static condition and requires no
agent execution. Its only job is to prove the end-to-end plumbing works —
that pytest discovers marked files under ``benchmarks/``, that the
``benchmark`` marker is registered (no ``PytestUnknownMarkWarning``), and
that the ``benchmark_workspace`` fixture resolves. If this test cannot be
collected or selected with ``-m benchmark``, the bootstrap is broken.

The real coding tasks live under ``benchmarks/tasks/`` (issue #30); this
file is intentionally separate so the canary stays independent of fixture
data and solution logic.

Acceptance (issue #27):

    uv run pytest -m benchmark          # collects and passes the smoke task
    uv run pytest                       # collects both tests/ and benchmarks/
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.benchmark
def test_smoke_marker_and_fixture_resolve(benchmark_workspace: Path) -> None:
    """Static canary: the suite is discoverable, the marker is known, and the
    workspace fixture yields a usable empty directory.

    No agent runs here. The assertion is a static truth that can only be
    evaluated once pytest has collected this file, recognised the
    ``benchmark`` marker, and materialised the ``benchmark_workspace``
    fixture — so a pass mechanically satisfies ADR-0005's "Decision".
    """
    assert benchmark_workspace.is_dir()
    assert list(benchmark_workspace.iterdir()) == []
