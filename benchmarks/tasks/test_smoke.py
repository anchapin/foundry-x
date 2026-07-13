"""Infrastructure smoke benchmark for the benchmark suite (issue #27).

This is the canary that issue #27 asked for: a single trivial test, marked
``@pytest.mark.benchmark``, that asserts a static condition and requires no
agent execution. Its only job is to prove the end-to-end plumbing works --
that pytest discovers marked files under ``benchmarks/``, that the
``benchmark`` marker is registered (no ``PytestUnknownMarkWarning``), and
that the ``benchmark_workspace`` fixture resolves. If this test cannot be
collected or selected with ``-m benchmark``, the bootstrap is broken.

This canary now lives under ``benchmarks/tasks/`` (issue #108) so the
in-process registry discovers it via its ``TASK`` attribute alongside
every other benchmark. The smoke task's role as a cheap static canary is
unchanged -- it still requires no fixture data and no solution logic --
only its physical placement was moved to align with the documented
"every benchmark task lives under benchmarks/tasks/" layout
(benchmarks/README.md).

Acceptance (issue #27):

    uv run pytest -m benchmark          # collects and passes the smoke task
    uv run pytest                       # collects both tests/ and benchmarks/
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask

#: Smoke-tier BenchmarkTask so the in-process registry (issue #108) can
#: enumerate this canary through the same ``TASK = BenchmarkTask(...)``
#: attribute every other benchmark uses. ``difficulty_tier="smoke"`` matches
#: the documented tier ladder (benchmarks/models.py) and the cost-shape of
#: this test (no agent invocation, no fixture data).
TASK = BenchmarkTask(
    name="smoke_marker_and_fixture_resolve",
    description=(
        "Infrastructure canary: the benchmark marker is registered and the "
        "benchmark_workspace fixture yields an empty temp directory. "
        "Proves the end-to-end plumbing without invoking any agent."
    ),
    difficulty_tier="smoke",
    expected_outcome=(
        "pytest discovers the test, recognises the benchmark marker, and "
        "the benchmark_workspace fixture yields an empty directory."
    ),
    tags=["smoke", "infrastructure"],
)


@pytest.mark.benchmark
def test_smoke_marker_and_fixture_resolve(benchmark_workspace: Path) -> None:
    """Static canary: the suite is discoverable, the marker is known, and the
    workspace fixture yields a usable empty directory.

    No agent runs here. The assertion is a static truth that can only be
    evaluated once pytest has collected this file, recognised the
    ``benchmark`` marker, and materialised the ``benchmark_workspace``
    fixture -- so a pass mechanically satisfies ADR-0005's "Decision".
    """
    assert benchmark_workspace.is_dir()
    assert list(benchmark_workspace.iterdir()) == []
