"""Static validation of ``.github/workflows/test.yml`` benchmark trigger (issue #1280).

Verifies that the ``benchmark`` job in ``test.yml`` triggers when core
components that can silently break benchmark tasks change:
- ``src/foundry_x/execution/``  (e.g. runner.py)
- ``src/foundry_x/trace/``     (e.g. logger.py)

Previously the benchmark job only ran when ``benchmarks/`` files changed,
which meant core changes that break benchmarks silently passed CI.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "test.yml"

# Paths whose changes should always trigger the benchmark suite (issue #1280).
BENCHMARK_TRIGGER_PATHS = (
    "benchmarks/",
    "src/foundry_x/execution/",
    "src/foundry_x/trace/",
)


@pytest.fixture(scope="module")
def config() -> dict:
    assert TEST_WORKFLOW.exists(), f"test.yml missing at {TEST_WORKFLOW}"
    data = yaml.safe_load(TEST_WORKFLOW.read_text())
    assert isinstance(data, dict), "test.yml must parse to a mapping."
    return data


def _benchmark_job_if_condition(config: dict) -> str | None:
    """Return the `if` expression on the benchmark job, or None."""
    jobs = config.get("jobs", {})
    benchmark = jobs.get("benchmark")
    if benchmark is None:
        return None
    return benchmark.get("if")


def test_benchmark_job_exists(config: dict) -> None:
    """``test.yml`` must have a ``benchmark`` job."""
    assert "jobs" in config, "test.yml must have a jobs section."
    assert "benchmark" in config["jobs"], (
        "test.yml must have a 'benchmark' job to catch regressions in core "
        "components that benchmark tasks depend on (issue #1280)."
    )


def test_benchmark_job_has_if_condition(config: dict) -> None:
    """The ``benchmark`` job must have an ``if`` condition to gate when it runs."""
    cond = _benchmark_job_if_condition(config)
    assert cond is not None, (
        "benchmark job must have an 'if' condition gating when it runs "
        "(issue #1280)."
    )


def test_benchmark_if_checks_all_trigger_paths(config: dict) -> None:
    """The ``if`` condition must cover every path that can silently regress benchmarks."""
    cond = _benchmark_job_if_condition(config)
    assert cond is not None, "benchmark job must have an 'if' condition."
    for path in BENCHMARK_TRIGGER_PATHS:
        # The condition uses contains(..., 'path') substring matching.
        assert path in cond or f"'{path}'" in cond or f'"{path}"' in cond, (
            f"benchmark job 'if' condition must include {path!r} so changes to "
            f"that path trigger the benchmark suite and block silent regressions "
            f"(issue #1280). Current condition: {cond!r}"
        )
