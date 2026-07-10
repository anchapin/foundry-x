"""Shared execution helpers for benchmark tasks.

These helpers stand in for the agent runner until ``Runner.run_task`` is
wired (currently ``NotImplementedError``). They let each task validate
its input/output contract against a golden solution and lock the green
regression baseline now, so the Critic gate (ADR-0004) is non-vacuous.

When the Runner lands, a task swaps ``run_solution`` for the real agent
invocation; the workspace staging and the final assertion stay unchanged.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def run_solution(workspace: Path, source: str) -> subprocess.CompletedProcess[str]:
    """Write ``source`` to ``solution.py`` and run it inside ``workspace``."""
    (workspace / "solution.py").write_text(source)
    return subprocess.run(
        [sys.executable, "solution.py"],
        cwd=workspace,
        check=True,
        capture_output=True,
        text=True,
    )


def run_module(workspace: Path, module_name: str) -> subprocess.CompletedProcess[str]:
    """Run a module (e.g. ``pytest``) inside ``workspace``."""
    return subprocess.run(
        [sys.executable, "-m", module_name, "-q", "-p", "no:cacheprovider"],
        cwd=workspace,
        capture_output=True,
        text=True,
    )
