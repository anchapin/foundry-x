from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

_NOTES_TAIL_CHARS = 4000


class CriticVerdict(BaseModel):
    """Result of a Critic gate run against a proposed harness edit (ADR-0006)."""

    approved: bool
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    notes: str = ""


class Critic:
    """Gatekeeper that evaluates proposed harness edits in a sandbox.

    Per ADR-0004 every harness edit must pass through this gate before it is
    marked active. The gate applies the ``proposed_diff`` against a *copy* of
    the harness inside a temporary directory and runs pytest there — the live
    ``harness_dir`` is never mutated.

    Benchmark-subset selection (ADR-0004 step 2) is deferred until the
    ``benchmarks/`` subsystem wires ``@pytest.mark.benchmark`` (ADR-0005).
    """

    def __init__(
        self,
        harness_dir: Path,
        benchmark_path: Path | None = None,
        pytest_args: list[str] | None = None,
    ) -> None:
        self.harness_dir = harness_dir
        self.benchmark_path = benchmark_path
        self.pytest_args = pytest_args or ["-q", "tests/test_smoke.py"]

    def evaluate(self, proposed_diff: str) -> CriticVerdict:
        """Apply ``proposed_diff`` to a sandbox copy of the harness and run pytest.

        Steps (ADR-0004):

        1. Copy ``harness_dir`` into a fresh ``TemporaryDirectory``.
        2. Apply ``proposed_diff`` via ``git apply``. A patch that does not
           apply cleanly is rejected immediately (``failed_checks=["git apply"]``).
        3. Run pytest with ``self.pytest_args`` in the sandbox.

        The verdict's ``approved`` flag is ``True`` only when every check that
        runs succeeds. All filesystem mutations are confined to the temp copy.
        """
        with tempfile.TemporaryDirectory(prefix="critic-sandbox-") as sandbox:
            sandbox_root = Path(sandbox) / "harness"
            shutil.copytree(self.harness_dir, sandbox_root)

            passed_checks: list[str] = []
            failed_checks: list[str] = []

            # 1. Apply the proposed diff to the sandbox copy only.
            if proposed_diff.strip():
                apply_result = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn"],
                    input=proposed_diff,
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                )
                if apply_result.returncode != 0:
                    return CriticVerdict(
                        approved=False,
                        passed_checks=[],
                        failed_checks=["git apply"],
                        notes=_tail(apply_result.stderr or apply_result.stdout),
                    )
                passed_checks.append("git apply")

            # 2. Run pytest in the sandbox.
            pytest_result = subprocess.run(
                [sys.executable, "-m", "pytest", *self.pytest_args],
                cwd=sandbox_root,
                capture_output=True,
                text=True,
            )
            if pytest_result.returncode == 0:
                passed_checks.append("pytest")
            else:
                failed_checks.append("pytest")

            combined = (pytest_result.stdout or "") + (pytest_result.stderr or "")
            return CriticVerdict(
                approved=not failed_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                notes=_tail(combined),
            )


def _tail(text: str) -> str:
    """Return the trailing window of *text* for inclusion in verdict notes."""
    return text.strip()[-_NOTES_TAIL_CHARS:]
