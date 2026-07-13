from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.models import BenchmarkTask
from benchmarks.registry import load_all_tasks

from foundry_x.evolution.sandbox import (
    DockerSandbox,
    SandboxConfig,
    SandboxRuntimeError,
)

_NOTES_TAIL_CHARS = 4000

_INJECTION_PATTERNS: tuple[str, ...] = (
    "ignore previous instructions",
    "<<system>>",
    "<<assistant>>",
    "<|",
    "end of context above",
)


def _contains_injection(text: str) -> bool:
    """Return True if *text* contains injection-like patterns.

    Scans for three categories flagged in docs/SECURITY.md:46-50:
    instruction override, role-tag injection, and ignored-context override.
    """
    lower = text.lower()
    return any(pattern.lower() in lower for pattern in _INJECTION_PATTERNS)


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
    the harness inside a temporary directory and runs pytest inside a named
    Docker container that is spawned and torn down per evaluation — the live
    ``harness_dir`` is never mutated.

    Benchmark-subset selection (ADR-0004 step 2) uses ``-m benchmark`` by
    default so every ``@pytest.mark.benchmark`` task gates the edit
    (ADR-0005, issue #185). The verdict's ``passed_checks`` lists every
    benchmark tag the run covered.
    """

    def __init__(
        self,
        harness_dir: Path,
        benchmark_path: Path | None = None,
        pytest_args: list[str] | None = None,
        benchmark_tasks: list[BenchmarkTask] | None = None,
        max_diff_lines: int = 200,
        sandbox_config: SandboxConfig | None = None,
        use_sandbox: bool = True,
    ) -> None:
        self.harness_dir = harness_dir
        self.benchmark_path = benchmark_path
        self.pytest_args = pytest_args or ["-q", "-m", "benchmark"]
        self._benchmark_tasks: list[BenchmarkTask] | None = (
            list(benchmark_tasks) if benchmark_tasks is not None else None
        )
        if max_diff_lines < 1:
            raise ValueError("max_diff_lines must be >= 1")
        self.max_diff_lines = max_diff_lines
        self._sandbox_config = sandbox_config or SandboxConfig()
        self._use_sandbox = use_sandbox

    @property
    def benchmark_tasks(self) -> list[BenchmarkTask]:
        """The ``BenchmarkTask`` instances this Critic will gate against (issue #108).

        Lazy-loaded from the in-process registry on first access; cached on
        the instance so subsequent accesses are O(1). Tests can pre-seed
        ``benchmark_tasks=...`` in the constructor to avoid touching the
        registry at all (see ``tests/test_critic.py``).
        """
        if self._benchmark_tasks is None:
            self._benchmark_tasks = load_all_tasks()
        return self._benchmark_tasks

    def evaluate(self, proposed_diff: str) -> CriticVerdict:
        """Apply ``proposed_diff`` to a sandbox harness copy and gate it.

        Steps (ADR-0004):

        1. Copy ``harness_dir`` into a fresh ``TemporaryDirectory``.
        2. Reject diffs containing injection-like text (``content_rejected``).
        3. Reject diffs exceeding ``max_diff_lines`` (``diff_too_large``).
        4. Apply ``proposed_diff`` via ``git apply``. A patch that does not
           apply cleanly is rejected immediately (``failed_checks=["git apply"]``).
        5. Run ``harness/scripts/load_check.py`` against the sandbox copy
           (issue #187). A harness that fails to load -- broken
           ``skills/*.json``, an unimportable hook, an empty system prompt
           -- is rejected *before* pytest is spawned, so the verdict names
           the precondition (``failed_checks=["load_check"]``) rather than
           a confusing downstream pytest error.
        6. Run pytest with ``self.pytest_args`` in the sandbox.

        The verdict's ``approved`` flag is ``True`` only when every check that
        runs succeeds. All filesystem mutations are confined to the temp copy.
        The container is torn down when the evaluation finishes.
        """
        with tempfile.TemporaryDirectory(prefix="critic-sandbox-") as tmp_dir:
            tmp_path = Path(tmp_dir)
            sandbox_root = tmp_path / "harness"
            self._copy_harness(self.harness_dir, sandbox_root)

            passed_checks: list[str] = []
            failed_checks: list[str] = []

            # 1. Reject diffs containing injection-like text (SECURITY.md §Prompt injection).
            if _contains_injection(proposed_diff):
                return CriticVerdict(
                    approved=False,
                    passed_checks=[],
                    failed_checks=["content_rejected"],
                    notes="diff contains injection-like text (SECURITY.md §Prompt injection)",
                )

            # 2. Reject diffs exceeding the line cap (SECURITY.md §Rate limits).
            diff_line_count = len(proposed_diff.splitlines())
            if diff_line_count > self.max_diff_lines:
                return CriticVerdict(
                    approved=False,
                    passed_checks=[],
                    failed_checks=["diff_too_large"],
                    notes=f"diff has {diff_line_count} lines, cap is {self.max_diff_lines}",
                )

            # 3. Apply the proposed diff to the sandbox harness copy on the
            # host.  The diff is applied before the container starts so a
            # broken patch is detected without paying the container-start cost.
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

            # 4. Run load_check and pytest.
            # When _use_sandbox is True (production), both commands run inside
            # a Docker container.  When False (test environments without Docker),
            # they run directly on the host against the harness copy.
            if self._use_sandbox:
                passed, failed, combined = self._evaluate_in_docker(
                    sandbox_root, passed_checks, failed_checks
                )
            else:
                passed, failed, combined = self._evaluate_direct(
                    sandbox_root, passed_checks, failed_checks
                )

            if combined is not None:
                return CriticVerdict(
                    approved=not failed,
                    passed_checks=passed,
                    failed_checks=failed,
                    notes=_tail(combined),
                )
            else:
                return CriticVerdict(
                    approved=False,
                    passed_checks=passed,
                    failed_checks=failed,
                    notes="",
                )

    def _evaluate_in_docker(
        self,
        sandbox_root: Path,
        passed_checks: list[str],
        failed_checks: list[str],
    ) -> tuple[list[str], list[str], str | None]:
        """Run load_check and pytest inside a Docker container (issue #353)."""
        sandbox = DockerSandbox(sandbox_root, self._sandbox_config)
        try:
            with sandbox:
                load_check_result = sandbox.run(
                    [
                        sys.executable,
                        "/app/harness/scripts/load_check.py",
                        "--harness-dir",
                        "/app/harness",
                    ],
                    cwd="/app/harness",
                )
        except SandboxRuntimeError as exc:
            return (
                passed_checks,
                [*failed_checks, "sandbox"],
                str(exc),
            )

        if load_check_result.returncode != 0:
            return (
                passed_checks,
                [*failed_checks, "load_check"],
                load_check_result.stderr or load_check_result.stdout,
            )
        passed_checks.append("load_check")

        pytest_result = sandbox.run(
            [sys.executable, "-m", "pytest", *self.pytest_args],
            cwd="/app/harness",
        )
        return self._process_pytest_result(
            sandbox_root, passed_checks, failed_checks, pytest_result
        )

    def _evaluate_direct(
        self,
        sandbox_root: Path,
        passed_checks: list[str],
        failed_checks: list[str],
    ) -> tuple[list[str], list[str], str | None]:
        """Run load_check and pytest directly on the host (test fallback)."""
        load_check_script = sandbox_root / "scripts" / "load_check.py"
        load_result = subprocess.run(
            [sys.executable, str(load_check_script), "--harness-dir", str(sandbox_root)],
            cwd=sandbox_root,
            capture_output=True,
            text=True,
        )
        if load_result.returncode != 0:
            return (
                passed_checks,
                [*failed_checks, "load_check"],
                load_result.stderr or load_result.stdout,
            )
        passed_checks.append("load_check")

        pytest_result = subprocess.run(
            [sys.executable, "-m", "pytest", *self.pytest_args],
            cwd=sandbox_root,
            capture_output=True,
            text=True,
        )
        return self._process_pytest_result(
            sandbox_root, passed_checks, failed_checks, pytest_result
        )

    def _process_pytest_result(
        self,
        sandbox_root: Path,
        passed_checks: list[str],
        failed_checks: list[str],
        result: subprocess.CompletedProcess[str],
    ) -> tuple[list[str], list[str], str | None]:
        combined = (result.stdout or "") + (result.stderr or "")
        if result.returncode == 0:
            passed_checks.append("pytest")
            covered_tags = sorted({tag for task in self.benchmark_tasks for tag in task.tags})
            passed_checks.extend(f"benchmark:{tag}" for tag in covered_tags)
        else:
            failed_checks.append("pytest")
        return passed_checks, failed_checks, combined

    @staticmethod
    def _copy_harness(src: Path, dst: Path) -> None:
        """Copy ``src`` harness tree to ``dst``.

        ``shutil.copytree`` is used so the live ``harness_dir`` is never
        mutated during evaluation.  Each file is copied individually so the
        implementation matches the ``_hash_dir`` check in the test suite
        (``tests/test_critic.py``).
        """
        import shutil

        shutil.copytree(src, dst)


def _tail(text: str) -> str:
    """Return the trailing window of *text* for inclusion in verdict notes."""
    return text.strip()[-_NOTES_TAIL_CHARS:]
