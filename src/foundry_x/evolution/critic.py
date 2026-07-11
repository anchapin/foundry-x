from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.models import BenchmarkTask
from benchmarks.registry import load_all_tasks

_NOTES_TAIL_CHARS = 4000

# Wall-clock cap (seconds) for the Critic gate. Mirrors the Runner's
# ``RunLimits.task_timeout_s`` pattern (runner.py:DEFAULT_TASK_TIMEOUT_S) so
# a malformed benchmark that hangs in collection, or a proposed harness edit
# that re-introduces an infinite loop in a hook, cannot make the gate
# itself a runaway. See docs/SECURITY.md "Runaway detection" and ADR-0004
# §Consequences ("Evolution runs are slower but bounded").
DEFAULT_GATE_TIMEOUT_S: int = 300


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
        gate_timeout_s: int = DEFAULT_GATE_TIMEOUT_S,
    ) -> None:
        self.harness_dir = harness_dir
        self.benchmark_path = benchmark_path
        # Next-step wiring target (issue #107): prepend `python
        # harness/scripts/load_check.py --harness-dir <copy>` so a broken
        # `harness/skills/*.json` or an unimportable hook fails the gate
        # *before* pytest runs. Out of scope for #107.
        # Default selection runs the full benchmark suite via ``-m benchmark``
        # (ADR-0005, issue #185) — a harness edit that breaks any
        # ``@pytest.mark.benchmark`` task is caught at the gate.
        self.pytest_args = pytest_args or ["-q", "-m", "benchmark"]
        # Wall-clock cap (issue #188) applied to every subprocess inside
        # ``evaluate`` so a hanging benchmark or a proposed harness edit that
        # re-introduces an infinite loop in a hook cannot make the gate
        # itself a runaway (docs/SECURITY.md "Runaway detection").
        self.gate_timeout_s = gate_timeout_s
        # In-process registry wiring (issue #108): the Critic can now
        # enumerate benchmark tasks without spawning pytest. Stored as
        # ``None`` so the registry is loaded lazily on first access --
        # importing ``foundry_x.evolution.critic`` must not eagerly pull
        # in every task module (and the pytest import chain those tasks
        # transitively trigger).
        self._benchmark_tasks: list[BenchmarkTask] | None = (
            list(benchmark_tasks) if benchmark_tasks is not None else None
        )

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
        """Apply ``proposed_diff`` to a sandbox copy of the harness and run pytest.

        Steps (ADR-0004):

        1. Copy ``harness_dir`` into a fresh ``TemporaryDirectory``.
        2. Apply ``proposed_diff`` via ``git apply``. A patch that does not
           apply cleanly is rejected immediately (``failed_checks=["git apply"]``).
        3. Run pytest with ``self.pytest_args`` in the sandbox.

        Every subprocess inside this method is bounded by
        ``self.gate_timeout_s`` (issue #188). On
        :class:`subprocess.TimeoutExpired` the verdict is
        ``approved=False`` with ``failed_checks`` carrying the offending check
        name suffixed ``":timeout"`` (e.g. ``"pytest:timeout"``), and
        ``notes`` holds the trailing window of any partial output the
        process managed to write before being killed — or a wall-clock-cap
        message when no partial output was captured.

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
                try:
                    apply_result = subprocess.run(
                        ["git", "apply", "--whitespace=nowarn"],
                        input=proposed_diff,
                        cwd=sandbox_root,
                        capture_output=True,
                        text=True,
                        timeout=self.gate_timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    return CriticVerdict(
                        approved=False,
                        passed_checks=[],
                        failed_checks=["git apply:timeout"],
                        notes=_timeout_notes(exc, self.gate_timeout_s, "git apply"),
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
            try:
                pytest_result = subprocess.run(
                    [sys.executable, "-m", "pytest", *self.pytest_args],
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                    timeout=self.gate_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                return CriticVerdict(
                    approved=False,
                    passed_checks=passed_checks,
                    failed_checks=["pytest:timeout"],
                    notes=_timeout_notes(exc, self.gate_timeout_s, "pytest"),
                )
            if pytest_result.returncode == 0:
                passed_checks.append("pytest")
                # Record every benchmark tag the run covered (issue #185).
                covered_tags = sorted({tag for task in self.benchmark_tasks for tag in task.tags})
                passed_checks.extend(f"benchmark:{tag}" for tag in covered_tags)
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


def _timeout_notes(exc: subprocess.TimeoutExpired, gate_timeout_s: int, check: str) -> str:
    """Build verdict ``notes`` for a ``subprocess.TimeoutExpired``.

    Prefers the trailing window of any partial stdout/stderr the subprocess
    managed to write before being killed (the same shape callers expect for
    non-timeout failures). Falls back to a wall-clock-cap message naming the
    offending check so the verdict is never empty: the
    ``test_pytest_exceeds_timeout_rejected`` acceptance test (issue #188)
    asserts ``verdict.notes`` is truthy.
    """
    partial = (exc.stdout or b"") + (exc.stderr or b"")
    if isinstance(partial, bytes):
        partial = partial.decode("utf-8", errors="replace")
    if partial.strip():
        return _tail(partial)
    return _tail(f"{check} exceeded {gate_timeout_s}s wall-clock cap")
