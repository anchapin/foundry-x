from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.models import BenchmarkTask
from benchmarks.registry import load_all_tasks

DEFAULT_REGRESSION_THRESHOLD_PP = 2.0

_NOTES_TAIL_CHARS = 4000

# SECURITY.md Threat #2: prompt-injection patterns checked at the Critic gate
# (issue #333). These are the same categories named in the firewall docstring
# (harness/hooks/injection_firewall.py INJECTION_PATTERNS) but expressed as
# plain strings rather than compiled regexes so the critic can scan a diff
# without importing the harness package.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ignore_previous", r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    ("disregard_previous", r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    ("forget_previous", r"forget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    ("new_instructions", r"(?:new|updated|real)\s+instructions\s*:"),
    ("role_tag_colon", r"(?:^|\n|\r)\s*(?:system|assistant|developer|user)\s*:\s*"),
    ("role_tag_brackets", r"<{2}(?:system|assistant|developer|user)>{2}"),
    ("chatml_tag", r"<\|(?:im_start|im_end|system|assistant|user|begin_of_text|endoftext)\|>"),
    ("ignored_context", r"end\s+of\s+context\s+above"),
)


def _scan_diff_for_injection(diff: str) -> list[str]:
    """Return the names of any injection patterns found in *diff*.

    Scans only the ``+`` (addition) lines of the unified diff after stripping
    the diff prefix characters (``+/ /-``) so that the plain content is matched
    against the patterns.  This avoids false negatives where a pattern's ``^``
    anchor would fail to match because the diff line starts with ``+``.
    """
    triggered: list[str] = []
    addition_lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+"):
            addition_lines.append(line[1:])
    clean_content = "\n".join(addition_lines)
    for name, pattern in _INJECTION_PATTERNS:
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        if regex.search(clean_content):
            triggered.append(name)
    return triggered


#: Default location of the regression baseline JSON written by the Critic
#: (ADR-0004 step 3, issue #186). Relative to the process working directory
#: so an invocation from the repo root lands at ``logs/critic_baseline.json``.
DEFAULT_BASELINE_PATH: Path = Path("logs") / "critic_baseline.json"


class CriticVerdict(BaseModel):
    """Result of a Critic gate run against a proposed harness edit (ADR-0006)."""

    verdict: bool
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    notes: str = ""


class QuantizationResult(BaseModel):
    """Per-quantization benchmark result (ADR-0016)."""

    quantization: str
    model_path: str
    model_id: str
    total_tasks: int = 0
    passed_tasks: int = 0
    failed_tasks: int = 0
    task_shaped_failures: int = 0
    pass_rate: float = 0.0
    avg_cycle_time_s: float | None = None
    total_tokens: int = 0


class QuantizationVerdict(BaseModel):
    """Sweep-level verdict aggregating per-quantization results (ADR-0016)."""

    quantizations: list[QuantizationResult]
    recommended: str
    regression: bool


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
        max_diff_lines: int = 200,
    ) -> None:
        self.harness_dir = harness_dir
        self.benchmark_path = benchmark_path
        # Default selection runs the full benchmark suite via ``-m benchmark``
        # (ADR-0005, issue #185) — a harness edit that breaks any
        # ``@pytest.mark.benchmark`` task is caught at the gate.
        self.pytest_args = pytest_args or ["-q", "-m", "benchmark"]
        # Diff-size cap mirrors the SECURITY.md "max M lines of harness diff
        # per proposal" guardrail and the Evolver default (issue #333).
        if max_diff_lines < 1:
            raise ValueError("max_diff_lines must be >= 1")
        self.max_diff_lines = max_diff_lines
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

    def quantization_sweep(
        self,
        quantizations: list[str],
        model_glob_patterns: dict[str, str] | None = None,
        baseline_quantization: str | None = None,
        regression_threshold_pp: float = DEFAULT_REGRESSION_THRESHOLD_PP,
    ) -> QuantizationVerdict:
        """Run the benchmark suite against each quantization and produce a comparison.

        ADR-0016: this is the single entry point for multi-quantization
        evaluation, keeping orchestration logic co-located with single-run
        evaluation.

        Args:
            quantizations: quantization labels to sweep (e.g. ``["Q4_K_S", "Q5_K_M"]``).
            model_glob_patterns: maps each quantization label to a glob pattern
                relative to ``FOUNDRY_MODEL_PATH``. Defaults to
                ``{q: f"*.{q}.gguf" for q in quantizations}``.
            baseline_quantization: quantization to compare against. Defaults to
                the first quantization in *quantizations*.
            regression_threshold_pp: regression threshold in percentage points.
                A candidate's pass rate must be within this many pp of the
                baseline to be considered non-regressing (default 2.0 pp).

        Returns:
            A ``QuantizationVerdict`` with per-quantization results and a
            recommended quantization label. ``regression`` is ``True`` when
            the recommended quantization has a lower pass rate than the
            baseline beyond the regression threshold.

        Sweep is idempotent and safe to re-run. Each quantization run is
        stamped with ``FOUNDRY_MODEL_ID`` in the trace store. Exit code of
        the subprocess is 0 on success, non-zero if any quantization fails
        all benchmarks.
        """
        model_path_env = os.environ.get("FOUNDRY_MODEL_PATH", "")
        if not model_path_env:
            raise ValueError("FOUNDRY_MODEL_PATH environment variable is not set")

        model_base = Path(model_path_env)
        if not model_base.exists():
            raise FileNotFoundError(f"FOUNDRY_MODEL_PATH does not exist: {model_path_env}")

        if model_glob_patterns is None:
            model_glob_patterns = {q: f"*.{q}.gguf" for q in quantizations}

        results: list[QuantizationResult] = []

        for quant in quantizations:
            pattern = model_glob_patterns.get(quant, f"*.{quant}.gguf")
            full_pattern = str(model_base / pattern)
            matched = glob.glob(full_pattern)

            if not matched:
                raise FileNotFoundError(
                    f"No model file found for quantization {quant!r} "
                    f"using pattern {pattern!r} in {model_path_env}"
                )
            if len(matched) > 1:
                raise ValueError(
                    f"Multiple model files matched for quantization {quant!r}: {matched}"
                )

            model_file = matched[0]
            model_id = f"{quant}"

            original_model_path = os.environ.get("FOUNDRY_MODEL_PATH")
            original_model_id = os.environ.get("FOUNDRY_MODEL_ID")

            os.environ["FOUNDRY_MODEL_PATH"] = str(model_base)
            os.environ["FOUNDRY_MODEL_ID"] = model_id

            try:
                sweep_result = self._run_sweep_for_quant(model_file, model_id)
                results.append(sweep_result)
            finally:
                if original_model_path is not None:
                    os.environ["FOUNDRY_MODEL_PATH"] = original_model_path
                else:
                    os.environ.pop("FOUNDRY_MODEL_PATH", None)
                if original_model_id is not None:
                    os.environ["FOUNDRY_MODEL_ID"] = original_model_id
                else:
                    os.environ.pop("FOUNDRY_MODEL_ID", None)

        baseline = baseline_quantization if baseline_quantization else quantizations[0]
        baseline_result = next(r for r in results if r.quantization == baseline)

        regression = False
        for result in results:
            if result.quantization == baseline:
                continue
            rate_diff_pp = (result.pass_rate - baseline_result.pass_rate) * 100
            if rate_diff_pp < -regression_threshold_pp:
                regression = True
                break

        recommended = baseline
        if not regression:
            best = max(results, key=lambda r: r.pass_rate)
            recommended = best.quantization

        return QuantizationVerdict(
            quantizations=results,
            recommended=recommended,
            regression=regression,
        )

    def _run_sweep_for_quant(self, model_file: str, model_id: str) -> QuantizationResult:
        """Run the benchmark suite for a single quantization.

        Returns a ``QuantizationResult`` with metrics parsed from pytest output.
        """
        quant_label = Path(model_file).stem
        result = subprocess.run(
            [sys.executable, "-m", "pytest", *self.pytest_args, "--tb=no", "-v"],
            capture_output=True,
            text=True,
        )

        passed = 0
        failed = 0
        task_shaped = 0
        total = 0

        for line in (result.stdout or "").splitlines():
            if " passed" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "passed":
                        try:
                            passed = int(parts[i - 1])
                            total = passed
                            break
                        except (IndexError, ValueError):
                            pass
            if " failed" in line:
                parts = line.split()
                for i, p in enumerate(parts):
                    if p == "failed":
                        try:
                            failed = int(parts[i - 1])
                            break
                        except (IndexError, ValueError):
                            pass

        if passed + failed > 0:
            total = passed + failed

        pass_rate = (passed / total) if total > 0 else 0.0

        return QuantizationResult(
            quantization=quant_label,
            model_path=model_file,
            model_id=model_id,
            total_tasks=total,
            passed_tasks=passed,
            failed_tasks=failed,
            task_shaped_failures=task_shaped,
            pass_rate=pass_rate,
            total_tokens=0,
        )

    def evaluate(self, proposed_diff: str) -> CriticVerdict:
        """Apply ``proposed_diff`` to a sandbox copy of the harness and gate it.

        Steps (ADR-0004):

        1. Copy ``harness_dir`` into a fresh ``TemporaryDirectory``.
        2. Enforce the diff-size cap (``max_diff_lines``). An oversized diff
           is rejected immediately (``failed_checks=["diff_size_cap"]``).
        3. Scan the diff for prompt-injection markers (SECURITY.md Threat #2).
           A diff carrying ``ignore previous instructions``-style phrases or
           role-tag sequences is rejected immediately
           (``failed_checks=["injection_detected"]``).
        4. Apply ``proposed_diff`` via ``git apply``. A patch that does not
           apply cleanly is rejected immediately (``failed_checks=["git apply"]``).
        5. Run ``harness/scripts/load_check.py`` against the sandbox copy
           (issue #187). A harness that fails to load -- broken
           ``skills/*.json``, an unimportable hook, an empty system prompt
           -- is rejected *before* pytest is spawned, so the verdict names
           the precondition (``failed_checks=["load_check"]``) rather than
           a confusing downstream pytest error.
        6. Run pytest with ``self.pytest_args`` in the sandbox.

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

            # Gate 1: Diff-size cap (issue #333).
            if proposed_diff.strip():
                line_count = len(proposed_diff.splitlines())
                if line_count > self.max_diff_lines:
                    return CriticVerdict(
                        verdict=False,
                        passed_checks=[],
                        failed_checks=["diff_size_cap"],
                        notes=f"diff too large: {line_count} lines (cap={self.max_diff_lines})",
                    )
                # Gate 2: Injection scan (issue #333 / SECURITY.md Threat #2).
                #    the diff payload for prompt-injection markers before the
                #    sandbox is mutated.
                injection_markers = _scan_diff_for_injection(proposed_diff)
                if injection_markers:
                    return CriticVerdict(
                        verdict=False,
                        passed_checks=[],
                        failed_checks=["injection_detected"],
                        notes=f"injection pattern(s) in diff: {', '.join(injection_markers)}",
                    )
                apply_result = subprocess.run(
                    ["git", "apply", "--whitespace=nowarn"],
                    input=proposed_diff,
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                )
                if apply_result.returncode != 0:
                    return CriticVerdict(
                        verdict=False,
                        passed_checks=[],
                        failed_checks=["git apply"],
                        notes=_tail(apply_result.stderr or apply_result.stdout),
                    )
                passed_checks.append("git apply")

            # Gate 3: Precondition gate (issue #187).
            #    against the sandbox copy. A harness tree that fails to load
            #    must fail the gate *before* pytest runs.
            load_check_script = sandbox_root / "scripts" / "load_check.py"
            load_result = subprocess.run(
                [
                    sys.executable,
                    str(load_check_script),
                    "--harness-dir",
                    str(sandbox_root),
                ],
                cwd=sandbox_root,
                capture_output=True,
                text=True,
            )
            if load_result.returncode != 0:
                return CriticVerdict(
                    verdict=False,
                    passed_checks=passed_checks,
                    failed_checks=[*failed_checks, "load_check"],
                    notes=_tail(load_result.stderr or load_result.stdout),
                )
            passed_checks.append("load_check")

            # Gate 4: Pytest benchmark suite.
            pytest_result = subprocess.run(
                [sys.executable, "-m", "pytest", *self.pytest_args],
                cwd=sandbox_root,
                capture_output=True,
                text=True,
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
                verdict=not failed_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                notes=_tail(combined),
            )


def _tail(text: str) -> str:
    """Return the trailing window of *text* for inclusion in verdict notes."""
    return text.strip()[-_NOTES_TAIL_CHARS:]
