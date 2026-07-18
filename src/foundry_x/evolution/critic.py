from __future__ import annotations

import base64
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from pydantic import BaseModel, Field

from benchmarks.models import BenchmarkTask
from benchmarks.registry import load_all_tasks
from foundry_x.trace.logger import TraceLogger

DEFAULT_REGRESSION_THRESHOLD_PP = 2.0

_NOTES_TAIL_CHARS = 4000

# SECURITY.md Threat #2: prompt-injection patterns checked at the Critic gate
# (issue #333). These are the same categories named in the firewall docstring
# (harness/hooks/injection_firewall.py INJECTION_PATTERNS) but expressed as
# plain strings rather than compiled regexes so the critic can scan a diff
# without importing the harness package.
#
# CRITICAL: This tuple MUST stay in sync with INJECTION_PATTERNS in
# harness/hooks/injection_firewall.py. When adding a pattern to the firewall,
# add it here as well. See ADR-0009 and issue #646.
_INJECTION_PATTERNS: tuple[tuple[str, str], ...] = (
    ("ignore_previous", r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    ("disregard_previous", r"disregard\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    ("forget_previous", r"forget\s+(?:all\s+)?(?:previous|prior|above)\s+instructions"),
    ("new_instructions", r"(?:new|updated|real)\s+instructions\s*:"),
    ("role_tag_colon", r"(?:^|\n|\r)\s*(?:system|assistant|developer|user)\s*:\s*"),
    ("role_tag_brackets", r"<{2}(?:system|assistant|developer|user)>{2}"),
    ("chatml_tag", r"<\|(?:im_start|im_end|system|assistant|user|begin_of_text|endoftext)\|>"),
    ("ignored_context", r"end\s+of\s+context\s+above"),
    # --- Issue #122 / issue #579: sync with injection_firewall.py patterns ---
    ("ignore_spanish", r"ignora\s+(?:las\s+)?instrucciones\s+anteriores"),
    # Issue #755: sync with injection_firewall.py non-English evasion patterns
    (
        "ignore_french",
        r"(?:ignorer\s+(?:les\s+)?instructions|oublier\s+(?:les\s+)?consignes)",
    ),
    ("ignore_german", r"ignoriere\s+(?:vorherige\s+)?(?:die\s+)?Anweisungen"),
    ("ignore_portuguese", r"ignore\s+(?:as\s+)?instruções\s+anteriores"),
    ("ignore_italian", r"ignora\s+(?:le\s+)?istruzioni\s+precedenti"),
    (
        "role_tag_json_escaped",
        r'\\"role\\":\\"(?:system|assistant|developer|user)',
    ),
    ("unicode_confusable", r"[\u200B-\u200F\u2028-\u202F\u2060-\u2064\uFEFF]"),
    ("base64_payload", r"[A-Za-z0-9+/]{16,}={0,2}"),
)


_BASE64_MAX_LEN = 4096

_ZERO_WIDTH_RE = re.compile(r"[\u200B-\u200F\u2028-\u202F\u2060-\u2064\uFEFF]")


def _scan_diff_for_injection(diff: str) -> list[str]:
    """Return the names of any injection patterns found in *diff*.

    Scans only the ``+`` (addition) lines of the unified diff after stripping
    the diff prefix characters (``+/ /-``) so that the plain content is matched
    against the patterns.  This avoids false negatives where a pattern's ``^``
    anchor would fail to match because the diff line starts with ``+``.

    The pipeline mirrors the firewall's ``scan_for_injection``:
    1. Strip zero-width / format characters from a copy of the content. The
       original is kept for ``unicode_confusable`` detection.
    2. Run all patterns against the cleaned text (except ``base64_payload``,
       which is handled via decode + rescan).
    3. Run ``unicode_confusable`` against the raw text.
    4. Decode every base64 candidate and rescan the decoded content for ASCII
       markers; emit ``base64_payload`` only when decoded content matches.
    """
    addition_lines: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+"):
            addition_lines.append(line[1:])
    raw_content = "\n".join(addition_lines)
    cleaned_content = _ZERO_WIDTH_RE.sub("", raw_content)

    triggered: list[str] = []
    base64_payload_pat = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}", re.IGNORECASE)

    for name, pattern in _INJECTION_PATTERNS:
        if name == "base64_payload":
            continue
        target = raw_content if name == "unicode_confusable" else cleaned_content
        regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
        if regex.search(target):
            triggered.append(name)

    for m in base64_payload_pat.finditer(cleaned_content):
        candidate = m.group(0)
        if len(candidate) > _BASE64_MAX_LEN:
            continue
        try:
            decoded_bytes = base64.b64decode(candidate, validate=True)
        except Exception:
            continue
        try:
            decoded = decoded_bytes.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for name, pattern in _INJECTION_PATTERNS:
            if name in ("base64_payload", "unicode_confusable"):
                continue
            regex = re.compile(pattern, re.IGNORECASE | re.MULTILINE)
            if regex.search(decoded):
                triggered.append("base64_payload")
                break

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
    edit_index: int | None = None
    failure_class: str | None = None


class TaskResult(BaseModel):
    """Per-task pass/fail result within a quantization sweep (issue #495)."""

    name: str
    passed: bool


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
    token_efficiency: float | None = Field(
        default=None,
        description=(
            "Tokens processed per second of cycle time (total_tokens / avg_cycle_time_s). "
            "Requires both total_tokens > 0 and avg_cycle_time_s > 0 to compute."
        ),
    )
    cost_per_task: float | None = Field(
        default=None,
        description=(
            "Estimated cost per completed task in dollars. Computed as "
            "(total_tokens * cost_per_token) / passed_tasks when both are available. "
            "Requires FOUNDRY_COST_PER_TOKEN or cost_per_token argument to be set."
        ),
    )
    task_results: list[TaskResult] = Field(
        default_factory=list,
        description=(
            "Per-task pass/fail breakdown for this quantization. Enables "
            "comparable success rates across quantizations (issue #495)."
        ),
    )


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
        gate_timeout_s: float | None = None,
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
        # Wall-clock cap applied to every subprocess spawned inside
        # ``evaluate()`` (issue #890, ADR-0004). ``None`` preserves the
        # historical unbounded behaviour; a positive float bounds git apply,
        # load_check, and pytest so a hanging child cannot inflate
        # ``kpi-cycle-time`` to infinity.
        if gate_timeout_s is not None and gate_timeout_s <= 0:
            raise ValueError("gate_timeout_s must be > 0 or None")
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

    def quantization_sweep(
        self,
        quantizations: list[str],
        model_glob_patterns: dict[str, str] | None = None,
        baseline_quantization: str | None = None,
        regression_threshold_pp: float = DEFAULT_REGRESSION_THRESHOLD_PP,
        cost_per_token: float | None = None,
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
            cost_per_token: cost per token in USD for cost-per-task computation.
                When provided, ``cost_per_task`` is computed for each quantization
                result. Can also be set via ``FOUNDRY_COST_PER_TOKEN`` env var.

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

        effective_cost = cost_per_token
        if effective_cost is None:
            cost_env = os.environ.get("FOUNDRY_COST_PER_TOKEN")
            if cost_env is not None:
                effective_cost = float(cost_env)

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
                sweep_result = self._run_sweep_for_quant(
                    model_file, model_id, cost_per_token=effective_cost
                )
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

    def _run_sweep_for_quant(
        self,
        model_file: str,
        model_id: str,
        cost_per_token: float | None = None,
    ) -> QuantizationResult:
        """Run the benchmark suite for a single quantization.

        Returns a ``QuantizationResult`` with metrics parsed from pytest output.
        If *cost_per_token* is provided, ``cost_per_task`` is computed.
        ``token_efficiency`` is computed when ``total_tokens`` and
        ``avg_cycle_time_s`` are both available (requires trace integration).
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
        task_results: list[TaskResult] = []

        output_lines = (result.stdout or "").splitlines()
        for line in output_lines:
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

            stripped = line.strip()
            if " PASSED" in stripped or " FAILED" in stripped:
                task_name = self._extract_task_name(stripped)
                if task_name:
                    task_results.append(
                        TaskResult(
                            name=task_name,
                            passed="PASSED" in stripped,
                        )
                    )

        if passed + failed > 0:
            total = passed + failed

        pass_rate = (passed / total) if total > 0 else 0.0

        trace_path = os.environ.get("FOUNDRY_TRACE_PATH", "./logs/traces.db")
        total_tokens, avg_cycle_time_s = self._compute_token_metrics(model_id, trace_path)

        token_efficiency: float | None = None
        if total_tokens > 0 and avg_cycle_time_s is not None and avg_cycle_time_s > 0:
            token_efficiency = total_tokens / avg_cycle_time_s

        cost_per_task: float | None = None
        if cost_per_token is not None and passed > 0 and total_tokens > 0:
            cost_per_task = (total_tokens * cost_per_token) / passed

        return QuantizationResult(
            quantization=quant_label,
            model_path=model_file,
            model_id=model_id,
            total_tasks=total,
            passed_tasks=passed,
            failed_tasks=failed,
            task_shaped_failures=task_shaped,
            pass_rate=pass_rate,
            total_tokens=total_tokens,
            avg_cycle_time_s=avg_cycle_time_s,
            token_efficiency=token_efficiency,
            cost_per_task=cost_per_task,
            task_results=task_results,
        )

    def _extract_task_name(self, line: str) -> str | None:
        """Extract task name from a pytest verbose output line.

        Handles formats:
        - test_file.py::test_name[case] PASSED
        - test_file.py::test_name PASSED
        """
        import re

        pattern = r"::(test_\w+)"
        match = re.search(pattern, line)
        if match:
            return match.group(1)
        return None

    def _compute_token_metrics(
        self,
        model_id: str,
        trace_path: str | None = None,
    ) -> tuple[int, float | None]:
        """Query the trace store for sessions matching *model_id* and compute token metrics.

        After a pytest run creates sessions in the trace store, this method
        identifies sessions by their ``model_id`` and computes:
        - ``total_tokens``: sum of ``model_response.token_usage.total_tokens``
          across all steps in all matching sessions
        - ``avg_cycle_time_s``: mean wall-clock time from ``task_received``
          to ``outcome`` across sessions that have both events

        Returns ``(total_tokens, avg_cycle_time_s)``. Returns ``(0, None)`` if
        no matching sessions are found or the trace store cannot be read.

        The caller is responsible for computing
        ``token_efficiency = total_tokens / avg_cycle_time_s`` when both
        values are non-zero / non-None.
        """
        if trace_path is None:
            trace_path = os.environ.get("FOUNDRY_TRACE_PATH", "./logs/traces.db")

        try:
            logger = TraceLogger(trace_path)
        except Exception:
            return 0, None

        sessions = logger.list_sessions()
        matching = [s for s in sessions if s.model_id == model_id]

        if not matching:
            return 0, None

        total_tokens = 0
        cycle_times: list[float] = []

        for session in matching:
            events = logger.load_session(session.session_id)

            task_received_ts: str | None = None
            outcome_ts: str | None = None

            for event in events:
                kind = event.kind
                if kind == "task_received":
                    task_received_ts = event.timestamp
                elif kind == "outcome":
                    outcome_ts = event.timestamp
                elif kind == "model_response":
                    token_usage = event.payload.get("token_usage")
                    if token_usage and isinstance(token_usage, dict):
                        total_tokens += token_usage.get("total_tokens", 0)

            if task_received_ts and outcome_ts:
                try:
                    t0 = datetime.fromisoformat(task_received_ts)
                    t1 = datetime.fromisoformat(outcome_ts)
                    delta = (t1 - t0).total_seconds()
                    if delta > 0:
                        cycle_times.append(delta)
                except ValueError:
                    pass

        avg_cycle_time_s = sum(cycle_times) / len(cycle_times) if cycle_times else None
        return total_tokens, avg_cycle_time_s

    def evaluate(
        self, proposed_diff: str, *, edit_index: int | None = None, failure_class: str | None = None
    ) -> CriticVerdict:
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
                        edit_index=edit_index,
                        failure_class=failure_class,
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
                        edit_index=edit_index,
                        failure_class=failure_class,
                    )
                try:
                    apply_result = subprocess.run(
                        ["git", "apply", "--whitespace=nowarn"],
                        input=proposed_diff,
                        cwd=sandbox_root.parent,
                        capture_output=True,
                        text=True,
                        timeout=self.gate_timeout_s,
                    )
                except subprocess.TimeoutExpired as exc:
                    return CriticVerdict(
                        verdict=False,
                        passed_checks=passed_checks,
                        failed_checks=[*failed_checks, "git apply:timeout"],
                        notes=_timeout_notes(exc),
                        edit_index=edit_index,
                        failure_class=failure_class,
                    )
                if apply_result.returncode != 0:
                    return CriticVerdict(
                        verdict=False,
                        passed_checks=[],
                        failed_checks=["git apply"],
                        notes=_tail(apply_result.stderr or apply_result.stdout),
                        edit_index=edit_index,
                        failure_class=failure_class,
                    )
                passed_checks.append("git apply")

            # Gate 3: Precondition gate (issue #187).
            #    against the sandbox copy. A harness tree that fails to load
            #    must fail the gate *before* pytest runs.
            load_check_script = sandbox_root / "scripts" / "load_check.py"
            try:
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
                    timeout=self.gate_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                return CriticVerdict(
                    verdict=False,
                    passed_checks=passed_checks,
                    failed_checks=[*failed_checks, "load_check:timeout"],
                    notes=_timeout_notes(exc),
                    edit_index=edit_index,
                    failure_class=failure_class,
                )
            if load_result.returncode != 0:
                return CriticVerdict(
                    verdict=False,
                    passed_checks=passed_checks,
                    failed_checks=[*failed_checks, "load_check"],
                    notes=_tail(load_result.stderr or load_result.stdout),
                    edit_index=edit_index,
                    failure_class=failure_class,
                )
            passed_checks.append("load_check")

            # Gate 4: Pytest benchmark suite (issue #548).
            # Plumb token_budget from BenchmarkTask through to the subprocess
            # via FOUNDRY_TOKEN_BUDGET so the Runner enforces it.
            token_budget: int | None = None
            for task in self.benchmark_tasks:
                if task.token_budget is not None:
                    if token_budget is None or task.token_budget < token_budget:
                        token_budget = task.token_budget
            pytest_env = os.environ.copy()
            if token_budget is not None:
                pytest_env["FOUNDRY_TOKEN_BUDGET"] = str(token_budget)
            try:
                pytest_result = subprocess.run(
                    [sys.executable, "-m", "pytest", *self.pytest_args],
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                    env=pytest_env,
                    timeout=self.gate_timeout_s,
                )
            except subprocess.TimeoutExpired as exc:
                return CriticVerdict(
                    verdict=False,
                    passed_checks=passed_checks,
                    failed_checks=[*failed_checks, "pytest:timeout"],
                    notes=_timeout_notes(exc),
                    edit_index=edit_index,
                    failure_class=failure_class,
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
                edit_index=edit_index,
                failure_class=failure_class,
            )


def _tail(text: str) -> str:
    """Return the trailing window of *text* for inclusion in verdict notes."""
    return text.strip()[-_NOTES_TAIL_CHARS:]


def _timeout_notes(exc: subprocess.TimeoutExpired) -> str:
    """Build a ``CriticVerdict.notes`` string from a ``TimeoutExpired``.

    Mirrors the contract in :meth:`Critic.evaluate`'s docstring: the trailing
    window of any partial output the process managed to write before being
    killed, or a wall-clock-cap message when no partial output was captured.

    Handles both ``str`` (``text=True``) and ``bytes`` payloads defensively,
    since :class:`subprocess.TimeoutExpired` attributes are not guaranteed to
    be populated on every platform when ``subprocess.run`` kills the child.
    """
    raw_out: object = exc.output
    raw_err: object = exc.stderr
    if isinstance(raw_out, bytes):
        raw_out = raw_out.decode("utf-8", errors="replace")
    if isinstance(raw_err, bytes):
        raw_err = raw_err.decode("utf-8", errors="replace")
    out_s = raw_out if isinstance(raw_out, str) else ""
    err_s = raw_err if isinstance(raw_err, str) else ""
    combined = f"{out_s}\n{err_s}".strip()
    if combined:
        return _tail(combined)
    return f"subprocess exceeded gate_timeout_s={exc.timeout}s and was killed"
