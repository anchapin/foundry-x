from __future__ import annotations

import base64
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from benchmarks.models import BenchmarkTask
from benchmarks.registry import load_all_tasks
from foundry_x.trace.logger import TraceLogger

DEFAULT_REGRESSION_THRESHOLD_PP = 2.0

#: Default tag set for the fast-reject ("smoke") tier (issue #1042). The
#: ``test_smoke.py`` task carries ``tags=["smoke", "infrastructure"]``, so the
#: default subset is the single infrastructure canary that every harness edit
#: must keep green. Operators override this via the ``smoke_benchmark_tags``
#: constructor argument or the ``FOUNDRY_SMOKE_BENCHMARK_TAGS`` env var.
DEFAULT_SMOKE_BENCHMARK_TAGS: list[str] = ["smoke"]

#: Env var name for runtime configuration of the smoke-tier benchmark subset
#: (issue #1042). Comma-separated tag values, e.g. ``"smoke,core"``.
_SMOKE_BENCHMARK_TAGS_ENV = "FOUNDRY_SMOKE_BENCHMARK_TAGS"

#: Env var name for runtime configuration of the gate timeout (issue #1172).
#: When ``Critic.__init__`` receives ``gate_timeout_s=None`` (the default),
#: the value is resolved from this env var so operators can set an escape-hatch
#: timeout without a code change.  A positive float bounds every subprocess
#: spawned inside ``evaluate()`` so a hanging child cannot inflate
#: ``kpi-cycle-time`` to infinity.  ``None`` preserves the historical
#: unbounded behaviour.
_GATE_TIMEOUT_S_ENV = "FOUNDRY_GATE_TIMEOUT_S"

_NOTES_TAIL_CHARS = 4000

#: Patterns for files that are NOT critical to benchmark outcomes.
#: A diff that touches ONLY these files (and no harness/, benchmarks/tasks/,
#: or src/ files) can early-exit the Critic gate without running pytest.
_NON_CRITICAL_PATTERNS: tuple[str, ...] = (
    "docs/",
    ".pre-commit-config.yaml",
    "pyproject.toml",
)

#: Patterns for files that ARE critical to benchmark outcomes.
#: A diff touching ANY of these must run the full Critic gate.
_CRITICAL_PATTERNS: tuple[str, ...] = (
    "harness/",
    "benchmarks/tasks/",
    "src/",
)


def _diff_touches_only_non_critical_files(diff: str) -> bool:
    """Return True when every file path in *diff* is non-critical.

    Parses ``--- a/<path>`` lines from the unified diff to extract the set of
    files the diff touches, then checks each path against
    :data:`_CRITICAL_PATTERNS`.  If none of the touched files match a critical
    pattern, the diff is a no-op for benchmark purposes and the Critic gate can
    exit early without running pytest.

    Args:
        diff: A unified-diff string (same format as
            ``Critic.evaluate(proposed_diff=...)``).

    Returns:
        True when all touched files are non-critical (docs/, .pre-commit-config.yaml,
        or pyproject.toml); False when any touched file is critical or when the
        diff is empty.
    """
    if not diff.strip():
        return False
    touched_files: list[str] = []
    for line in diff.splitlines():
        if line.startswith("--- a/"):
            path = line[5:].lstrip("/")
            touched_files.append(path)
    if not touched_files:
        return False
    for path in touched_files:
        for critical in _CRITICAL_PATTERNS:
            if path.startswith(critical):
                return False
    return True


#: GGUF v3 quantization types studied in ADR-0020 (K-quants and legacy
#: types). These are the quantizations for which the intelligence floor
#: table in ADR-0020 has (projected) pass-rate data.
KNOWN_V3_QUANTIZATIONS: tuple[str, ...] = (
    "Q8_0",
    "Q6_K",
    "Q5_K_M",
    "Q5_K_S",
    "Q4_K_M",
    "Q4_K_S",
)

#: GGUF v4 quantization types (IQ family) added in issue #1050. These use
#: importance-matrix (imatrix) based quantization and offer different
#: quality/VRAM tradeoffs than the v3 K-quants. ``Q2_K`` is included as a
#: baseline at the aggressive end of the VRAM/compression spectrum.
KNOWN_V4_QUANTIZATIONS: tuple[str, ...] = (
    "IQ4_XS",
    "IQ3_S",
    "IQ3_XXS",
    "IQ2_XXS",
    "Q2_K",
)

#: All known GGUF quantization types (v3 + v4). Used for documentation and
#: optional validation; the sweep accepts arbitrary labels so non-standard
#: or future quantizations are not blocked.
KNOWN_QUANTIZATIONS: tuple[str, ...] = (*KNOWN_V3_QUANTIZATIONS, *KNOWN_V4_QUANTIZATIONS)

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


def _parse_model_registry() -> dict[str, dict[str, str]] | None:
    """Parse FOUNDRY_MODEL_REGISTRY env var into a family-config dict.

    Returns None when the env var is absent or invalid. A missing key in
    the registry is not fatal — the caller skips that family with a warning.
    """
    raw = os.environ.get("FOUNDRY_MODEL_REGISTRY")
    if not raw:
        return None
    try:
        import json

        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            sys.stderr.write(
                "FOUNDRY_MODEL_REGISTRY must be a JSON object mapping "
                "family names to {path, quantization, endpoint} objects.\n"
            )
            return None
        for key, value in parsed.items():
            if not isinstance(value, dict):
                sys.stderr.write(
                    f"FOUNDRY_MODEL_REGISTRY entry for {key!r} must be an object "
                    f"with 'path', 'quantization', and 'endpoint' keys.\n"
                )
                return None
        return parsed
    except json.JSONDecodeError as exc:
        sys.stderr.write(f"FOUNDRY_MODEL_REGISTRY is not valid JSON: {exc}\n")
        return None


def _resolve_smoke_tags(explicit: list[str] | None) -> list[str]:
    """Resolve the smoke-tier benchmark tag set (issue #1042).

    Resolution order: an explicit constructor argument wins; otherwise the
    ``FOUNDRY_SMOKE_BENCHMARK_TAGS`` env var is parsed as a comma-separated
    list (whitespace trimmed, empties dropped); otherwise the
    ``DEFAULT_SMOKE_BENCHMARK_TAGS`` constant is used.

    Always returns a fresh, deduplicated list so callers can mutate it
    without aliasing the module constant.
    """
    if explicit is not None:
        return list(dict.fromkeys(explicit))
    raw = os.environ.get(_SMOKE_BENCHMARK_TAGS_ENV, "").strip()
    if raw:
        tags = [t.strip() for t in raw.split(",") if t.strip()]
        if tags:
            return list(dict.fromkeys(tags))
    return list(DEFAULT_SMOKE_BENCHMARK_TAGS)


def _resolve_gate_timeout(explicit: float | None) -> float | None:
    """Resolve the gate timeout (issue #1172).

    Resolution order: an explicit constructor argument wins; otherwise the
    ``FOUNDRY_GATE_TIMEOUT_S`` env var is parsed as a float; otherwise
    ``None`` (unbounded) is returned, preserving the historical behaviour.

    ``None`` is returned when the env var is absent, empty, or not a valid
    positive float, so that an operator can unset the env var to restore the
    default unbounded behaviour without editing code.
    """
    if explicit is not None:
        return explicit
    raw = os.environ.get(_GATE_TIMEOUT_S_ENV, "").strip()
    if not raw:
        return None
    try:
        value = float(raw)
    except ValueError:
        return None
    if value <= 0:
        return None
    return value


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
        except ValueError:
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
    """Result of a Critic gate run against a proposed harness edit (ADR-0006).

    ``verdict`` is ``True`` when the Critic approved the edit, ``False`` when
    it rejected the edit, and ``None`` when the gate was explicitly skipped
    via ``--no-verify`` (issue #888). The ``None`` value preserves the audit
    trail: ``record_verdict`` still persists a ``critic_verdict`` trace event
    so downstream consumers (regression report, KPIs) can distinguish a
    skipped gate from a rejected one.
    """

    verdict: bool | None
    passed_checks: list[str] = Field(default_factory=list)
    failed_checks: list[str] = Field(default_factory=list)
    skipped_checks: list[str] = Field(default_factory=list)
    notes: str = ""
    edit_index: int | None = None
    failure_class: str | None = None
    target_file: str | None = None


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
    notes: str = Field(
        default="",
        description=(
            "Free-form annotation. Populated when gate_timeout_s kills the "
            "sweep subprocess so the verdict records why pass_rate is 0.0 "
            "(issue #937)."
        ),
    )


class QuantizationVerdict(BaseModel):
    """Sweep-level verdict aggregating per-quantization results (ADR-0016)."""

    quantizations: list[QuantizationResult]
    recommended: str
    regression: bool


class ModelFamilySweepResult(BaseModel):
    """Per-family aggregation of QuantizationResult lists (ADR-0025)."""

    model_family: str
    results: list[QuantizationResult]
    recommended: str
    regression: bool


class ModelFamilyVerdict(BaseModel):
    """Cross-family verdict aggregating per-family sweep results (ADR-0025)."""

    family_results: list[ModelFamilySweepResult]
    recommended_family: str
    recommended_quantization: str
    regression: bool
    regression_by_family: dict[str, bool] = Field(default_factory=dict)


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
        smoke_benchmark_tags: list[str] | None = None,
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
        # Resolution order: explicit constructor argument →
        # ``FOUNDRY_GATE_TIMEOUT_S`` env var → ``None`` (issue #1172).
        resolved_gate_timeout = _resolve_gate_timeout(gate_timeout_s)
        if resolved_gate_timeout is not None and resolved_gate_timeout <= 0:
            raise ValueError("gate_timeout_s must be > 0 or None")
        self.gate_timeout_s = resolved_gate_timeout
        # Two-tier gate (issue #1042): the fast-reject ("smoke") tier runs a
        # configurable subset of benchmark tasks before the full suite. The
        # subset is selected by intersecting each task's ``tags`` with
        # ``smoke_benchmark_tags``. Resolution order: explicit constructor
        # argument → ``FOUNDRY_SMOKE_BENCHMARK_TAGS`` env var (comma-separated)
        # → ``DEFAULT_SMOKE_BENCHMARK_TAGS``.
        self.smoke_benchmark_tags = _resolve_smoke_tags(smoke_benchmark_tags)
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

    @property
    def smoke_tasks(self) -> list[BenchmarkTask]:
        """Benchmark tasks selected for the fast-reject ("smoke") tier (issue #1042).

        A task is selected when any of its ``tags`` intersects
        ``self.smoke_benchmark_tags``. The registry is the source of truth
        (ADR-0005); tags are free-form grouping labels on ``BenchmarkTask``.
        Returns a fresh list; empty when no task carries a smoke tag (a
        misconfiguration — :meth:`_smoke_pytest_args` falls back to the full
        suite so the gate never rejects on an empty selection).
        """
        smoke_set = set(self.smoke_benchmark_tags)
        return [t for t in self.benchmark_tasks if smoke_set & set(t.tags)]

    def _smoke_pytest_args(self) -> list[str]:
        """Build pytest args selecting only smoke-tagged benchmark tasks (issue #1042).

        The smoke tier runs the same cheap gates as the full tier, then runs
        pytest restricted to the subset of ``@pytest.mark.benchmark`` tasks
        whose ``tags`` intersect :attr:`smoke_benchmark_tags`. Selection uses
        a pytest ``-k`` OR-expression built from the matched task names —
        each ``BenchmarkTask.name`` is a substring of its test function
        (``test_<name>``), so ``-k <name>`` matches it without coupling to
        the on-disk file layout.

        When no task carries a smoke tag the method returns the full
        ``self.pytest_args`` unchanged: an empty selection would otherwise
        make pytest exit 5 ("no tests ran") and falsely reject the edit.
        Falling back to the full suite is safe (it is the existing default
        behaviour) and surfaces the misconfiguration at the same cost.
        """
        names = sorted({t.name for t in self.smoke_tasks})
        if not names:
            return list(self.pytest_args)
        expr = " or ".join(names)
        return [*self.pytest_args, "-k", expr]

    def quantization_sweep(
        self,
        quantizations: list[str],
        model_glob_patterns: dict[str, str] | None = None,
        baseline_quantization: str | None = None,
        regression_threshold_pp: float = DEFAULT_REGRESSION_THRESHOLD_PP,
        cost_per_token: float | None = None,
        context_tokens: int | None = None,
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
            context_tokens: context window size in tokens to set via the
                ``FOUNDRY_CONTEXT_TOKENS`` environment variable for the sweep
                subprocesses (issue #1050). When ``None``, the existing env
                value is left untouched. Useful for studying the intelligence
                floor at different context window sizes (16k, 32k, 128k).

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

        # Set the context window for this sweep run (issue #1050). The
        # env var propagates to the pytest subprocesses spawned in
        # ``_run_sweep_for_quant`` (which inherit ``os.environ``). The
        # original value is restored after the sweep so concurrent callers
        # are unaffected.
        original_context_tokens = os.environ.get("FOUNDRY_CONTEXT_TOKENS")
        if context_tokens is not None:
            os.environ["FOUNDRY_CONTEXT_TOKENS"] = str(context_tokens)

        results: list[QuantizationResult] = []

        if os.environ.get("FOUNDRY_PARALLEL_SWEEP") == "1":
            max_workers = max(1, (os.cpu_count() or 2) - 1)

            quant_work: list[tuple[str, str]] = []
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
                quant_work.append((model_file, model_id))

            def _run_for_quant_in_env(
                model_file: str,
                model_id: str,
                base_env: dict[str, str],
            ) -> QuantizationResult:
                env = os.environ.copy()
                env.update(base_env)
                env["FOUNDRY_MODEL_PATH"] = str(model_base)
                env["FOUNDRY_MODEL_ID"] = model_id
                return self._run_sweep_for_quant(
                    model_file,
                    model_id,
                    cost_per_token=effective_cost,
                    env=env,
                )

            base_env: dict[str, str] = {}
            if context_tokens is not None:
                base_env["FOUNDRY_CONTEXT_TOKENS"] = str(context_tokens)

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {
                    executor.submit(_run_for_quant_in_env, mf, mid, base_env): (mf, mid)
                    for mf, mid in quant_work
                }
                for future in as_completed(futures):
                    results.append(future.result())
        else:
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

        if context_tokens is not None:
            if original_context_tokens is not None:
                os.environ["FOUNDRY_CONTEXT_TOKENS"] = original_context_tokens
            else:
                os.environ.pop("FOUNDRY_CONTEXT_TOKENS", None)

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

    def model_family_sweep(
        self,
        model_families: list[str],
        quantizations: list[str],
        baseline_quantization: str | None = None,
        regression_threshold_pp: float = DEFAULT_REGRESSION_THRESHOLD_PP,
        cost_per_token: float | None = None,
    ) -> ModelFamilyVerdict:
        """Run the benchmark suite across multiple model families and quantizations.

        ADR-0025: two-axis sweep (model family + quantization) that enables
        cross-model-family comparison. Each family is swept independently and
        results are aggregated into a ModelFamilyVerdict.

        Args:
            model_families: list of model family identifiers to sweep.
            quantizations: quantization labels to sweep within each family.
            baseline_quantization: quantization to compare against per family.
            regression_threshold_pp: regression threshold in percentage points.
            cost_per_token: cost per token in USD for cost-per-task computation.

        Returns:
            A ``ModelFamilyVerdict`` with per-family results, recommended
            family+quantization combination, and regression flags.
        """
        registry = _parse_model_registry()
        family_results: list[ModelFamilySweepResult] = []
        regression_by_family: dict[str, bool] = {}

        for family in model_families:
            family_config = registry.get(family) if registry else None

            if family_config is None:
                sys.stderr.write(
                    f"Warning: family {family!r} not in FOUNDRY_MODEL_REGISTRY, "
                    f"skipping. Set FOUNDRY_MODEL_REGISTRY to include it.\n"
                )
                continue

            model_path = family_config.get("path", "")
            model_endpoint = family_config.get("endpoint", "")

            original_model_path = os.environ.get("FOUNDRY_MODEL_PATH")
            original_model_id = os.environ.get("FOUNDRY_MODEL_ID")
            original_endpoint = os.environ.get("FOUNDRY_LLAMACPP_HOST")

            os.environ["FOUNDRY_MODEL_PATH"] = str(Path(model_path).parent) if model_path else ""
            os.environ["FOUNDRY_MODEL_ID"] = family
            if model_endpoint:
                os.environ["FOUNDRY_LLAMACPP_HOST"] = model_endpoint

            try:
                verdict = self.quantization_sweep(
                    quantizations=quantizations,
                    baseline_quantization=baseline_quantization,
                    regression_threshold_pp=regression_threshold_pp,
                    cost_per_token=cost_per_token,
                )
                family_results.append(
                    ModelFamilySweepResult(
                        model_family=family,
                        results=verdict.quantizations,
                        recommended=verdict.recommended,
                        regression=verdict.regression,
                    )
                )
                regression_by_family[family] = verdict.regression
            finally:
                if original_model_path is not None:
                    os.environ["FOUNDRY_MODEL_PATH"] = original_model_path
                else:
                    os.environ.pop("FOUNDRY_MODEL_PATH", None)
                if original_model_id is not None:
                    os.environ["FOUNDRY_MODEL_ID"] = original_model_id
                else:
                    os.environ.pop("FOUNDRY_MODEL_ID", None)
                if original_endpoint is not None:
                    os.environ["FOUNDRY_LLAMACPP_HOST"] = original_endpoint
                elif model_endpoint:
                    os.environ.pop("FOUNDRY_LLAMACPP_HOST", None)

        if not family_results:
            raise ValueError("No families could be swept. Check FOUNDRY_MODEL_REGISTRY.")

        best_family_result = max(family_results, key=lambda f: max(r.pass_rate for r in f.results))
        recommended_family = best_family_result.model_family
        recommended_quantization = best_family_result.recommended
        overall_regression = any(regression_by_family.values())

        return ModelFamilyVerdict(
            family_results=family_results,
            recommended_family=recommended_family,
            recommended_quantization=recommended_quantization,
            regression=overall_regression,
            regression_by_family=regression_by_family,
        )

    def _run_sweep_for_quant(
        self,
        model_file: str,
        model_id: str,
        cost_per_token: float | None = None,
        env: dict[str, str] | None = None,
    ) -> QuantizationResult:
        """Run the benchmark suite for a single quantization.

        Returns a ``QuantizationResult`` with metrics parsed from pytest output.
        If *cost_per_token* is provided, ``cost_per_task`` is computed.
        ``token_efficiency`` is computed when ``total_tokens`` and
        ``avg_cycle_time_s`` are both available (requires trace integration).

        When *env* is provided it is passed to the subprocess as its environment
        (derived from a copy of ``os.environ`` plus any overrides). This keeps
        per-quantization env vars (FOUNDRY_MODEL_PATH, FOUNDRY_MODEL_ID,
        FOUNDRY_CONTEXT_TOKENS) isolated in the parallel path so concurrent
        workers do not interfere with each other.
        """
        quant_label = Path(model_file).stem
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", *self.pytest_args, "--tb=no", "-v"],
                capture_output=True,
                text=True,
                timeout=self.gate_timeout_s,
                check=False,
                env=env,
            )
        except subprocess.TimeoutExpired as exc:
            # gate_timeout_s killed the sweep subprocess — return a
            # zero-pass-rate result with a timeout note so the verdict records
            # *why* rather than hanging indefinitely (issue #937).
            return QuantizationResult(
                quantization=quant_label,
                model_path=model_file,
                model_id=model_id,
                pass_rate=0.0,
                notes=_timeout_notes(exc),
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
        except Exception:  # Defensive: trace_path may be corrupt or locked  # noqa: BLE001
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
        self,
        proposed_diff: str,
        *,
        edit_index: int | None = None,
        failure_class: str | None = None,
        tier: Literal["smoke", "full"] = "full",
    ) -> CriticVerdict:
        """Apply ``proposed_diff`` to a sandbox copy of the harness and gate it.

        Two-tier gate (issue #1042): the ``tier`` argument selects which
        benchmark subset the pytest gate runs after the cheap gates pass.

        * ``tier="full"`` (default) — existing behaviour: the full
          ``@pytest.mark.benchmark`` suite via ``self.pytest_args``.
        * ``tier="smoke"`` — fast-reject: run only the subset of benchmark
          tasks whose ``tags`` intersect :attr:`smoke_benchmark_tags`
          (computed by :meth:`_smoke_pytest_args`). A smoke-tier failure
          rejects the edit *without* running the full suite, so
          ``kpi-cycle-time`` for obvious rejections drops to the cost of the
          cheap gates + the smoke subset. The caller decides whether to
          escalate a passing smoke verdict to the full suite.

        All other steps (sandbox copy, diff-size cap, injection scan,
        ``git apply``, ``load_check``) run identically in both tiers —
        the only difference is which benchmark tasks the pytest subprocess
        executes.

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
            skipped_checks: list[str] = []

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
                # Gate 3: No-op diff check (issue #1347).
                #    A diff that touches only non-critical files (docs/, CI configs,
                #    .pre-commit-config.yaml, pyproject.toml) cannot affect benchmark
                #    outcomes, so we approve it immediately without spawning pytest.
                if _diff_touches_only_non_critical_files(proposed_diff):
                    return CriticVerdict(
                        verdict=True,
                        passed_checks=["diff_noop"],
                        failed_checks=[],
                        notes="diff touches only non-critical files; gate skipped",
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
                        check=False,
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
                    check=False,
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
            # Two-tier gate (issue #1042): the smoke tier restricts the
            # subprocess to the smoke-tagged benchmark subset so an obvious
            # rejection never pays the full-suite cost. Plumb token_budget
            # from BenchmarkTask through to the subprocess via
            # FOUNDRY_TOKEN_BUDGET so the Runner enforces it.
            effective_pytest_args = (
                self._smoke_pytest_args() if tier == "smoke" else self.pytest_args
            )
            token_budget: int | None = None
            for task in self.benchmark_tasks:
                if task.token_budget is not None and (
                    token_budget is None or task.token_budget < token_budget
                ):
                    token_budget = task.token_budget
            pytest_env = os.environ.copy()
            if token_budget is not None:
                pytest_env["FOUNDRY_TOKEN_BUDGET"] = str(token_budget)
            try:
                pytest_result = subprocess.run(
                    [sys.executable, "-m", "pytest", *effective_pytest_args],
                    cwd=sandbox_root,
                    capture_output=True,
                    text=True,
                    env=pytest_env,
                    timeout=self.gate_timeout_s,
                    check=False,
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
                # The smoke tier covers only the smoke-tagged subset; the full
                # tier covers the whole registry (issue #1042).
                covered_source = self.smoke_tasks if tier == "smoke" else self.benchmark_tasks
                covered_tags = sorted({tag for task in covered_source for tag in task.tags})
                passed_checks.extend(f"benchmark:{tag}" for tag in covered_tags)
            elif pytest_result.returncode == 5:
                # Exit code 5 means "no tests collected" — the smoke-tier -k
                # expression matched nothing (issue #1351). Distinguish this
                # from a genuine test failure (exit 1) so the verdict correctly
                # surfaces the misconfiguration rather than a generic "pytest"
                # label.
                failed_checks.append("pytest:no_tests_collected")
                # Record the selected-but-not-collected task names as skipped
                # (issue #1349). In smoke tier the -k expression names the smoke
                # tasks; in full tier this case should not occur.
                skipped_names = [t.name for t in self.smoke_tasks] if tier == "smoke" else []
                skipped_checks.extend(skipped_names)

            else:
                failed_checks.append("pytest")

            # When smoke tier fails, record the full-suite tasks that were never
            # run as skipped_checks (issue #1349). These are benchmark tasks
            # whose tags do NOT intersect smoke_benchmark_tags.
            if tier == "smoke" and failed_checks:
                smoke_names = {t.name for t in self.smoke_tasks}
                skipped_names = [t.name for t in self.benchmark_tasks if t.name not in smoke_names]
                skipped_checks.extend(skipped_names)

            combined = (pytest_result.stdout or "") + (pytest_result.stderr or "")
            return CriticVerdict(
                verdict=not failed_checks,
                passed_checks=passed_checks,
                failed_checks=failed_checks,
                skipped_checks=skipped_checks,
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
