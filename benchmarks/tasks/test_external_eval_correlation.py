"""Benchmark task: validate external-eval plumbing against HumanEval+ slice.

This is the offline, deterministic half of issue #900. The real-model
half is orchestrated by ``infra/scripts/run_external_eval.sh``; this
task exists so the Critic gate (ADR-0004) catches regressions in the
plumbing without spending model tokens.

What this task verifies
-----------------------
1. **Slice integrity.** ``load_humaneval_slice`` parses the 20-task
   ``humaneval_plus_sample.jsonl`` and every canonical solution passes
   its own ``check`` via ``run_canonical_solution``. A broken slice
   would silently degrade the real-model study (every task would score
   ``False`` regardless of agent output); this assertion catches that
   regression at pytest time.

2. **Candidate-scoring path.** A deliberately-wrong candidate body
   scores ``False`` and a syntactically-broken candidate raises
   :class:`HumanEvalExecutionError`. The orchestrator script depends on
   both branches behaving this way; if either regressed, the
   correlation study would report a misleadingly high (or undefined)
   pass rate.

3. **Correlation math.** A synthetic paired-series scenario with known
   answer is run through ``pearson_binary`` (with ``min_pairs``
   overridden for the test). This locks the math so a future refactor
   cannot silently break the headline number the ADR reports.

4. **Interpretation thresholds.** The ``interpret_correlation`` mapping
   is asserted at each ADR-0023 threshold (``0.7``, ``0.3``) so a
   casual tweak to the thresholds surfaces as a test failure rather
   than as a quiet shift in the study's verdict.

The task does **not** invoke the Runner, the TraceLogger, or any model
adapter. It exercises pure machinery under ``src/foundry_x/evaluation/``
plus the committed slice under ``benchmarks/external/``. That keeps the
task within the "<30 s on a developer laptop" budget ADR-0005 inherits
from the rest of the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from benchmarks.models import BenchmarkTask
from foundry_x.evaluation.correlation import (
    UnderpoweredStudyError,
    ZeroVarianceError,
    interpret_correlation,
    pearson_binary,
)
from foundry_x.evaluation.humaneval_plus import (
    HumanEvalExecutionError,
    load_humaneval_slice,
    run_canonical_solution,
    run_candidate_solution,
    slice_pass_rates,
)

TASK = BenchmarkTask(
    name="external_eval_correlation",
    description=(
        "Validate the external-eval plumbing (HumanEval+ slice loader + "
        "Pearson correlation math) offline, without invoking a live model."
    ),
    prompt=(
        "Loads the 20-task HumanEval+ slice, runs every canonical solution "
        "against its own check, exercises the candidate-scoring path with "
        "known-good and known-bad candidates, and verifies the Pearson "
        "correlation math on a synthetic paired series."
    ),
    difficulty_tier="medium",
    expected_outcome=(
        "All 20 canonical solutions pass; wrong candidate scores False; "
        "broken candidate raises HumanEvalExecutionError; synthetic "
        "correlation matches the textbook value to 1e-9."
    ),
    tags=["external-eval", "validation"],
)


SLICE_PATH = Path(__file__).resolve().parent.parent / "external" / "humaneval_plus_sample.jsonl"


@pytest.mark.benchmark
def test_external_eval_correlation() -> None:
    """Deterministic plumbing-validation check for TASK (issue #900)."""
    # --- 1. Slice integrity ------------------------------------------------
    assert SLICE_PATH.is_file(), f"slice missing: {SLICE_PATH}"
    tasks = load_humaneval_slice(SLICE_PATH)
    assert len(tasks) >= 20, f"issue #900 requires a >=20-task slice; got {len(tasks)}"

    passed, total = slice_pass_rates(tasks)
    assert passed == total, (
        f"slice integrity failed: {passed}/{total} canonical solutions passed; "
        "a broken canonical solution would silently inflate the real-model "
        "study's external pass rate. Re-verify the JSONL."
    )

    # --- 2. Candidate-scoring path ----------------------------------------
    sample = tasks[0]
    assert run_canonical_solution(sample) is True

    wrong_body = "    return None\n"
    assert run_candidate_solution(sample, wrong_body) is False, (
        "candidate-scoring path returned True for a deliberately-wrong body; "
        "the orchestrator script depends on this branch returning False."
    )

    broken_body = "    return this is not valid python!!!\n"
    with pytest.raises(HumanEvalExecutionError):
        run_candidate_solution(sample, broken_body)

    # --- 3. Correlation math ----------------------------------------------
    # Textbook example: perfect positive correlation -> 1.0.
    perfect_xs = [float(i) for i in range(40)]
    perfect_ys = [2.0 * x + 1.0 for x in perfect_xs]
    r_perfect = pearson_binary(perfect_xs, perfect_ys, min_pairs=10)
    assert abs(r_perfect - 1.0) < 1e-9, f"perfect positive correlation expected, got {r_perfect}"

    # Perfect negative correlation -> -1.0.
    inverse_ys = [-2.0 * x for x in perfect_xs]
    r_inverse = pearson_binary(perfect_xs, inverse_ys, min_pairs=10)
    assert abs(r_inverse - (-1.0)) < 1e-9, f"perfect negative correlation expected, got {r_inverse}"

    # No correlation -> 0.0 (constructed so the math is exact).
    uncorrelated_xs = [-1.0, -1.0, 1.0, 1.0] * 10
    uncorrelated_ys = [-1.0, 1.0, -1.0, 1.0] * 10
    r_zero = pearson_binary(uncorrelated_xs, uncorrelated_ys, min_pairs=10)
    assert abs(r_zero) < 1e-9, f"zero correlation expected, got {r_zero}"

    # Underpowered study surfaces rather than returning a confident NaN.
    with pytest.raises(UnderpoweredStudyError):
        pearson_binary([0.5, 0.6], [0.4, 0.7])

    # Zero-variance series surfaces with the offending series named.
    constant_xs = [0.5] * 40
    varying_ys = [float(i) / 40.0 for i in range(40)]
    with pytest.raises(ZeroVarianceError, match="internal_rates"):
        pearson_binary(constant_xs, varying_ys, min_pairs=10)

    # --- 4. Interpretation thresholds (ADR-0023) --------------------------
    assert interpret_correlation(0.7) == "valid proxy"
    assert interpret_correlation(0.95) == "valid proxy"
    assert interpret_correlation(0.69) == "weak proxy"
    assert interpret_correlation(0.3) == "weak proxy"
    assert interpret_correlation(0.29) == "invalid proxy"
    assert interpret_correlation(-0.5) == "invalid proxy"
