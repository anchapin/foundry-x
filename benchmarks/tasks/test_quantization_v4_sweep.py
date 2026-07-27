"""Benchmark task: GGUF v4 quantization sweep for 5600G/6600 XT validation (issue #1078).

Implements the ADR-0020 §GGUF v4 Quantization Extension methodology: run the
benchmark suite across v4 IQ quantizations (IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS)
plus Q2_K and the v3 K-quant baselines (Q4_K_M, Q5_K_M, Q8_0) to validate
whether any v4 IQ quantization offers a better quality/VRAM tradeoff than
Q5_K_M (~5.5 GB).

The sweep is executed via ``foundry-sweep sweep``::

    FOUNDRY_MODEL_PATH=/srv/models \\
        foundry-sweep sweep \\
        --quantizations Q2_K,IQ2_XXS,IQ3_XXS,IQ3_S,IQ4_XS,Q4_K_M,Q5_K_M,Q8_0 \\
        --harness-dir harness \\
        --baseline Q8_0 \\
        --regression-threshold 2.0 \\
        --output logs/sweep_v4.json

The key question (Open Question #7 in ADR-0020) is whether IQ4_XS (~4.1 GB)
matches Q5_K_M quality within 2 pp. If so, 8 GB card operators could run a
higher-quality model per VRAM dollar.

The sweep emits one ``V4QuantizationSweepResult`` per quantization. These
are the data contract that the KPI layer consumes for intelligence floor
characterization.

Execution requirements:
- GPU hardware (RX 6600 XT or equivalent with >= 8 GB VRAM)
- GGUF v4 model files with imatrix calibration for IQ4_XS, IQ3_S, IQ3_XXS, IQ2_XXS
- FOUNDRY_MODEL_PATH pointing to the model directory
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from benchmarks.models import BenchmarkTask, V4QuantizationSweepResult
from foundry_x.evolution.critic import KNOWN_V4_QUANTIZATIONS

TASK = BenchmarkTask(
    name="quantization_v4_sweep",
    description=(
        "GGUF v4 IQ quantization sweep: validate whether IQ4_XS, IQ3_S, and related "
        "v4 quantizations offer better quality/VRAM tradeoffs than Q5_K_M on 5600G/6600 XT. "
        "Issue #1078, ADR-0020 §GGUF v4 Quantization Extension."
    ),
    tags=["agent-loop", "quantization", "sweep", "v4", "5600G", "6600-XT", "IQ4_XS"],
    difficulty_tier="medium",
)

_V4_QUANTIZATIONS = list(KNOWN_V4_QUANTIZATIONS)

_V4_VRAM_GB: dict[str, float] = {
    "IQ4_XS": 4.1,
    "IQ3_S": 3.3,
    "IQ3_XXS": 3.1,
    "IQ2_XXS": 2.7,
    "Q2_K": 2.6,
}

_V3_BASELINE_QUANTIZATIONS = ["Q4_K_M", "Q5_K_M", "Q8_0"]

_ALL_SWEEP_QUANTIZATIONS = _V4_QUANTIZATIONS + _V3_BASELINE_QUANTIZATIONS

_REGRESSION_THRESHOLD_PP = 2.0

_SWEEP_RESULTS: list[V4QuantizationSweepResult] = []


def _run_foundry_sweep(
    quantizations: list[str],
    harness_dir: Path,
    output_path: Path,
    baseline: str = "Q8_0",
    regression_threshold: float = 2.0,
) -> dict[str, Any]:
    """Run ``foundry-sweep sweep`` and return parsed JSON verdict.

    Parameters
    ----------
    quantizations:
        Comma-separated quantization labels to sweep.
    harness_dir:
        Path to the harness directory.
    output_path:
        Path to write the JSON verdict output.
    baseline:
        Baseline quantization for regression comparison.
    regression_threshold:
        Regression threshold in percentage points.

    Returns
    -------
    dict[str, Any]
        Parsed JSON verdict from the sweep.

    Raises
    ------
    RuntimeError
        If the sweep command fails or returns non-zero exit.
    """
    cmd = [
        sys.executable,
        "-m",
        "foundry_x.evolution.cli",
        "sweep",
        "--quantizations",
        ",".join(quantizations),
        "--harness-dir",
        str(harness_dir),
        "--baseline",
        baseline,
        "--regression-threshold",
        str(regression_threshold),
        "--output",
        str(output_path),
    ]

    env = os.environ.copy()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"foundry-sweep failed with exit code {result.returncode}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    if not output_path.exists():
        raise RuntimeError(
            f"foundry-sweep exited 0 but did not write output to {output_path}.\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    with open(output_path) as f:
        return json.load(f)


def _compute_sweep_results(
    verdict: dict[str, Any],
    q8_0_pass_rate: float | None = None,
) -> list[V4QuantizationSweepResult]:
    """Compute ``V4QuantizationSweepResult`` list from a sweep verdict.

    Parameters
    ----------
    verdict:
        Parsed JSON verdict from ``foundry-sweep sweep``.
    q8_0_pass_rate:
        Explicit Q8_0 baseline pass rate. If not provided, extracted from verdict.

    Returns
    -------
    list[V4QuantizationSweepResult]
        One result per quantization in the sweep.
    """
    results: list[V4QuantizationSweepResult] = []

    quantization_results = verdict.get("quantizations", [])
    if not isinstance(quantization_results, list):
        quantization_results = []

    q8_0_rate = q8_0_pass_rate
    if q8_0_rate is None:
        for qr in quantization_results:
            if qr.get("quantization") == "Q8_0":
                q8_0_rate = qr.get("pass_rate", 0.0)
                break
        if q8_0_rate is None:
            q8_0_rate = 1.0

    q5_km_rate: float | None = None
    for qr in quantization_results:
        if qr.get("quantization") == "Q5_K_M":
            q5_km_rate = qr.get("pass_rate", 0.0)
            break

    for qr in quantization_results:
        quant = qr.get("quantization", "unknown")
        pass_rate = qr.get("pass_rate", 0.0)

        vrams_gb = _V4_VRAM_GB.get(quant, 0.0)
        regression_vs_baseline = (pass_rate - q8_0_rate) * 100

        within_2pp_of_q5km = False
        if q5_km_rate is not None:
            within_2pp_of_q5km = abs(pass_rate - q5_km_rate) <= 0.02

        recommended = False
        notes = ""

        if quant in _V4_QUANTIZATIONS:
            if within_2pp_of_q5km and vrams_gb > 0:
                if q5_km_rate is not None and vrams_gb < (_V4_VRAM_GB.get("Q5_K_M", 5.5)):
                    recommended = True
                    notes = "IQ quant at lower VRAM than Q5_K_M; within 2pp of Q5_K_M"
                elif vrams_gb > 0 and pass_rate >= q8_0_rate - (_REGRESSION_THRESHOLD_PP / 100):
                    recommended = True
                    notes = f"IQ quant within {_REGRESSION_THRESHOLD_PP}pp of Q8_0"
            elif pass_rate < q8_0_rate - (_REGRESSION_THRESHOLD_PP / 100):
                notes = f"Fails regression threshold ({_REGRESSION_THRESHOLD_PP}pp vs Q8_0)"
        elif quant == "Q5_K_M":
            recommended = True
            notes = "Current recommended floor"
        elif quant == "Q8_0":
            notes = "Full-precision baseline"

        results.append(
            V4QuantizationSweepResult(
                quantization=quant,
                pass_rate=pass_rate,
                vrams_gb=vrams_gb,
                regression_vs_baseline=regression_vs_baseline,
                within_2pp_of_q5km=within_2pp_of_q5km,
                recommended=recommended,
                notes=notes,
            )
        )

    return results


@pytest.mark.benchmark
def test_quantization_v4_sweep_run(
    benchmark_workspace: Path,
) -> None:
    """Run the v4 quantization sweep across all configured quantizations.

    Executes ``foundry-sweep sweep`` with v4 IQ quantizations and v3
    K-quant baselines, then parses the verdict into structured
    ``V4QuantizationSweepResult`` entries for KPI consumption.

    This test is marked ``@pytest.mark.benchmark`` and requires:
    - GPU hardware (RX 6600 XT or equivalent)
    - GGUF v4 model files in ``FOUNDRY_MODEL_PATH``
    - The harness directory at the repo root ``harness/``

    On success, results are appended to the module-level ``_SWEEP_RESULTS``
    list for the aggregation test to consume.
    """
    model_path = os.environ.get("FOUNDRY_MODEL_PATH")
    if not model_path:
        pytest.skip("FOUNDRY_MODEL_PATH not set; cannot run v4 quantization sweep")

    harness_dir = Path("harness")
    if not harness_dir.exists():
        pytest.skip(f"Harness directory {harness_dir} does not exist")

    output_path = benchmark_workspace / "sweep_v4.json"

    try:
        verdict = _run_foundry_sweep(
            quantizations=_ALL_SWEEP_QUANTIZATIONS,
            harness_dir=harness_dir,
            output_path=output_path,
            baseline="Q8_0",
            regression_threshold=_REGRESSION_THRESHOLD_PP,
        )
    except RuntimeError as exc:
        pytest.fail(str(exc))

    results = _compute_sweep_results(verdict)
    _SWEEP_RESULTS.clear()
    _SWEEP_RESULTS.extend(results)


@pytest.mark.benchmark
def test_quantization_v4_sweep_aggregate() -> None:
    """Aggregate v4 quantization sweep results and assert quality/VRAM tradeoffs.

    Reads the module-level ``_SWEEP_RESULTS`` list (populated by
    ``test_quantization_v4_sweep_run``) and validates the key acceptance
    criterion from ADR-0020 §Open Question #7:

    Does IQ4_XS match Q5_K_M within 2 pp?

    If so, IQ4_XS (~4.1 GB) would be recommended as a better VRAM/quality
    tradeoff than Q5_K_M (~5.5 GB) for 8 GB card operators.

    Also validates that the Q5_K_M baseline is within 2 pp of Q8_0 (the
    existing regression threshold).
    """
    if not _SWEEP_RESULTS:
        pytest.skip("No sweep results available; run test_quantization_v4_sweep_run first")

    result_by_quant = {r.quantization: r for r in _SWEEP_RESULTS}

    q8_0 = result_by_quant.get("Q8_0")
    q5_km = result_by_quant.get("Q5_K_M")
    iq4_xs = result_by_quant.get("IQ4_XS")

    assert q8_0 is not None, "Q8_0 baseline must be present in sweep results"
    assert q5_km is not None, "Q5_K_M baseline must be present in sweep results"

    assert q5_km.pass_rate >= q8_0.pass_rate - (_REGRESSION_THRESHOLD_PP / 100), (
        f"Q5_K_M must be within {_REGRESSION_THRESHOLD_PP} pp of Q8_0 baseline; "
        f"got Q8_0={q8_0.pass_rate:.3f}, Q5_K_M={q5_km.pass_rate:.3f}"
    )

    if iq4_xs is not None:
        delta_vs_q5km = (iq4_xs.pass_rate - q5_km.pass_rate) * 100
        assert abs(delta_vs_q5km) <= 2.0, (
            f"IQ4_XS must be within 2 pp of Q5_K_M (ADR-0020 Open Question #7); "
            f"got IQ4_XS={iq4_xs.pass_rate:.3f} ({iq4_xs.pass_rate * 100:.1f}%), "
            f"Q5_K_M={q5_km.pass_rate:.3f} ({q5_km.pass_rate * 100:.1f}%), "
            f"delta={delta_vs_q5km:+.1f} pp"
        )

        if iq4_xs.vrams_gb > 0 and q5_km.vrams_gb > 0:
            assert iq4_xs.vrams_gb < q5_km.vrams_gb, (
                f"IQ4_XS must use less VRAM than Q5_K_M to be a better tradeoff; "
                f"got IQ4_XS={iq4_xs.vrams_gb} GB, Q5_K_M={q5_km.vrams_gb} GB"
            )

    for r in _SWEEP_RESULTS:
        assert 0.0 <= r.pass_rate <= 1.0, (
            f"pass_rate must be between 0.0 and 1.0 for {r.quantization}; got {r.pass_rate}"
        )
        assert r.regression_vs_baseline >= -100.0, (
            f"regression_vs_baseline cannot be worse than -100% for {r.quantization}; "
            f"got {r.regression_vs_baseline}"
        )
