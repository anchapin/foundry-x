"""Pearson correlation for paired binary pass/fail observations (issue #900).

This module is the math layer of the external-eval validation study
(ADR-0023). It implements the correlation computation that decides
whether the internal benchmark suite under ``benchmarks/tasks/`` is a
faithful proxy for real-world coding capability as measured by
HumanEval+ / SWE-bench.

Why pure functions
------------------
The correlation math must be testable without a live model endpoint
(the real-model run is orchestrated by
``infra/scripts/run_external_eval.sh``). Keeping this layer pure -- no
I/O, no pytest, no Runner -- lets the unit tests under ``tests/`` assert
the math directly and lets the benchmark task under
``benchmarks/tasks/test_external_eval_correlation.py`` validate the
plumbing deterministically.

Methodology (per-configuration)
-------------------------------
ADR-0023 settles on a *per-configuration* methodology because the
natural unit of observation is a full agent run, not a single task. For
each agent configuration *k* (model x quantization x harness variant):

- ``internal_pass_rate_k`` = (# internal tasks passed) / N_internal
- ``external_pass_rate_k`` = (# external tasks passed) / N_external

Pearson correlation is then computed across the *K* configurations.
Issue #900 requires K >= 30 for the study to be reportable.

Edge cases the ADR explicitly enumerates
----------------------------------------
- **Fewer than 30 paired observations** -> the study is under-powered
  and :func:`pearson_binary` raises :class:`UnderpoweredStudyError`
  rather than returning a misleadingly confident number.
- **Zero variance in either series** (e.g., the agent passes every
  internal task regardless of configuration, so the internal series is
  all-ones) -> Pearson is mathematically undefined (``0/0``); the
  function raises :class:`ZeroVarianceError` with a message naming the
  offending series so the operator can choose a more discriminating
  task set.

Both errors are surfaced, never swallowed (AGENTS.md S2).
"""

from __future__ import annotations

from collections.abc import Sequence


class UnderpoweredStudyError(ValueError):
    """Raised when the study has fewer than the required paired observations.

    Issue #900 acceptance criterion 2 requires >=30 paired observations.
    Returning a Pearson number computed on fewer pairs would overstate
    the statistical confidence of the study; raising forces the operator
    to add more configurations before reporting a result.
    """


class ZeroVarianceError(ValueError):
    """Raised when one of the input series has zero variance.

    Pearson correlation divides by the product of the two standard
    deviations; if either is zero the result is mathematically
    undefined. Naming the offending series in the error message turns a
    silent ``NaN`` into a concrete operational signal: the task set is
    not discriminating enough to measure correlation against.
    """


#: Minimum number of paired observations required for a reportable
#: Pearson correlation, per issue #900 acceptance criterion 2. Spelled
#: out as a named constant so error messages and ADRs cite a single
#: source of truth.
MIN_PAIRED_OBSERVATIONS: int = 30


def pearson_binary(
    internal_rates: Sequence[float],
    external_rates: Sequence[float],
    *,
    min_pairs: int = MIN_PAIRED_OBSERVATIONS,
) -> float:
    """Compute Pearson correlation between two paired series.

    Args:
        internal_rates: Per-configuration internal pass rates in ``[0.0, 1.0]``.
        external_rates: Per-configuration external pass rates in ``[0.0, 1.0]``.
            Must be the same length as ``internal_rates``.
        min_pairs: Minimum number of paired observations required for
            the result to be considered reportable. Defaults to
            :data:`MIN_PAIRED_OBSERVATIONS` (30, per issue #900).
            Override only in unit tests that need to assert the math on
            small synthetic inputs.

    Returns:
        The Pearson product-moment correlation coefficient in ``[-1.0, 1.0]``.

    Raises:
        ValueError: If the two series differ in length or are empty.
        UnderpoweredStudyError: If the number of paired observations is
            below ``min_pairs``.
        ZeroVarianceError: If either series has zero variance (Pearson
            is undefined in that case).
    """
    n = len(internal_rates)
    if n != len(external_rates):
        raise ValueError(
            f"internal_rates and external_rates must have equal length; "
            f"got {n} and {len(external_rates)}"
        )
    if n == 0:
        raise ValueError("cannot compute Pearson correlation on empty series")
    if n < min_pairs:
        raise UnderpoweredStudyError(
            f"study has {n} paired observations but issue #900 requires >= {min_pairs}; "
            "add more agent configurations (model x quantization x harness variant) "
            "before reporting a correlation."
        )

    xs = [float(x) for x in internal_rates]
    ys = [float(y) for y in external_rates]

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denom_x = sum((x - mean_x) ** 2 for x in xs)
    denom_y = sum((y - mean_y) ** 2 for y in ys)

    if denom_x == 0.0:
        raise ZeroVarianceError(
            "internal_rates has zero variance (every configuration produced the "
            "same internal pass rate); Pearson correlation is undefined. Choose "
            "a more discriminating internal task set."
        )
    if denom_y == 0.0:
        raise ZeroVarianceError(
            "external_rates has zero variance (every configuration produced the "
            "same external pass rate); Pearson correlation is undefined. Choose "
            "a more discriminating external task set."
        )

    return num / (denom_x * denom_y) ** 0.5


def interpret_correlation(coefficient: float) -> str:
    """Return a one-line plain-English reading of a correlation coefficient.

    The thresholds match those encoded in ADR-0023:

    - ``>= 0.7``  -> ``"valid proxy"`` -- internal suite is a faithful
      proxy for external coding capability per issue #900 criterion 3.
    - ``0.3 <= r < 0.7`` -> ``"weak proxy"`` -- directionally useful but
      not reportable as a validation; identify uncorrelated task
      families (issue #900 criterion 4).
    - ``r < 0.3``  -> ``"invalid proxy"`` -- recommend external framework
      adoption per ADR-0005.

    The function is pure so the ADR, the CLI, and the test suite share
    one mapping from number to verdict.
    """
    if coefficient >= 0.7:
        return "valid proxy"
    if coefficient >= 0.3:
        return "weak proxy"
    return "invalid proxy"


__all__ = [
    "MIN_PAIRED_OBSERVATIONS",
    "UnderpoweredStudyError",
    "ZeroVarianceError",
    "interpret_correlation",
    "pearson_binary",
]
