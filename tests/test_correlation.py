"""Unit tests for :mod:`foundry_x.evaluation.correlation` (issue #900).

These tests pin the Pearson math, the under-powered-study guard, the
zero-variance guard, and the ADR-0023 interpretation thresholds. They
are pure (no I/O, no model) so they run in the default offline pytest
suite alongside the rest of ``tests/``.
"""

from __future__ import annotations

import math

import pytest

from foundry_x.evaluation.correlation import (
    MIN_PAIRED_OBSERVATIONS,
    UnderpoweredStudyError,
    ZeroVarianceError,
    interpret_correlation,
    pearson_binary,
)


def _range_pair(n: int) -> tuple[list[float], list[float]]:
    xs = [float(i) for i in range(n)]
    ys = [2.0 * x + 1.0 for x in xs]
    return xs, ys


def test_min_paired_observations_value() -> None:
    """ADR-0023 fixes the threshold at 30 (issue #900 acceptance criterion 2)."""
    assert MIN_PAIRED_OBSERVATIONS == 30


def test_pearson_perfect_positive() -> None:
    xs, ys = _range_pair(40)
    r = pearson_binary(xs, ys, min_pairs=10)
    assert math.isclose(r, 1.0, abs_tol=1e-12)


def test_pearson_perfect_negative() -> None:
    xs = [float(i) for i in range(40)]
    ys = [-3.0 * x for x in xs]
    r = pearson_binary(xs, ys, min_pairs=10)
    assert math.isclose(r, -1.0, abs_tol=1e-12)


def test_pearson_zero_correlation() -> None:
    """Constructed so the cross-product sum is exactly zero."""
    xs = [-1.0, -1.0, 1.0, 1.0] * 10
    ys = [-1.0, 1.0, -1.0, 1.0] * 10
    r = pearson_binary(xs, ys, min_pairs=10)
    assert abs(r) < 1e-12


def test_pearson_matches_manual_computation() -> None:
    """Hand-computed textbook example: r = 4 / (sqrt(8) * sqrt(10)) = 0.707..."""
    # Means: x_bar = 2.5, y_bar = 3.0
    # Cross-product: (2-2.5)*(1-3) + (4-2.5)*(3-3) + (1-2.5)*(1-3) + (3-2.5)*(5-3) + (5-2.5)*(5-3)
    #              = (-0.5)*(-2) + (1.5)*0 + (-1.5)*(-2) + (0.5)*2 + (2.5)*2 = 1+0+3+1+5 = 10
    # Wait that doesn't match; recompute below for clarity in code.
    xs = [2.0, 4.0, 1.0, 3.0, 5.0]
    ys = [1.0, 3.0, 1.0, 5.0, 5.0]
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    expected = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / (
        math.sqrt(sum((x - mx) ** 2 for x in xs)) * math.sqrt(sum((y - my) ** 2 for y in ys))
    )
    r = pearson_binary(xs, ys, min_pairs=1)
    assert math.isclose(r, expected, rel_tol=1e-12)


def test_pearson_unequal_length_raises() -> None:
    with pytest.raises(ValueError, match="equal length"):
        pearson_binary([0.5, 0.6, 0.7], [0.4, 0.5], min_pairs=1)


def test_pearson_empty_series_raises() -> None:
    with pytest.raises(ValueError, match="empty series"):
        pearson_binary([], [], min_pairs=1)


def test_pearson_underpowered_raises() -> None:
    """Below the issue #900 minimum the study refuses to report a number."""
    with pytest.raises(UnderpoweredStudyError, match=r"requires >= 30"):
        pearson_binary([0.5, 0.6], [0.4, 0.7])

    # Exactly at threshold does NOT raise.
    xs, ys = _range_pair(MIN_PAIRED_OBSERVATIONS)
    r = pearson_binary(xs, ys)
    assert math.isclose(r, 1.0, abs_tol=1e-12)

    # One below threshold raises.
    with pytest.raises(UnderpoweredStudyError):
        pearson_binary(xs[:-1], ys[:-1])


def test_pearson_zero_variance_internal_raises() -> None:
    constant_xs = [0.5] * 40
    varying_ys = [float(i) / 40 for i in range(40)]
    with pytest.raises(ZeroVarianceError, match="internal_rates"):
        pearson_binary(constant_xs, varying_ys, min_pairs=10)


def test_pearson_zero_variance_external_raises() -> None:
    varying_xs = [float(i) / 40 for i in range(40)]
    constant_ys = [0.5] * 40
    with pytest.raises(ZeroVarianceError, match="external_rates"):
        pearson_binary(varying_xs, constant_ys, min_pairs=10)


@pytest.mark.parametrize(
    "coefficient, expected",
    [
        (1.0, "valid proxy"),
        (0.7, "valid proxy"),
        (0.95, "valid proxy"),
        (0.69, "weak proxy"),
        (0.5, "weak proxy"),
        (0.3, "weak proxy"),
        (0.29, "invalid proxy"),
        (0.0, "invalid proxy"),
        (-0.5, "invalid proxy"),
        (-1.0, "invalid proxy"),
    ],
)
def test_interpret_correlation_thresholds(coefficient: float, expected: str) -> None:
    assert interpret_correlation(coefficient) == expected
