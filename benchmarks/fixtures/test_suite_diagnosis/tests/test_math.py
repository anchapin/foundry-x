"""Tests for the pkg.math module."""

from pkg.math import add, mul


def test_add() -> None:
    assert add(2, 2) == 5


def test_mul() -> None:
    assert mul(3, 4) == 12
