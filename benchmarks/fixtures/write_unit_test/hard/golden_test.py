import pytest
from target import calculator


def test_calculator_add():
    assert calculator("2 + 3") == 5.0


def test_calculator_subtract():
    assert calculator("10 - 4") == 6.0


def test_calculator_multiply():
    assert calculator("3 * 4") == 12.0


def test_calculator_divide():
    assert calculator("10 / 2") == 5.0


def test_calculator_division_by_zero():
    assert calculator("5 / 0") is None


def test_calculator_invalid_expression():
    with pytest.raises(ValueError, match="Invalid expression"):
        calculator("not an expression")


def test_calculator_wrong_arity():
    with pytest.raises(ValueError, match="Invalid expression"):
        calculator("1 + 2 + 3")


def test_calculator_unknown_operator():
    with pytest.raises(ValueError, match="Invalid expression"):
        calculator("1 ^ 2")


def test_calculator_negative_result():
    assert calculator("3 - 8") == -5.0


def test_calculator_float_result():
    result = calculator("10 / 4")
    assert abs(result - 2.5) < 0.0001
