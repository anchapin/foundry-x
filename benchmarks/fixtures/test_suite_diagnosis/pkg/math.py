"""Arithmetic helpers for the test_suite_diagnosis benchmark."""


def add(a: int, b: int) -> int:
    """Return the sum of *a* and *b*."""
    return a + b


def mul(a: int, b: int) -> int:
    """Return the product of *a* and *b*."""
    return a * b


if __name__ == "__main__":
    print(add(2, 2))
    print(mul(3, 4))
