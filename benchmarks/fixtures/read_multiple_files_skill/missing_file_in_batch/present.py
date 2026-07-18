"""Helper used by the missing_file_in_batch benchmark case."""


def add(a: int, b: int) -> int:
    """Return the sum of two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Return the product of two integers."""
    return a * b


if __name__ == "__main__":
    print(add(2, 3))
    print(multiply(4, 5))
