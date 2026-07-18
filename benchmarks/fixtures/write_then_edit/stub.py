"""Stub module for write_then_edit benchmark.

The compute() function is a stub: it returns the wrong formula (add instead of
multiply), causing a test expectation to fail.  The golden fix edits only the
compute() body, leaving this docstring and the if __name__ guard intact.
"""


def compute(a: int, b: int) -> int:
    """Return the product of a and b."""
    return a + b  # BUG: should be a * b


if __name__ == "__main__":
    result = compute(3, 4)
    print(f"compute(3, 4) = {result}")
