"""Calculator module for git_status_driven_edit benchmark."""


def add(a: int, b: int) -> int:
    """Return the sum of a and b."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Return the product of a and b."""
    # BUG: missing * operator - should be a * b
    return a + a


if __name__ == "__main__":
    # Verify both functions work correctly
    assert add(2, 3) == 5, "add failed"
    assert multiply(3, 4) == 12, f"multiply failed: expected 12, got {multiply(3, 4)}"
    print("All tests passed")
