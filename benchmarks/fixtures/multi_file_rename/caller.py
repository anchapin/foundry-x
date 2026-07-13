"""Caller module for the multi_file_rename benchmark (seeded/broken).

Imports ``callee`` and calls ``callee.calculate_value()`` -- the NEW
name.  Because the callee still defines the OLD name ``compute_value``,
running ``python -m caller`` raises ``AttributeError``.

This file is the *seeded* (broken) state.  The golden import update
lives in ``benchmarks/tasks/test_multi_file_rename.py``
(``GOLDEN_CALLER``).
"""

import callee


def main() -> None:
    """Print the value produced by the callee (new name, not yet defined)."""
    print(callee.calculate_value())


if __name__ == "__main__":
    main()
