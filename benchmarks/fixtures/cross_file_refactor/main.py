"""CLI entry point that uses the string utility functions.

This file imports and calls ``normalize_string`` from ``utils``.  After
the agent renames the function to ``sanitize_string`` in ``utils.py``, this
file must also be updated to call the new name.  Both files must agree on
the new name for ``python main.py`` to run without import or attribute
errors.
"""

from utils import normalize_string


def main() -> None:
    """Print the sanitized version of a sample string."""
    sample = "  Hello, World!  "
    result = normalize_string(sample)
    print(f"result={result}")


if __name__ == "__main__":
    main()
