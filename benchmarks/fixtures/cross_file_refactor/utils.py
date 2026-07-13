"""Utility functions for string manipulation.

This is the *seeded* (initial) state for the cross_file_refactor benchmark.
Both files use the OLD function name ``normalize_string``.  The agent must
rename it to ``sanitize_string`` and update all call sites.
"""


def normalize_string(text: str) -> str:
    """Return a trimmed, lowercased copy of *text* stripped of punctuation.

    Rename target: ``sanitize_string``.
    """
    import re

    text = text.strip().lower()
    return re.sub(r"[^\w\s]", "", text)
