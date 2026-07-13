"""Utility functions for string manipulation.

This is the *seeded* (initial) state for the cross_file_refactor benchmark.
The library defines ``sanitize_string`` (the NEW name).  The agent must
rename it to ``normalize_string`` and update main.py's import to match.
"""


def sanitize_string(text: str) -> str:
    import re

    text = text.strip().lower()
    return re.sub(r"[^\w\s]", "", text)
